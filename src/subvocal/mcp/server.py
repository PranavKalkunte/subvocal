"""Model Context Protocol (MCP) reference server for Subvocal Middleware."""

import json
import sys
import threading
import time
from typing import Any

from subvocal import __version__ as _SDK_VERSION
from subvocal.context.schema import AppState, UserContext
from subvocal.core.interfaces import ActionExecutor, ContextProvider
from subvocal.core.llm_providers import HeuristicProvider
from subvocal.core.models import Action, CommandToken
from subvocal.core.pipeline import SubvocalPipeline

# ══════════════════════════════════════════════════════════════════════════════
# Mock Implementations for Reference Running
# ══════════════════════════════════════════════════════════════════════════════

class MockContextProvider(ContextProvider):
    """Exposes mock system context state."""
    def get_context(self) -> UserContext:
        return UserContext(
            app_state=AppState(
                current_app="Claude Desktop",
                page_url="https://modelcontextprotocol.io",
            )
        )

    def get_provider_name(self) -> str:
        return "mock_mcp_context"


class MockActionExecutor(ActionExecutor):
    """Executes actions and logs execution history."""
    def __init__(self):
        self.history: list[Action] = []

    def execute(self, action: Action) -> Any:
        self.history.append(action)
        return {"status": "SUCCESS", "action_id": action.intent_id}

    def can_execute(self, action: Action) -> bool:  # noqa: ARG002
        return True


# ══════════════════════════════════════════════════════════════════════════════
# MCP Server Implementation
# ══════════════════════════════════════════════════════════════════════════════

class SubvocalMCPServer:
    """Zero-dependency Model Context Protocol server communicating via stdio."""

    def __init__(self):
        # 1. Initialize Pipeline dependencies
        from subvocal.hardware.drivers import SyntheticSignalGenerator
        self.hardware = SyntheticSignalGenerator()
        
        # Load pre-trained classifier if available
        try:
            from subvocal.emg_core.ml.infer import InferenceEngine
            engine = InferenceEngine(user_id="pretrained", model_type="cnn")
            self.classify_fn = lambda f: engine.predict(f)
            self.model_status = "cnn (Pre-trained)"
        except Exception:
            self.classify_fn = lambda f: None
            self.model_status = "Fallback (No Model)"

        self.llm = HeuristicProvider()
        self.context = MockContextProvider()
        self.executor = MockActionExecutor()

        self.pipeline = SubvocalPipeline(
            hardware=self.hardware,
            classify_fn=self.classify_fn,
            llm_provider=self.llm,
            context_provider=self.context,
            executor=self.executor,
            phrase_timeout_seconds=2.0
        )

        self.initialized = False

    def handle_request(self, req: dict[str, Any]) -> dict[str, Any] | None:
        """Process an incoming JSON-RPC 2.0 request and return the response dictionary."""
        method = req.get("method")
        msg_id = req.get("id")
        params = req.get("params", {})

        # Handle notification (no id)
        if msg_id is None:
            if method == "notifications/initialized":
                self.initialized = True
            return None

        # Handshake: initialize request
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {
                        "tools": {},
                        "resources": {}
                    },
                    "serverInfo": {
                        "name": "SubvocalMiddlewareMCPServer",
                        "version": _SDK_VERSION
                    }
                }
            }

        # Handle tools discovery
        if method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "tools": self._get_tools_schema()
                }
            }

        # Handle tool calling
        if method in ("tools/call", "tools/list", "resources/list", "resources/read") and not self.initialized:
            return self._error_response(msg_id, -32002, "Server not initialized. Send 'initialize' and await 'notifications/initialized' first.")

        if method == "tools/call":
            tool_name = params.get("name")
            tool_args = params.get("arguments", {})
            return self._call_tool(msg_id, tool_name, tool_args)

        # Handle resources discovery
        if method == "resources/list":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "resources": [
                        {
                            "uri": "subvocal://intent/history",
                            "name": "Intent History Log",
                            "description": "Log of recently executed subvocal command intents.",
                            "mimeType": "application/json"
                        }
                    ]
                }
            }

        # Handle reading resource contents
        if method == "resources/read":
            uri = params.get("uri")
            if uri == "subvocal://intent/history":
                history_data = [act.model_dump() for act in self.executor.history]
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "contents": [
                            {
                                "uri": uri,
                                "mimeType": "application/json",
                                "text": json.dumps(history_data, indent=2)
                            }
                        ]
                    }
                }
            else:
                return self._error_response(msg_id, -32602, f"Resource not found: {uri}")

        # Fallback for unknown methods
        return self._error_response(msg_id, -32601, f"Method not found: {method}")

    def _get_tools_schema(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "get_pipeline_status",
                "description": "Retrieve status metrics of the subvocal middleware including classifier model and active hardware source.",
                "inputSchema": {"type": "object", "properties": {}}
            },
            {
                "name": "get_buffer",
                "description": "Fetch list of currently classified command tokens queued in the active phrase buffer.",
                "inputSchema": {"type": "object", "properties": {}}
            },
            {
                "name": "inject_mock_token",
                "description": "Simulate muscle contraction signal decoding by injecting a custom token into the buffer.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "token": {"type": "string", "description": "Token value (e.g. 'clk', 'gt', 'typ')."},
                        "confidence": {"type": "number", "description": "Token confidence score (0.0 to 1.0)."}
                    },
                    "required": ["token"]
                }
            },
            {
                "name": "process_phrase",
                "description": "Force immediate decoding of all accumulated tokens in the buffer into an actionable intent and dispatch it.",
                "inputSchema": {"type": "object", "properties": {}}
            },
            {
                "name": "trigger_calibration",
                "description": "Trigger the per-user transfer learning calibration process to fine-tune a model on a new calibration set.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "string", "description": "Target user calibration dataset identifier."},
                        "model_type": {"type": "string", "description": "Base pre-trained model type (cnn, gru, transformer)."}
                    },
                    "required": ["user_id", "model_type"]
                }
            }
        ]

    def _call_tool(self, msg_id: Any, name: str, args: dict[str, Any]) -> dict[str, Any]:
        try:
            if name == "get_pipeline_status":
                status_text = (
                    f"Subvocal Pipeline Status:\n"
                    f"- Hardware Source: Connected ({self.hardware.__class__.__name__})\n"
                    f"- Active Classifier: {self.model_status}\n"
                    f"- Buffer Size: {len(self.pipeline.token_buffer)} tokens\n"
                    f"- Total Actions Dispatched: {len(self.executor.history)}"
                )
                return self._success_response(msg_id, status_text)

            elif name == "get_buffer":
                tokens = [t.model_dump() for t in self.pipeline.token_buffer]
                return self._success_response(msg_id, json.dumps(tokens, indent=2))

            elif name == "inject_mock_token":
                token_text = args.get("token")
                if not isinstance(token_text, str) or not token_text:
                    return self._error_response(msg_id, -32602, "Invalid params: 'token' must be a non-empty string")
                confidence = float(args.get("confidence", 0.95))
                token_obj = CommandToken(
                    text=token_text,
                    confidence=confidence,
                    timestamp=time.time()
                )
                self.pipeline.token_buffer.append(token_obj)
                self.pipeline._last_token_time = time.time()
                return self._success_response(msg_id, f"Successfully injected token '{token_text}' with confidence {confidence:.2f}")

            elif name == "process_phrase":
                if not self.pipeline.token_buffer:
                    return self._success_response(msg_id, "No tokens in buffer. Inject tokens before running process_phrase.")
                action = self.pipeline.process_phrase()
                if action:
                    return self._success_response(msg_id, f"Reconstructed and executed action:\n{json.dumps(action.model_dump(), indent=2)}")
                else:
                    return self._success_response(msg_id, "Pipeline triggered phrase processing, but no action was dispatched.")

            elif name == "trigger_calibration":
                user_id = args.get("user_id")
                model_type = args.get("model_type")
                if not isinstance(user_id, str) or not isinstance(model_type, str):
                    return self._error_response(msg_id, -32602, "Invalid params: 'user_id' and 'model_type' must be strings")
                # Run calibration in a background thread so the stdio loop stays responsive
                def _run_calibration(uid: str, mtype: str) -> None:
                    try:
                        # Deferred import: calibration needs the [ml] extra
                        from subvocal.emg_core.ml.train import calibrate_model

                        calibrate_model(uid, mtype)
                    except ImportError:
                        sys.stderr.write(
                            f"[calibration] {uid}/{mtype} failed: the [ml] extra is required. "
                            'Install it with: pip install "subvocal[ml]"\n'
                        )
                    except Exception as exc:
                        sys.stderr.write(f"[calibration] {uid}/{mtype} failed: {exc}\n")
                threading.Thread(target=_run_calibration, args=(user_id, model_type), daemon=True).start()
                return self._success_response(msg_id, f"Calibration started for user '{user_id}' model '{model_type}'. Running in background.")

            else:
                return self._error_response(msg_id, -32602, f"Unknown tool: {name}")

        except Exception as e:
            return self._error_response(msg_id, -32000, f"Tool execution failed: {str(e)}")

    def _success_response(self, msg_id: Any, text: str) -> dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "content": [{"type": "text", "text": text}]
            }
        }

    def _error_response(self, msg_id: Any, code: int, message: str) -> dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {
                "code": code,
                "message": message
            }
        }

    def run(self):
        """Standard input/output reader loop."""
        for line in sys.stdin:
            if not line.strip():
                continue
            try:
                req = json.loads(line)
                resp = self.handle_request(req)
                if resp:
                    sys.stdout.write(json.dumps(resp) + "\n")
                    sys.stdout.flush()
            except Exception as e:
                err_resp = self._error_response(None, -32603, f"Internal JSON-RPC parsing error: {str(e)}")
                sys.stdout.write(json.dumps(err_resp) + "\n")
                sys.stdout.flush()


def main() -> None:
    """Console entry point for the ``subvocal-mcp`` stdio server."""
    server = SubvocalMCPServer()
    server.run()


if __name__ == "__main__":
    main()
