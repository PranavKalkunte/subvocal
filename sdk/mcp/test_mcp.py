"""Unit and integration tests for Model Context Protocol (MCP) server."""

import os
import sys
import json
import time
import unittest

# Add package paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.server import SubvocalMCPServer
from core.models import CommandToken


class TestSubvocalMCPServer(unittest.TestCase):

    def setUp(self):
        """Initialize server instance and complete MCP handshake before each test."""
        self.server = SubvocalMCPServer()
        self.server.handle_request({"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {}})
        self.server.handle_request({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def test_01_initialize_handshake(self):
        """Test initialize handshake request and notification lifecycle on a fresh server."""
        # Use a fresh server so the assertions are not trivially satisfied by setUp
        server = SubvocalMCPServer()
        self.assertFalse(server.initialized)

        # 1. Send initialize request
        req = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "clientInfo": {"name": "TestClient", "version": "1.0.0"}
            }
        }
        resp = server.handle_request(req)
        self.assertIsNotNone(resp)
        self.assertEqual(resp["jsonrpc"], "2.0")
        self.assertEqual(resp["id"], 1)
        self.assertEqual(resp["result"]["protocolVersion"], "2025-03-26")
        self.assertIn("tools", resp["result"]["capabilities"])
        self.assertIn("resources", resp["result"]["capabilities"])
        self.assertFalse(server.initialized)  # notification not sent yet

        # 2. Complete initialized notification
        notif = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        }
        resp_notif = server.handle_request(notif)
        self.assertIsNone(resp_notif)
        self.assertTrue(server.initialized)

    def test_02_tools_list(self):
        """Test tools discovery schema properties."""
        req = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list"
        }
        resp = self.server.handle_request(req)
        self.assertIsNotNone(resp)
        self.assertEqual(resp["id"], 2)
        tools = resp["result"]["tools"]
        tool_names = [t["name"] for t in tools]
        self.assertIn("get_pipeline_status", tool_names)
        self.assertIn("get_buffer", tool_names)
        self.assertIn("inject_mock_token", tool_names)
        self.assertIn("process_phrase", tool_names)
        self.assertIn("trigger_calibration", tool_names)

    def test_03_pipeline_status_tool(self):
        """Test get_pipeline_status tool call response."""
        req = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "get_pipeline_status",
                "arguments": {}
            }
        }
        resp = self.server.handle_request(req)
        self.assertIsNotNone(resp)
        content = resp["result"]["content"][0]
        self.assertEqual(content["type"], "text")
        self.assertIn("Subvocal Pipeline Status", content["text"])

    def test_04_token_injection_and_processing(self):
        """Test inject_mock_token and process_phrase execution flow."""
        # 1. Inject token 'clk'
        req_inject = {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "inject_mock_token",
                "arguments": {"token": "clk", "confidence": 0.98}
            }
        }
        resp_inject = self.server.handle_request(req_inject)
        self.assertIsNotNone(resp_inject)
        self.assertIn("Successfully injected token", resp_inject["result"]["content"][0]["text"])
        self.assertEqual(len(self.server.pipeline.token_buffer), 1)
        self.assertEqual(self.server.pipeline.token_buffer[0].text, "clk")

        # 2. Read token buffer
        req_buf = {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {
                "name": "get_buffer",
                "arguments": {}
            }
        }
        resp_buf = self.server.handle_request(req_buf)
        self.assertIsNotNone(resp_buf)
        buf_data = json.loads(resp_buf["result"]["content"][0]["text"])
        self.assertEqual(len(buf_data), 1)
        self.assertEqual(buf_data[0]["text"], "clk")

        # 3. Process phrase
        req_proc = {
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/call",
            "params": {
                "name": "process_phrase",
                "arguments": {}
            }
        }
        resp_proc = self.server.handle_request(req_proc)
        self.assertIsNotNone(resp_proc)
        text = resp_proc["result"]["content"][0]["text"]
        self.assertIn("Reconstructed and executed action", text)
        self.assertEqual(len(self.server.pipeline.token_buffer), 0)
        self.assertEqual(len(self.server.executor.history), 1)

    def test_05_resources_handling(self):
        """Test resource listing and reading."""
        # 1. List resources
        req_list = {
            "jsonrpc": "2.0",
            "id": 7,
            "method": "resources/list"
        }
        resp_list = self.server.handle_request(req_list)
        self.assertIsNotNone(resp_list)
        resources = resp_list["result"]["resources"]
        self.assertEqual(len(resources), 1)
        self.assertEqual(resources[0]["uri"], "subvocal://intent/history")

        # 2. Add an action to history and read resource contents
        self.server.pipeline.token_buffer.append(CommandToken(text="gt", confidence=0.9, timestamp=time.time()))
        self.server.pipeline.process_phrase()
        
        req_read = {
            "jsonrpc": "2.0",
            "id": 8,
            "method": "resources/read",
            "params": {
                "uri": "subvocal://intent/history"
            }
        }
        resp_read = self.server.handle_request(req_read)
        self.assertIsNotNone(resp_read)
        content = resp_read["result"]["contents"][0]
        self.assertEqual(content["mimeType"], "application/json")
        history = json.loads(content["text"])
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["action_type"], "goto")


if __name__ == "__main__":
    unittest.main()
