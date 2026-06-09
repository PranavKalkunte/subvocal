"""Concrete implementations of the LLMProvider interface using urllib.
"""

import os
import json
import time
import urllib.request
import urllib.error
from typing import List, Optional, Any, Dict

from subvocal.context.schema import UserContext
from .models import CommandToken, Intent
from .interfaces import LLMProvider
from .prompts import PromptManager


class BaseLLMProvider(LLMProvider):
    """Base provider containing prompt formatting and request helpers."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        mock_response: Optional[str] = None,
        prompt_version: str = "v1",
    ):
        """Initializes the base LLM provider.

        Args:
            api_key: API authorization key. Resolves from environment if None.
            model_name: Name of the LLM model to request.
            mock_response: Optional mock string response to bypass HTTP calls (for testing).
            prompt_version: The version string of the prompt template to use.
        """
        self.api_key = api_key
        self.model_name = model_name
        self.mock_response = mock_response
        self.prompt_version = prompt_version
        self._prompt_mgr = PromptManager(default_version=prompt_version)

    def _get_api_key(self, env_var: str) -> str:
        key = self.api_key or os.environ.get(env_var)
        if not key and not self.mock_response:
            raise ValueError(
                f"Missing API key. Please specify it in the constructor or set the environment variable '{env_var}'."
            )
        return key or ""

    def _execute_http_post(self, url: str, data: bytes, headers: Dict[str, str]) -> str:
        """Sends an HTTP POST request synchronously using urllib."""
        if self.mock_response is not None:
            return self.mock_response

        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                return response.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8") if e.fp else ""
            raise RuntimeError(f"HTTP request to {url} failed with status {e.code}: {e.reason}. Detail: {err_body}")
        except Exception as e:
            raise RuntimeError(f"Failed to connect to {url}: {e}")

    def _format_context(self, context: UserContext) -> Dict[str, str]:
        """Utility to format structured context into raw strings for prompts."""
        contacts_str = ", ".join([f"{c.name} ({c.shorthand_name})" for c in context.contacts])
        calendar_str = ", ".join(
            [f"{e.title} at {e.start_time} ({e.shorthand_title})" for e in context.calendar]
        )
        web_elements_str = ", ".join(
            [f"{el.label} [{el.element_type}] ({el.shorthand_label})" for el in context.app_state.visible_elements]
        )
        history_str = "\n".join([f"{m.role}: {m.text}" for m in context.conversation_history])

        return {
            "contacts": contacts_str or "(none)",
            "calendar": calendar_str or "(none)",
            "web_context": web_elements_str or "(none)",
            "history": history_str or "(none)",
        }

    def _parse_llm_output(self, output: str, raw_shorthand: str, default_confidence: float = 0.95) -> Intent:
        """Parses the LLM plain text output into a validated Intent."""
        cleaned = output.replace('"', '').replace("'", "").strip()
        parts = cleaned.split()
        if not parts:
            return Intent(
                command="UNKNOWN",
                arguments=[],
                confidence=0.0,
                resolved_text=cleaned,
                raw_shorthand=raw_shorthand,
                timestamp=time.time()
            )

        command = parts[0].upper()
        arguments = parts[1:]

        return Intent(
            command=command,
            arguments=arguments,
            confidence=default_confidence,
            resolved_text=cleaned,
            raw_shorthand=raw_shorthand,
            timestamp=time.time()
        )


class ClaudeProvider(BaseLLMProvider):
    """Anthropic Claude LLM provider client."""

    def get_provider_name(self) -> str:
        return "anthropic"

    def reconstruct_intent(self, tokens: List[CommandToken], context: UserContext) -> Intent:
        key = self._get_api_key("ANTHROPIC_API_KEY")
        model = self.model_name or "claude-3-5-sonnet-20241022"

        raw_shorthand = " ".join([t.text for t in tokens])
        mean_conf = sum(t.confidence for t in tokens) / len(tokens) if tokens else 0.95
        from subvocal.shorthand.decoder import heuristic_decode_phrase
        heur_phrase, _ = heuristic_decode_phrase(
            raw_shorthand,
            ui_elements=[el.label for el in context.app_state.visible_elements],
            contacts=[c.name for c in context.contacts],
            calendar_events=[e.title for e in context.calendar]
        )

        formatted_ctx = self._format_context(context)
        prompt = self._prompt_mgr.format_prompt(
            version=self.prompt_version,
            noisy_input=raw_shorthand,
            heuristic_recommendation=heur_phrase,
            **formatted_ctx
        )

        # Prepare request payload
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        body = json.dumps({
            "model": model,
            "max_tokens": 100,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0
        }).encode("utf-8")

        response_str = self._execute_http_post(url, body, headers)

        # Extract message content
        if self.mock_response is not None:
            output = response_str
        else:
            res_json = json.loads(response_str)
            output = res_json["content"][0]["text"]

        return self._parse_llm_output(output, raw_shorthand, mean_conf)


class OpenAIProvider(BaseLLMProvider):
    """OpenAI GPT completions provider client."""

    def get_provider_name(self) -> str:
        return "openai"

    def reconstruct_intent(self, tokens: List[CommandToken], context: UserContext) -> Intent:
        key = self._get_api_key("OPENAI_API_KEY")
        model = self.model_name or "gpt-4o"

        raw_shorthand = " ".join([t.text for t in tokens])
        mean_conf = sum(t.confidence for t in tokens) / len(tokens) if tokens else 0.95
        from subvocal.shorthand.decoder import heuristic_decode_phrase
        heur_phrase, _ = heuristic_decode_phrase(
            raw_shorthand,
            ui_elements=[el.label for el in context.app_state.visible_elements],
            contacts=[c.name for c in context.contacts],
            calendar_events=[e.title for e in context.calendar]
        )

        formatted_ctx = self._format_context(context)
        prompt = self._prompt_mgr.format_prompt(
            version=self.prompt_version,
            noisy_input=raw_shorthand,
            heuristic_recommendation=heur_phrase,
            **formatted_ctx
        )

        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json"
        }
        body = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 100
        }).encode("utf-8")

        response_str = self._execute_http_post(url, body, headers)

        if self.mock_response is not None:
            output = response_str
        else:
            res_json = json.loads(response_str)
            output = res_json["choices"][0]["message"]["content"]

        return self._parse_llm_output(output, raw_shorthand, mean_conf)


class GeminiProvider(BaseLLMProvider):
    """Google Gemini content generation provider client."""

    def get_provider_name(self) -> str:
        return "gemini"

    def reconstruct_intent(self, tokens: List[CommandToken], context: UserContext) -> Intent:
        key = self._get_api_key("GEMINI_API_KEY")
        model = self.model_name or "gemini-1.5-flash"

        raw_shorthand = " ".join([t.text for t in tokens])
        mean_conf = sum(t.confidence for t in tokens) / len(tokens) if tokens else 0.95
        from subvocal.shorthand.decoder import heuristic_decode_phrase
        heur_phrase, _ = heuristic_decode_phrase(
            raw_shorthand,
            ui_elements=[el.label for el in context.app_state.visible_elements],
            contacts=[c.name for c in context.contacts],
            calendar_events=[e.title for e in context.calendar]
        )

        formatted_ctx = self._format_context(context)
        prompt = self._prompt_mgr.format_prompt(
            version=self.prompt_version,
            noisy_input=raw_shorthand,
            heuristic_recommendation=heur_phrase,
            **formatted_ctx
        )

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
        headers = {
            "Content-Type": "application/json"
        }
        body = json.dumps({
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.0}
        }).encode("utf-8")

        response_str = self._execute_http_post(url, body, headers)

        if self.mock_response is not None:
            output = response_str
        else:
            res_json = json.loads(response_str)
            output = res_json["candidates"][0]["content"]["parts"][0]["text"]

        return self._parse_llm_output(output, raw_shorthand, mean_conf)


class LlamaProvider(BaseLLMProvider):
    """Local Llama provider client (running via Ollama)."""

    def __init__(
        self,
        ollama_host: Optional[str] = None,
        model_name: Optional[str] = None,
        mock_response: Optional[str] = None,
        prompt_version: str = "v1",
    ):
        """Initializes the Ollama Llama provider client."""
        super().__init__(api_key="none", model_name=model_name, mock_response=mock_response, prompt_version=prompt_version)
        self.ollama_host = ollama_host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434"

    def get_provider_name(self) -> str:
        return "llama"

    def reconstruct_intent(self, tokens: List[CommandToken], context: UserContext) -> Intent:
        model = self.model_name or "llama3"

        raw_shorthand = " ".join([t.text for t in tokens])
        mean_conf = sum(t.confidence for t in tokens) / len(tokens) if tokens else 0.95
        from subvocal.shorthand.decoder import heuristic_decode_phrase
        heur_phrase, _ = heuristic_decode_phrase(
            raw_shorthand,
            ui_elements=[el.label for el in context.app_state.visible_elements],
            contacts=[c.name for c in context.contacts],
            calendar_events=[e.title for e in context.calendar]
        )

        formatted_ctx = self._format_context(context)
        prompt = self._prompt_mgr.format_prompt(
            version=self.prompt_version,
            noisy_input=raw_shorthand,
            heuristic_recommendation=heur_phrase,
            **formatted_ctx
        )

        # Ollama exposes OpenAI-compatible endpoint
        url = f"{self.ollama_host.rstrip('/')}/v1/chat/completions"
        headers = {
            "Content-Type": "application/json"
        }
        body = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 100
        }).encode("utf-8")

        response_str = self._execute_http_post(url, body, headers)

        if self.mock_response is not None:
            output = response_str
        else:
            res_json = json.loads(response_str)
            output = res_json["choices"][0]["message"]["content"]

        return self._parse_llm_output(output, raw_shorthand, mean_conf)
