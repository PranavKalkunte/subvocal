"""Tests for the typed exception hierarchy, HeuristicProvider, provider
retry behavior, and pipeline observability (stats, callbacks, policy raise)."""

import os
import tempfile
import time
import unittest
import unittest.mock
import urllib.error

from subvocal import (
    ConfigurationError,
    HardwareError,
    HeuristicProvider,
    MissingDependencyError,
    PolicyViolationError,
    ProviderError,
    SubvocalError,
    SubvocalPipeline,
    resolve_provider,
)
from subvocal.context.schema import UserContext
from subvocal.core.llm_providers import ClaudeProvider, OpenAIProvider
from subvocal.core.models import CommandToken, Frame, Sample
from subvocal.core.testing import (
    MockActionExecutor,
    MockContextProvider,
    MockHardwareSource,
    MockLLMProvider,
)


def _token(text="gt", conf=0.9):
    return CommandToken(text=text, confidence=conf, timestamp=time.time())


class TestExceptionHierarchy(unittest.TestCase):
    def test_all_derive_from_subvocal_error(self):
        for exc in (ConfigurationError, HardwareError, MissingDependencyError,
                    ProviderError, PolicyViolationError):
            self.assertTrue(issubclass(exc, SubvocalError))

    def test_builtin_compatibility(self):
        """Pre-existing except clauses on builtin types keep working."""
        self.assertTrue(issubclass(HardwareError, RuntimeError))
        self.assertTrue(issubclass(MissingDependencyError, ImportError))
        self.assertTrue(issubclass(ProviderError, RuntimeError))
        self.assertTrue(issubclass(ConfigurationError, ValueError))
        self.assertTrue(issubclass(PolicyViolationError, PermissionError))

    def test_hardware_raises_typed_error(self):
        from subvocal.hardware.drivers import SyntheticSignalGenerator
        gen = SyntheticSignalGenerator()
        with self.assertRaises(HardwareError):
            gen.read_frame(window_ms=10)


class TestHeuristicProvider(unittest.TestCase):
    def test_offline_reconstruction(self):
        provider = HeuristicProvider()
        intent = provider.reconstruct_intent([_token("gt"), _token("g gl")], UserContext())
        self.assertEqual(intent.command, "GOTO")
        self.assertGreater(intent.confidence, 0.0)
        self.assertEqual(provider.get_provider_name(), "heuristic")

    def test_empty_tokens_yield_unknown(self):
        provider = HeuristicProvider()
        intent = provider.reconstruct_intent([], UserContext())
        self.assertEqual(intent.command, "UNKNOWN")

    def test_min_confidence_floor(self):
        provider = HeuristicProvider(min_confidence=1.1)  # impossible floor
        intent = provider.reconstruct_intent([_token("gt")], UserContext())
        self.assertEqual(intent.command, "UNKNOWN")


class TestResolveProvider(unittest.TestCase):
    def test_falls_back_to_heuristic_without_keys(self):
        env = {k: v for k, v in os.environ.items()
               if k not in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY")}
        with unittest.mock.patch.dict(os.environ, env, clear=True):
            self.assertIsInstance(resolve_provider(), HeuristicProvider)

    def test_env_key_selects_provider(self):
        with unittest.mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "k"}):
            self.assertIsInstance(resolve_provider(), ClaudeProvider)

    def test_prefer_overrides(self):
        self.assertIsInstance(resolve_provider(prefer="openai", api_key="k"), OpenAIProvider)

    def test_unknown_prefer_raises(self):
        with self.assertRaises(ConfigurationError):
            resolve_provider(prefer="not-a-provider")


class TestProviderRetries(unittest.TestCase):
    def _provider(self, **kwargs):
        return ClaudeProvider(api_key="k", backoff_seconds=0.0, **kwargs)

    def test_missing_key_is_configuration_error(self):
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        with unittest.mock.patch.dict(os.environ, env, clear=True):
            with self.assertRaises(ConfigurationError):
                ClaudeProvider().reconstruct_intent([_token()], UserContext())

    def test_invalid_timeout_rejected(self):
        with self.assertRaises(ConfigurationError):
            ClaudeProvider(api_key="k", timeout=0)

    def test_retries_transient_then_succeeds(self):
        provider = self._provider(max_retries=2)
        calls = {"n": 0}

        def flaky(req, timeout):
            calls["n"] += 1
            if calls["n"] < 3:
                raise urllib.error.URLError("transient connection reset")
            return unittest.mock.MagicMock(
                __enter__=lambda s: unittest.mock.MagicMock(
                    read=lambda: b'{"content": [{"text": "GOTO google.com"}]}'),
                __exit__=lambda s, *a: False,
            )

        with unittest.mock.patch("urllib.request.urlopen", side_effect=flaky):
            intent = provider.reconstruct_intent([_token()], UserContext())
        self.assertEqual(calls["n"], 3)
        self.assertEqual(intent.command, "GOTO")

    def test_exhausted_retries_raise_provider_error(self):
        provider = self._provider(max_retries=1)
        with unittest.mock.patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("down"),
        ) as mocked:
            with self.assertRaises(ProviderError):
                provider.reconstruct_intent([_token()], UserContext())
        self.assertEqual(mocked.call_count, 2)  # first try + one retry

    def test_non_retryable_http_error_fails_fast(self):
        provider = self._provider(max_retries=3)
        err = urllib.error.HTTPError("u", 401, "Unauthorized", {}, None)
        with unittest.mock.patch("urllib.request.urlopen", side_effect=err) as mocked:
            with self.assertRaises(ProviderError):
                provider.reconstruct_intent([_token()], UserContext())
        self.assertEqual(mocked.call_count, 1)


class _DenyAllPolicy:
    def is_authorized(self, action, context):
        return False


class TestPipelineObservability(unittest.TestCase):
    def _pipeline(self, **kwargs):
        tmp = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
        self.addCleanup(os.remove, tmp.name)
        return SubvocalPipeline(
            hardware=MockHardwareSource(),
            classify_fn=lambda frame: _token("gt"),
            llm_provider=MockLLMProvider(),
            context_provider=MockContextProvider(),
            executor=MockActionExecutor(),
            phrase_timeout_seconds=0.01,
            trace_path=tmp.name,
            **kwargs,
        )

    def test_stats_counters(self):
        pipeline = self._pipeline()
        pipeline.hardware.start()
        pipeline.step(window_ms=10)
        action = pipeline.process_phrase()
        self.assertIsNotNone(action)
        stats = pipeline.stats.as_dict()
        self.assertEqual(stats["frames_processed"], 1)
        self.assertEqual(stats["tokens_classified"], 1)
        self.assertEqual(stats["phrases_processed"], 1)
        self.assertEqual(stats["intents_resolved"], 1)
        self.assertEqual(stats["actions_executed"], 1)
        self.assertEqual(stats["errors"], 0)
        self.assertGreaterEqual(stats["uptime_seconds"], 0.0)

    def test_callbacks_fire_in_order(self):
        events = []
        pipeline = self._pipeline(
            on_token=lambda t: events.append(("token", t.text)),
            on_intent=lambda i: events.append(("intent", i.command)),
            on_action=lambda a, status: events.append(("action", status)),
        )
        pipeline.hardware.start()
        pipeline.step(window_ms=10)
        pipeline.process_phrase()
        self.assertEqual([e[0] for e in events], ["token", "intent", "action"])
        self.assertEqual(events[-1][1], "SUCCESS")

    def test_broken_observer_does_not_break_pipeline(self):
        def explode(token):
            raise RuntimeError("observer bug")

        pipeline = self._pipeline(on_token=explode)
        pipeline.hardware.start()
        pipeline.step(window_ms=10)
        action = pipeline.process_phrase()
        self.assertIsNotNone(action)

    def test_policy_block_counts_and_reports(self):
        statuses = []
        pipeline = self._pipeline(
            policy_engine=_DenyAllPolicy(),
            on_action=lambda a, status: statuses.append(status),
        )
        pipeline.hardware.start()
        pipeline.step(window_ms=10)
        self.assertIsNone(pipeline.process_phrase())
        self.assertEqual(pipeline.stats.actions_blocked, 1)
        self.assertEqual(statuses, ["REJECTED_UNAUTHORIZED"])

    def test_raise_on_policy_violation(self):
        errors = []
        pipeline = self._pipeline(
            policy_engine=_DenyAllPolicy(),
            raise_on_policy_violation=True,
            on_error=errors.append,
        )
        pipeline.hardware.start()
        pipeline.step(window_ms=10)
        with self.assertRaises(PolicyViolationError):
            pipeline.process_phrase()
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], PolicyViolationError)

    def test_provider_failure_routes_to_on_error(self):
        class FailingProvider(MockLLMProvider):
            def reconstruct_intent(self, tokens, context):
                raise ProviderError("api down")

        errors = []
        pipeline = self._pipeline(on_error=errors.append)
        pipeline.llm_provider = FailingProvider()
        pipeline.hardware.start()
        pipeline.step(window_ms=10)
        self.assertIsNone(pipeline.process_phrase())
        self.assertEqual(pipeline.stats.errors, 1)
        self.assertIsInstance(errors[0], ProviderError)


class TestFramePathways(unittest.TestCase):
    def test_frame_roundtrip_still_works(self):
        """Sanity: the upgraded pipeline still accepts plain frames."""
        now = time.time()
        frame = Frame(
            samples=[Sample(timestamp=now, channels=[0.1] * 4, sample_index=1)],
            start_time=now, end_time=now, fs=250.0,
        )
        self.assertEqual(frame.to_numpy().shape, (1, 4))


if __name__ == "__main__":
    unittest.main()
