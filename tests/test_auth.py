import time
import unittest
from unittest.mock import MagicMock

from pydantic import ValidationError

from subvocal.auth.grants import (
    ActionGrants,
    GrantsPolicy,
    clear_context_grants,
    ensure_permission,
    generate_token,
    get_context_grants,
    set_context_grants,
    verify_token,
)
from subvocal.context.schema import UserContext
from subvocal.core.models import Action
from subvocal.core.security import PolicyEngine


class TestAuthGrants(unittest.TestCase):

    def test_action_grants_validation(self):
        """Verify ActionGrants forbids extra keys and has expected defaults."""
        grants = ActionGrants(allowed_commands=["play", "pause"])
        self.assertEqual(grants.allowed_commands, ["play", "pause"])
        self.assertFalse(grants.admin)
        self.assertEqual(grants.min_confidence, 0.0)
        self.assertFalse(grants.enforced_dry_run)

        # Forbid extra fields
        with self.assertRaises(ValidationError):
            ActionGrants(allowed_commands=["play"], extra_field="forbidden")

    def test_token_generation_and_verification(self):
        """Verify token generation, signature validation, and expiration."""
        secret = "super-secret-key-123"
        grants = ActionGrants(allowed_commands=["volume_up", "volume_down"], min_confidence=0.8)

        # 1. Valid token roundtrip
        token = generate_token(grants, secret, expires_in=10)
        decoded = verify_token(token, secret)
        self.assertIsNotNone(decoded)
        self.assertEqual(decoded.allowed_commands, ["volume_up", "volume_down"])
        self.assertEqual(decoded.min_confidence, 0.8)

        # 2. Expired token validation fails
        expired_token = generate_token(grants, secret, expires_in=-10)
        self.assertIsNone(verify_token(expired_token, secret))

        # 3. Invalid signature token fails validation
        wrong_secret_decoded = verify_token(token, "wrong-secret-key")
        self.assertIsNone(wrong_secret_decoded)

        # 4. Tampered token fails validation
        tampered_token = token + "a"
        self.assertIsNone(verify_token(tampered_token, secret))

        # 5. Malformed token fails
        self.assertIsNone(verify_token("invalid_token_format", secret))

    def test_context_propagation(self):
        """Verify context variable propagation and recovery."""
        grants = ActionGrants(admin=True)
        self.assertIsNone(get_context_grants())

        # Set grants in context
        token = set_context_grants(grants)
        self.assertEqual(get_context_grants(), grants)

        # Clear/reset context grants
        clear_context_grants(token)
        self.assertIsNone(get_context_grants())

    def test_ensure_permission_helpers(self):
        """Verify ensure_permission checks match capabilities."""
        grants = ActionGrants(allowed_commands=["play", "pause"], min_confidence=0.7)
        token = set_context_grants(grants)
        try:
            # Authorized command and confidence
            ensure_permission("play", confidence=0.85)
            ensure_permission("pause", confidence=0.7)

            # Unauthorized command
            with self.assertRaises(PermissionError):
                ensure_permission("mute", confidence=0.9)

            # Confidence too low
            with self.assertRaises(PermissionError):
                ensure_permission("play", confidence=0.6)

        finally:
            clear_context_grants(token)

        # Deny if no context grants exist
        with self.assertRaises(PermissionError):
            ensure_permission("play", confidence=0.9)

    def test_grants_policy_authorization(self):
        """Verify GrantsPolicy integrates correctly with PolicyEngine."""
        policy = GrantsPolicy()
        engine = PolicyEngine(policies=[policy])
        context = MagicMock(spec=UserContext)

        action_play = Action(action_type="play", params={"confidence": 0.85}, intent_id="1", timestamp=time.time())
        action_mute = Action(action_type="mute", params={"confidence": 0.9}, intent_id="2", timestamp=time.time())

        # No grants -> Rejected (Fail Secure)
        self.assertFalse(engine.is_authorized(action_play, context))

        # With grants -> Authorized correctly
        grants = ActionGrants(allowed_commands=["play"], min_confidence=0.7)
        token = set_context_grants(grants)
        try:
            self.assertTrue(engine.is_authorized(action_play, context))
            # Command not whitelisted
            self.assertFalse(engine.is_authorized(action_mute, context))
        finally:
            clear_context_grants(token)


if __name__ == "__main__":
    unittest.main()
