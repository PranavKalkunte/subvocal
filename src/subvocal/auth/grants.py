import base64
import contextvars
import hashlib
import hmac
import json
import time
from typing import Optional

from pydantic import BaseModel, Field

from subvocal.context.schema import UserContext
from subvocal.core.models import Action
from subvocal.core.security import AuthorizationPolicy

# Context Variable for capability-scoped grants propagation
_grants_var: contextvars.ContextVar[Optional["ActionGrants"]] = contextvars.ContextVar(
    "subvocal_grants", default=None
)


class ActionGrants(BaseModel):
    """Capability-scoped credentials detailing authorized actions and constraints."""

    model_config = {"extra": "forbid"}

    allowed_commands: list[str] = Field(default_factory=list, description="Allowed action command names")
    admin: bool = Field(default=False, description="Full admin bypass for command matching")
    min_confidence: float = Field(default=0.0, description="Minimum confidence floor required to execute")
    enforced_dry_run: bool = Field(default=False, description="Forces all actions to resolve as DRY_RUN")


def set_context_grants(grants: ActionGrants) -> contextvars.Token:
    """Sets the given ActionGrants in the current thread/coroutine execution context."""
    return _grants_var.set(grants)


def get_context_grants() -> ActionGrants | None:
    """Retrieves active ActionGrants from the current context."""
    return _grants_var.get()


def clear_context_grants(token: contextvars.Token) -> None:
    """Resets the context grants to their previous state."""
    _grants_var.reset(token)


def generate_token(grants: ActionGrants, secret_key: str, expires_in: int = 3600) -> str:
    """Generates a secure HMAC-signed base64 token carrying capability claims."""
    payload = {
        "grants": grants.model_dump(),
        "exp": int(time.time()) + expires_in
    }
    payload_json = json.dumps(payload).encode("utf-8")
    payload_b64 = base64.urlsafe_b64encode(payload_json).decode("utf-8").rstrip("=")
    
    signature = hmac.new(secret_key.encode("utf-8"), payload_json, hashlib.sha256).digest()
    sig_b64 = base64.urlsafe_b64encode(signature).decode("utf-8").rstrip("=")
    
    return f"{payload_b64}.{sig_b64}"


def verify_token(token: str, secret_key: str) -> ActionGrants | None:
    """Verifies the HMAC signature and expiration of a base64 token, returning decoded ActionGrants."""
    try:
        parts = token.split(".")
        if len(parts) != 2:
            return None
        payload_b64, sig_b64 = parts
        
        # Restore padding
        payload_b64 += "=" * (4 - len(payload_b64) % 4)
        payload_json = base64.urlsafe_b64decode(payload_b64)
        
        # Verify signature
        expected_sig = hmac.new(secret_key.encode("utf-8"), payload_json, hashlib.sha256).digest()
        expected_sig_b64 = base64.urlsafe_b64encode(expected_sig).decode("utf-8").rstrip("=")
        
        if not hmac.compare_digest(sig_b64, expected_sig_b64):
            return None
            
        payload = json.loads(payload_json.decode("utf-8"))
        if payload.get("exp", 0) < time.time():
            return None
            
        return ActionGrants.model_validate(payload.get("grants", {}))
    except Exception:
        return None


def ensure_permission(action_type: str, confidence: float = 1.0) -> None:
    """Checks the active context grants and raises a PermissionError if the action is unauthorized."""
    grants = get_context_grants()
    if grants is None:
        raise PermissionError("No authorization grants found in execution context.")
    
    if grants.admin:
        if confidence < grants.min_confidence:
            raise PermissionError(
                f"Action '{action_type}' confidence {confidence} is below minimum {grants.min_confidence}."
            )
        return

    allowed = {cmd.lower() for cmd in grants.allowed_commands}
    if action_type.lower() not in allowed:
        raise PermissionError(f"Action '{action_type}' is not permitted by active grants.")
    
    if confidence < grants.min_confidence:
        raise PermissionError(
            f"Action '{action_type}' confidence {confidence} is below minimum {grants.min_confidence}."
        )


class GrantsPolicy(AuthorizationPolicy):
    """Enforces capability-scoped authorization checks on proposed Actions."""

    def is_authorized(self, action: Action, context: UserContext) -> bool:
        try:
            confidence = float(action.params.get("confidence", 1.0))
            ensure_permission(action.action_type, confidence)
            return True
        except PermissionError:
            return False
