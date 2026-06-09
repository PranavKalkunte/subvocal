"""Security policies and authorization gating for subvocal actions."""

from abc import ABC, abstractmethod
from typing import List, Set
from subvocal.core.models import Action
from subvocal.context.schema import UserContext


class AuthorizationPolicy(ABC):
    """Abstract base class for all subvocal execution policies."""

    @abstractmethod
    def is_authorized(self, action: Action, context: UserContext) -> bool:
        """Evaluate if the given action is authorized to execute.

        Args:
            action: The proposed Action object.
            context: The UserContext state snapshot at proposal time.

        Returns:
            True if authorized, False otherwise.
        """
        pass


class ConfidenceThresholdPolicy(AuthorizationPolicy):
    """Restricts execution of actions if classifier/reconstruction confidence is too low."""

    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold

    def is_authorized(self, action: Action, context: UserContext) -> bool:
        # Action params contains the confidence returned by the LLM/classifier
        confidence = float(action.params.get("confidence", 1.0))
        return confidence >= self.threshold


class CommandWhitelistPolicy(AuthorizationPolicy):
    """Restricts commands to a strict whitelist of allowed actions."""

    def __init__(self, allowed_commands: List[str]):
        self.allowed_commands = {cmd.lower() for cmd in allowed_commands}

    def is_authorized(self, action: Action, context: UserContext) -> bool:
        return action.action_type.lower() in self.allowed_commands


class ContextBoundPolicy(AuthorizationPolicy):
    """Blocks execution of sensitive commands unless the active application is safe."""

    def __init__(self, sensitive_commands: List[str], safe_applications: List[str]):
        self.sensitive_commands = {cmd.lower() for cmd in sensitive_commands}
        self.safe_applications = {app.lower() for app in safe_applications}

    def is_authorized(self, action: Action, context: UserContext) -> bool:
        if action.action_type.lower() in self.sensitive_commands:
            active_app = ""
            if context:
                if hasattr(context, "app_state") and context.app_state:
                    active_app = context.app_state.current_app
                elif hasattr(context, "active_application"):
                    active_app = context.active_application
            active_app = str(active_app or "").lower()
            return active_app in self.safe_applications
        return True


class PolicyEngine:
    """Orchestrates authorization consensus across multiple active policies."""

    def __init__(self, policies: List[AuthorizationPolicy] = None):
        self.policies = policies or []

    def add_policy(self, policy: AuthorizationPolicy) -> None:
        """Add a new policy to the engine."""
        self.policies.append(policy)

    def is_authorized(self, action: Action, context: UserContext) -> bool:
        """Returns True only if all configured policies approve the action."""
        for policy in self.policies:
            if not policy.is_authorized(action, context):
                return False
        return True
