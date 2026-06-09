"""Abstract base interfaces for subvocal middleware components.

Enables pluggable integration of various:
- Hardware sources (real-time stream, replay file, synthetic generators).
- LLM providers (Gemini, Claude, GPT, local Llama).
- Action executors (browser control, device APIs, notifications).
- Context providers (OS, active application, contacts, location).
"""

from abc import ABC, abstractmethod
from typing import List, Any, Optional, Tuple, Dict, Union
from subvocal.context.schema import UserContext
from .models import Frame, CommandToken, Intent, Action


class HardwareSource(ABC):
    """Abstract interface for subvocal/sEMG sensor data acquisition."""

    @abstractmethod
    def start(self) -> None:
        """Initialize connection and start streaming raw sEMG data."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop data acquisition and disconnect hardware clean."""
        pass

    @abstractmethod
    def read_frame(self, window_ms: int) -> Frame:
        """Reads a buffered window of raw sEMG samples.

        Args:
            window_ms: Time duration in milliseconds to buffer and retrieve.

        Returns:
            A Frame containing the Sample points.
        """
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Returns True if the sensor connection is healthy."""
        pass


class LLMProvider(ABC):
    """Abstract interface for turning noisy classified shorthand into semantic Intents."""

    @abstractmethod
    def reconstruct_intent(self, tokens: List[CommandToken], context: UserContext) -> Intent:
        """Resolves a noisy shorthand token stream to a structured Intent.

        Args:
            tokens: A sequence of classified CommandTokens.
            context: The UserContext state snapshot at execution time.

        Returns:
            The reconstructed Intent.
        """
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """Returns the name of the LLM provider (e.g., 'gemini', 'anthropic')."""
        pass


class ActionExecutor(ABC):
    """Abstract interface for dispatching resolved agentic Actions."""

    @abstractmethod
    def execute(self, action: Action) -> Any:
        """Dispatches the action to device APIs or agent tools.

        Args:
            action: The Action object to execute.

        Returns:
            The result of the execution.
        """
        pass

    @abstractmethod
    def can_execute(self, action: Action) -> bool:
        """Returns True if the executor supports this specific action type."""
        pass


class ContextProvider(ABC):
    """Abstract interface for retrieving live device or environment context."""

    @abstractmethod
    def get_context(self) -> UserContext:
        """Retrieves a snapshot of the active user/system context.

        Returns:
            The UserContext populated with active state metadata.
        """
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """Returns the name/identifier of this context provider."""
        pass


class Classifier(ABC):
    """Abstract interface for classifying physiological raw signals into command tokens."""

    @abstractmethod
    def predict(self, frame: Union[Frame, Any]) -> Optional[CommandToken]:
        """Classifies a Frame of raw signals into a CommandToken (applies gating/cooldown if configured)."""
        pass

    @abstractmethod
    def predict_raw(self, frame: Union[Frame, Any]) -> Tuple[str, float, List[float]]:
        """Predicts the probability distribution for a Frame of raw signals.

        Returns:
            (predicted_class_label, max_probability, all_probabilities_list)
        """
        pass

    @property
    @abstractmethod
    def labels(self) -> List[str]:
        """Returns the list of output labels/classes supported by the classifier."""
        pass

