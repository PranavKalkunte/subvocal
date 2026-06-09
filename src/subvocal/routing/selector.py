import logging
from abc import ABC, abstractmethod
from typing import Protocol

logger = logging.getLogger("subvocal.routing.selector")


class WorkerNode(Protocol):
    """Protocol matching standard attributes of routing nodes."""
    @property
    def id(self) -> str: ...
    @property
    def load(self) -> float: ...
    @property
    def status(self) -> str: ...
    @property
    def cpu_usage(self) -> float: ...


class NodeSelector(ABC):
    """Base interface for all worker node routing selectors."""

    @abstractmethod
    def select_node(self, nodes: list[WorkerNode]) -> WorkerNode:
        """Selects the best node from candidate worker nodes.

        Args:
            nodes: Candidates list.

        Returns:
            The chosen WorkerNode.

        Raises:
            ValueError: If nodes list is empty.
        """
        pass


class SessionCountSelector(NodeSelector):
    """Selects the worker node with the minimum active session load."""

    def select_node(self, nodes: list[WorkerNode]) -> WorkerNode:
        if not nodes:
            raise ValueError("No worker nodes available for selection.")
        
        # Returns the node with the lowest active session load
        return min(nodes, key=lambda n: n.load)


class CPULoadSelector(NodeSelector):
    """Selects the worker node with the lowest CPU load report."""

    def select_node(self, nodes: list[WorkerNode]) -> WorkerNode:
        if not nodes:
            raise ValueError("No worker nodes available for selection.")

        # Returns the node with the lowest reported cpu usage
        return min(nodes, key=lambda n: n.cpu_usage)
