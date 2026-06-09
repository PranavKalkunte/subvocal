import unittest

from subvocal.routing.selector import CPULoadSelector, SessionCountSelector


class DummyWorkerNode:
    def __init__(self, node_id: str, load: float, cpu_usage: float):
        self._id = node_id
        self._load = load
        self._cpu_usage = cpu_usage

    @property
    def id(self) -> str:
        return self._id

    @property
    def load(self) -> float:
        return self._load

    @property
    def status(self) -> str:
        return "active"

    @property
    def cpu_usage(self) -> float:
        return self._cpu_usage


class TestRoutingSelectors(unittest.TestCase):

    def test_session_count_selector(self):
        """Verify SessionCountSelector picks the node with minimum load."""
        node1 = DummyWorkerNode("node-1", load=0.8, cpu_usage=50.0)
        node2 = DummyWorkerNode("node-2", load=0.2, cpu_usage=90.0)
        node3 = DummyWorkerNode("node-3", load=0.5, cpu_usage=30.0)

        selector = SessionCountSelector()
        selected = selector.select_node([node1, node2, node3])
        self.assertEqual(selected.id, "node-2")

        # Empty nodes raises error
        with self.assertRaises(ValueError):
            selector.select_node([])

    def test_cpu_load_selector(self):
        """Verify CPULoadSelector picks the node with lowest CPU load."""
        node1 = DummyWorkerNode("node-1", load=0.8, cpu_usage=50.0)
        node2 = DummyWorkerNode("node-2", load=0.2, cpu_usage=90.0)
        node3 = DummyWorkerNode("node-3", load=0.5, cpu_usage=25.0)

        selector = CPULoadSelector()
        selected = selector.select_node([node1, node2, node3])
        self.assertEqual(selected.id, "node-3")

        with self.assertRaises(ValueError):
            selector.select_node([])


if __name__ == "__main__":
    unittest.main()
