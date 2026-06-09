import json
import logging
import socket
import threading
from typing import Any

logger = logging.getLogger("subvocal.stream.datachannel")


class BiometricDataChannelServer:
    """TCP server that broadcasts real-time biometric metrics to connected dashboard/client nodes."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8100):
        self.host = host
        self.port = port

        self._server_socket: socket.socket | None = None
        self._clients: list[socket.socket] = []
        self._lock = threading.Lock()
        self._is_running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Starts the TCP server in a background thread."""
        with self._lock:
            if self._is_running:
                return
            self._is_running = True

        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.bind((self.host, self.port))
        self._server_socket.listen(5)

        self._thread = threading.Thread(
            target=self._accept_loop, name=f"BiometricDataChannelServer-{self.port}", daemon=True
        )
        self._thread.start()
        logger.info("Biometric data channel server listening on %s:%d", self.host, self.port)

    def _accept_loop(self) -> None:
        while True:
            with self._lock:
                if not self._is_running:
                    break
            try:
                assert self._server_socket is not None
                client_sock, addr = self._server_socket.accept()
                with self._lock:
                    self._clients.append(client_sock)
                logger.debug("Client connected from %s to biometric data channel", addr)
            except Exception:
                break

    def broadcast(self, payload: dict[str, Any]) -> None:
        """Sends a JSON-serialized dictionary to all active listeners."""
        with self._lock:
            if not self._is_running:
                return

        payload_bytes = (json.dumps(payload) + "\n").encode("utf-8")
        failed_clients: list[socket.socket] = []

        with self._lock:
            clients = list(self._clients)

        for client in clients:
            try:
                client.sendall(payload_bytes)
            except Exception:
                failed_clients.append(client)

        if failed_clients:
            with self._lock:
                for fc in failed_clients:
                    if fc in self._clients:
                        self._clients.remove(fc)
                        try:
                            fc.close()
                        except Exception:
                            pass

    def close(self) -> None:
        """Closes the server and drops all client connections."""
        with self._lock:
            if not self._is_running:
                return
            self._is_running = False

        if self._server_socket:
            try:
                self._server_socket.close()
            except Exception:
                pass

        with self._lock:
            for client in self._clients:
                try:
                    client.close()
                except Exception:
                    pass
            self._clients.clear()

        if self._thread:
            self._thread.join(timeout=1.0)


class BiometricDataChannelClient:
    """Helper client to connect to the BiometricDataChannelServer and parse message frames."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8100):
        self.host = host
        self.port = port
        self._socket: socket.socket | None = None

    def connect(self) -> None:
        """Connects to the server socket."""
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.connect((self.host, self.port))

    def read_messages(self):
        """Generator yielding parsed messages from the streaming buffer."""
        if not self._socket:
            raise RuntimeError("Client not connected.")

        buffer = ""
        while True:
            try:
                data = self._socket.recv(4096).decode("utf-8")
                if not data:
                    break
                buffer += data
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    if line.strip():
                        yield json.loads(line)
            except Exception:
                break

    def close(self) -> None:
        """Disconnects the client socket."""
        if self._socket:
            try:
                self._socket.close()
            except Exception:
                pass
            self._socket = None
