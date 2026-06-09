import time
import unittest

from subvocal.stream.datachannel import BiometricDataChannelClient, BiometricDataChannelServer


class TestBiometricDataChannel(unittest.TestCase):

    def test_datachannel_server_client_roundtrip(self):
        """Verify TCP socket server can broadcast to client correctly."""
        # Use a non-default test port
        port = 8199
        server = BiometricDataChannelServer(host="127.0.0.1", port=port)
        server.start()

        # Connect client
        client = BiometricDataChannelClient(host="127.0.0.1", port=port)
        try:
            client.connect()
            
            # Broadcast message
            payload = {"event": "test_broadcast", "data": "hello"}
            
            # Allow client accept callback to execute in server thread
            time.sleep(0.05)
            server.broadcast(payload)
            
            # Read messages from client
            messages = []
            for msg in client.read_messages():
                messages.append(msg)
                # Break after one message to avoid infinite read blocking
                break

            self.assertEqual(len(messages), 1)
            self.assertEqual(messages[0]["event"], "test_broadcast")
            self.assertEqual(messages[0]["data"], "hello")

        finally:
            client.close()
            server.close()


if __name__ == "__main__":
    unittest.main()
