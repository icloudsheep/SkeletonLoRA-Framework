import unittest
from unittest.mock import Mock

from server import Server


class ServerTest(unittest.TestCase):
    def test_secure_aggregate_returns_unmodified_protocol_payload(self) -> None:
        secure_aggregate = Mock(return_value={"protocol": "ckks", "layers": {}})
        plaintext_aggregate = Mock()
        server = Server(
            aggregate_fn=plaintext_aggregate,
            secure_aggregate_fn=secure_aggregate,
        )
        payloads = [(0, {"ciphertext": b"a"}), (1, {"ciphertext": b"b"})]

        result = server.aggregate(payloads, round_id=3)

        self.assertEqual({"protocol": "ckks", "layers": {}}, result)
        secure_aggregate.assert_called_once_with(payloads, 3)
        plaintext_aggregate.assert_not_called()

    def test_plain_aggregate_receives_payload_values(self) -> None:
        server = Server(aggregate_fn=lambda values: sum(values))

        result = server.aggregate([(0, 2), (1, 3)], round_id=1)

        self.assertEqual(5, result)


if __name__ == "__main__":
    unittest.main()
