import unittest
from unittest.mock import Mock

from client import Client


class ClientTest(unittest.TestCase):
    def test_client_exposes_injected_encrypt_and_decrypt_operations(self) -> None:
        encrypt = Mock(return_value={"upload": b"ciphertext"})
        decrypt = Mock(return_value={"layer.lora_A.weight": "tensor"})
        client = Client(client_id=2, encrypt_fn=encrypt, decrypt_fn=decrypt)

        upload = client.encrypt({"state": "value"}, 2, 4)
        state = client.decrypt({"aggregate": b"ciphertext"}, 4)

        self.assertEqual({"upload": b"ciphertext"}, upload)
        self.assertEqual({"layer.lora_A.weight": "tensor"}, state)
        self.assertEqual("client_2", client.adapter_name)


if __name__ == "__main__":
    unittest.main()
