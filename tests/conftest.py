import base64
import io
import socket
import uuid

import aiohttp
from PIL import Image
import pytest


@pytest.fixture(autouse=True)
def no_external_network(monkeypatch):
    connect, connect_ex = socket.socket.connect, socket.socket.connect_ex

    def guard(original):
        def checked(sock, address):
            if isinstance(address, tuple) and address[0] not in ("127.0.0.1", "::1"):
                raise AssertionError("external networking is forbidden in tests")
            return original(sock, address)
        return checked

    monkeypatch.setattr(socket.socket, "connect", guard(connect))
    monkeypatch.setattr(socket.socket, "connect_ex", guard(connect_ex))


def encoded_image(fmt="PNG", size=(32, 24)):
    stream = io.BytesIO()
    Image.new("RGB", size, (25, 60, 90)).save(stream, format=fmt)
    return base64.b64encode(stream.getvalue()).decode("ascii")


def payload(**overrides):
    return {"request": uuid.uuid4().hex, "mode": "text", "prompt": "A fox runs through snow", "seconds": 5,
            "size": "480p", "aspect": "16:9", "seed": 123, "audio": True, **overrides}


def connector():
    return aiohttp.TCPConnector(resolver=aiohttp.ThreadedResolver())
