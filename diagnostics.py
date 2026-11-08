"""Bounded child-output tee with credential redaction before either destination."""

import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import sys
import threading


def secret_name(name):
    name = name.upper()
    return any(part in name for part in ("TOKEN", "SECRET", "PASSWORD", "_KEY", "AUTH_DATA"))


def model_environment(env):
    return {name: value for name, value in env.items() if not secret_name(name)}


class Redactor:
    def __init__(self, secrets):
        self.secrets = sorted({s for s in secrets if s}, key=len, reverse=True)
        self.keep = max(map(len, self.secrets), default=1)
        self.pending = ""

    def feed(self, text, *, final=False):
        self.pending += text
        output = []
        offset = 0
        limit = len(self.pending) if final else max(0, len(self.pending) - self.keep + 1)
        while offset < limit:
            match = next((s for s in self.secrets if self.pending.startswith(s, offset)), None)
            if match:
                output.append("[REDACTED]")
                offset += len(match)
            else:
                output.append(self.pending[offset])
                offset += 1
        self.pending = self.pending[offset:]
        return "".join(output)


class Diagnostics:
    def __init__(self, path=None, *, env=None, stdout=None, max_bytes=5 * 1024 * 1024):
        env = os.environ if env is None else env
        path = Path(path or env.get("COMFY_DIAGNOSTIC_LOG", "/var/log/portal/comfyui.log"))
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handler = RotatingFileHandler(path, maxBytes=max_bytes, backupCount=3, encoding="utf-8")
        self.handler.terminator = ""
        self.stdout = sys.stdout if stdout is None else stdout
        self.secrets = [value for key, value in env.items() if secret_name(key)]
        self.lock = threading.Lock()
        self.threads = []

    def write(self, text):
        if not text:
            return
        with self.lock:
            self.handler.emit(logging.LogRecord("child", logging.INFO, "", 0, text, (), None))
            self.stdout.write(text)
            self.stdout.flush()

    def pump(self, stream):
        redactor = Redactor(self.secrets)
        try:
            while text := stream.readline(8192):
                self.write(redactor.feed(text))
            self.write(redactor.feed("", final=True))
        finally:
            stream.close()

    def attach(self, child):
        thread = threading.Thread(target=self.pump, args=(child.stdout,), daemon=True)
        self.threads.append(thread)
        thread.start()

    def close(self):
        for thread in self.threads:
            thread.join(timeout=5)
        self.handler.close()
