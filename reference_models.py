"""Optional models: one shared background transfer, isolated from readiness and rendering."""

import asyncio
from pathlib import Path
import threading

from download import download_reference_models, download_lab_model
from functools import partial


class ReferenceModels:
    def __init__(self, root, ready_file, downloader=download_reference_models):
        self.root, self.ready_file, self.downloader = root, Path(ready_file), downloader
        self.task = None
        self.progress = 0.0
        self.stopped = threading.Event()

    def start(self):
        if self.task is None or (self.task.done() and not self.task.result()):
            self.progress = 0.0
            self.task = asyncio.create_task(self._download())
        return self.task

    async def _download(self):
        while not self.ready_file.exists():
            await asyncio.sleep(.1)
        loop = asyncio.get_running_loop()

        def progress(value):
            loop.call_soon_threadsafe(setattr, self, "progress", value)

        try:
            await asyncio.to_thread(self.downloader, self.root, progress, self.stopped)
            self.progress = 1.0
            return True
        except Exception:
            return False

    async def wait(self, progress):
        task = self.start()
        while not task.done():
            progress(self.progress)
            await asyncio.wait({task}, timeout=.1)
        if not task.result():
            raise RuntimeError("reference model download failed; retry the references job")

    async def close(self):
        self.stopped.set()
        if self.task and not self.task.done():
            self.task.cancel()
        if self.task:
            try:
                await self.task
            except asyncio.CancelledError:
                pass


class LabModels:
    """One lazy, shared transfer per optional profile; unrelated jobs keep running."""

    def __init__(self, root, ready_file, downloader=download_lab_model):
        self.root, self.ready_file, self.downloader = root, ready_file, downloader
        self.models = {}

    async def wait(self, profile, progress):
        if profile not in self.models:
            self.models[profile] = ReferenceModels(self.root, self.ready_file, partial(self.downloader, profile))
        await self.models[profile].wait(progress)

    async def close(self):
        await asyncio.gather(*(model.close() for model in self.models.values()))
