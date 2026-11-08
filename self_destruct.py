"""Vast retirement with indefinite capped retries; queued work is an activity lease."""

import asyncio
import math
import os
import time
from urllib.parse import quote


async def delete_instance(instance_id, key):
    import aiohttp

    url = f"https://console.vast.ai/api/v0/instances/{quote(instance_id, safe='')}/"
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20)) as http:
            async with http.delete(url, headers={"Authorization": f"Bearer {key}"},
                                   allow_redirects=False) as response:
                return response.status
    except (aiohttp.ClientError, OSError, asyncio.TimeoutError):
        raise OSError("network") from None


async def retire(log, *, reason, env=None, delete=None, sleep=asyncio.sleep):
    env = os.environ if env is None else env
    delete = delete_instance if delete is None else delete
    try:
        disabled = float(env.get("COMFY_SELF_DESTRUCT_IDLE_S", "540")) == 0
    except ValueError:
        disabled = False
    skipped = ("disabled" if disabled
               else "no_container_key" if not env.get("CONTAINER_API_KEY")
               else "no_container_id" if not env.get("CONTAINER_ID") else None)
    if skipped:
        log(f"COMFY_SELF_DESTRUCT skipped={skipped}")
        return
    delay = 1
    while True:
        try:
            status = await delete(env["CONTAINER_ID"], env["CONTAINER_API_KEY"])
        except (OSError, asyncio.TimeoutError):
            status = "network_error"
        log(f"COMFY_SELF_DESTRUCT {reason} status={status}")
        if isinstance(status, int) and (200 <= status < 300 or status == 404):
            return
        await sleep(delay)
        delay = min(60, delay * 2)


class IdleRetirement:
    def __init__(self, log, *, env=None, clock=time.monotonic,
                 delete=delete_instance, sleep=asyncio.sleep, request_retirement=None):
        env = os.environ if env is None else env
        self.threshold = float(env.get("COMFY_SELF_DESTRUCT_IDLE_S", "540"))
        if not math.isfinite(self.threshold) or self.threshold < 0:
            raise ValueError("invalid idle threshold")
        self.instance_id = env.get("CONTAINER_ID")
        self.key = env.get("CONTAINER_API_KEY")
        self.env = dict(env)
        self.request_retirement = request_retirement
        self.clock, self.delete, self.sleep, self.log = clock, delete, sleep, log
        self.last_activity = clock()
        self.in_flight = 0
        self.jobs = 0
        self.done = False
        self.retiring = False

    def request_started(self):
        self.in_flight += 1
        self.last_activity = self.clock()

    def request_finished(self):
        self.in_flight -= 1
        self.last_activity = self.clock()

    def job_started(self):
        self.jobs += 1
        self.last_activity = self.clock()

    def job_finished(self):
        self.jobs -= 1
        self.last_activity = self.clock()

    async def check(self):
        if self.done:
            return
        reason = ("disabled" if self.threshold == 0 else
                  "no_container_key" if not self.key and not self.request_retirement else
                  "no_container_id" if not self.instance_id and not self.request_retirement else None)
        if reason:
            self.done = True
            self.log(f"COMFY_SELF_DESTRUCT skipped={reason}")
            return
        idle = self.clock() - self.last_activity
        if self.in_flight or self.jobs or idle < self.threshold:
            return
        self.retiring = True
        if self.request_retirement:
            await self.request_retirement()
        else:
            await retire(self.log, reason=f"idle={int(idle)}", env=self.env,
                         delete=self.delete, sleep=self.sleep)
        self.done = True

    async def run(self):
        while not self.done:
            await self.check()
            if not self.done:
                await self.sleep(15)
