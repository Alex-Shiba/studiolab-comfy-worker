import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from self_destruct import IdleRetirement, delete_instance


def idle(clock, **kwargs):
    return IdleRetirement(kwargs.pop("log", lambda _: None), clock=lambda: clock[0],
                          env=kwargs.pop("env", {"CONTAINER_ID": "123", "CONTAINER_API_KEY": "test-secret"}), **kwargs)


async def test_exact_threshold_and_one_shot():
    now, delete = [0], AsyncMock(return_value=204)
    retirement = idle(now, delete=delete)
    now[0] = 539
    await retirement.check()
    delete.assert_not_awaited()
    now[0] = 540
    await retirement.check()
    await retirement.check()
    delete.assert_awaited_once_with("123", "test-secret")
    assert retirement.retiring


async def test_queue_running_request_leases_and_reset():
    now, delete = [0], AsyncMock(return_value=200)
    retirement = idle(now, delete=delete)
    retirement.job_started()
    retirement.job_started()
    now[0] = 1000
    await retirement.check()
    retirement.job_finished()
    now[0] = 2000
    await retirement.check()
    retirement.job_finished()
    retirement.request_started()
    now[0] = 3000
    await retirement.check()
    retirement.request_finished()
    now[0] = 3539
    await retirement.check()
    delete.assert_not_awaited()
    now[0] = 3540
    await retirement.check()
    delete.assert_awaited_once()


@pytest.mark.parametrize("env,reason", [({}, "no_container_key"), ({"CONTAINER_API_KEY": "secret"}, "no_container_id"),
                                        ({"COMFY_SELF_DESTRUCT_IDLE_S": "0"}, "disabled")])
async def test_disabled_or_missing_credentials(env, reason):
    delete, logs = AsyncMock(), []
    retirement = idle([0], env=env, delete=delete, log=logs.append)
    await retirement.run()
    await retirement.check()
    assert logs == ["COMFY_SELF_DESTRUCT skipped=" + reason]
    assert not retirement.retiring
    delete.assert_not_awaited()


async def test_retries_and_safe_logging():
    now, logs = [0], []
    delete = AsyncMock(side_effect=[OSError("test-secret"), asyncio.TimeoutError("test-secret"), 503, 200])
    sleep = AsyncMock()
    retirement = idle(now, delete=delete, sleep=sleep, log=logs.append)
    now[0] = 540
    await retirement.check()
    assert delete.await_count == 4
    assert [c.args for c in sleep.await_args_list] == [(1,), (2,), (4,)]
    assert "test-secret" not in " ".join(logs)


@pytest.mark.parametrize("status", [401, 403, 429, 500, 503])
async def test_all_failed_statuses_retry_beyond_four_attempts(status):
    now, delete = [0], AsyncMock(side_effect=[status] * 9 + [204])
    sleep = AsyncMock()
    retirement = idle(now, delete=delete, sleep=sleep)
    now[0] = 540
    await retirement.check()
    await retirement.check()
    assert delete.await_count == 10 and retirement.done and retirement.retiring
    assert [c.args[0] for c in sleep.await_args_list] == [1, 2, 4, 8, 16, 32, 60, 60, 60]


@pytest.mark.parametrize("value", ["nan", "inf", "-1", "bad"])
def test_invalid_threshold(value):
    with pytest.raises(ValueError):
        idle([0], env={"COMFY_SELF_DESTRUCT_IDLE_S": value})


async def test_configured_threshold_and_run_poll():
    now, delete = [0], AsyncMock(return_value=200)

    async def sleep(seconds):
        now[0] += seconds

    retirement = idle(now, env={"COMFY_SELF_DESTRUCT_IDLE_S": "45", "CONTAINER_ID": "123",
                                "CONTAINER_API_KEY": "test-secret"}, delete=delete, sleep=sleep)
    await retirement.run()
    assert now[0] == 45


async def test_delete_http_contract(monkeypatch):
    import aiohttp

    response = AsyncMock()
    response.__aenter__.return_value.status = 204
    session = AsyncMock()
    session.delete = Mock(return_value=response)
    context = AsyncMock()
    context.__aenter__.return_value = session
    monkeypatch.setattr(aiohttp, "ClientSession", Mock(return_value=context))
    assert await delete_instance("123", "test-secret") == 204
    session.delete.assert_called_once_with("https://console.vast.ai/api/v0/instances/123/",
                                          headers={"Authorization": "Bearer test-secret"}, allow_redirects=False)


async def test_server_delegates_retirement_without_credentials():
    now = [0]
    request = AsyncMock()
    retirement = idle(now, env={}, request_retirement=request)
    now[0] = 540
    await retirement.check()
    await retirement.check()
    request.assert_awaited_once()
    assert retirement.retiring and retirement.done


async def test_indefinite_retry_is_cancellable_without_marking_done():
    now, entered = [0], asyncio.Event()
    delete = AsyncMock(return_value=429)
    async def sleep(seconds):
        entered.set()
        await asyncio.Event().wait()
    retirement = idle(now, delete=delete, sleep=sleep)
    now[0] = 540
    task = asyncio.create_task(retirement.check())
    await entered.wait()
    assert retirement.retiring and not retirement.done
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_already_deleted_is_success():
    now, delete = [540], AsyncMock(return_value=404)
    retirement = idle([0], delete=delete)
    retirement.clock = lambda: now[0]
    await retirement.check()
    assert retirement.done
    delete.assert_awaited_once()
