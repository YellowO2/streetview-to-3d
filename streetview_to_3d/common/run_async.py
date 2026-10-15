"""Run a coroutine to its end from code that is not async."""
import asyncio


def run_async(coro):
    """coro's result, run on a fresh event loop."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()
