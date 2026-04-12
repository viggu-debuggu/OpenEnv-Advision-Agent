import asyncio
from advision_env.client import AdVisionEnv

async def main():
    async with AdVisionEnv(base_url='http://127.0.0.1:7860') as env:
        res = await env.reset()
        print("RESET SUCCESS:", res)

asyncio.run(main())
