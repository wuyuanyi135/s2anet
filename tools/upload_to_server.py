import argparse
import asyncio
import os

import aiohttp


async def main(args):
    async with aiohttp.client.ClientSession() as session:
        data = aiohttp.FormData()
        for image in args.images:
            data.add_field(os.path.basename(image), open(image, "rb"))

        await session.post("http://localhost:9987/inference", data=data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+")
    args = parser.parse_args()

    asyncio.run(main(args))
