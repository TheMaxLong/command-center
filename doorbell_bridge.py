#!/usr/bin/env python3.12
"""Tapo D210 → MPEG-TS bridge for go2rtc exec source.
Streams via pytapo's encrypted protocol, pipes raw TS to stdout."""
import asyncio, json, os, sys

from pytapo import HttpMediaSession
from pytapo.const import EncryptionMethod

IP         = os.environ.get("TAPO_IP", "192.168.x.x")
PORT       = 8800
CLOUD_PASS = os.environ.get("TAPO_PASSWORD", "")

PREVIEW_REQ = json.dumps({
    "type": "request",
    "seq": 1,
    "params": {
        "preview": {
            "audio": ["default"],
            "channels": [0],
            "resolutions": ["HD"],
        },
        "method": "get",
    },
})

async def main():
    session = HttpMediaSession(
        ip=IP,
        cloud_password=CLOUD_PASS,
        super_secret_key="",
        encryptionMethod=EncryptionMethod.SHA256,
        port=PORT,
        window_size=50,
    )

    await session.start()
    out = sys.stdout.buffer
    try:
        async for resp in session.transceive(PREVIEW_REQ, no_data_timeout=15.0):
            if resp.mimetype == "video/mp2t" and isinstance(resp.plaintext, bytes):
                out.write(resp.plaintext)
                out.flush()
    finally:
        await session.close()

asyncio.run(main())
