#!/usr/bin/env python3.12
"""
Press the doorbell (or trigger motion) then run this.
Tells you exactly what's happening at each step.
"""
import asyncio, hashlib, os, re, sys, time
from pathlib import Path

IP         = os.environ.get("TAPO_IP", "192.168.x.x")
PORT       = 8800
CLOUD_PASS = os.environ.get("TAPO_PASSWORD", "")
USER       = "admin"
D_BOUNDARY = b"--device-stream-boundary--"
C_BOUNDARY = b"client-stream-boundary--"

import json
PREVIEW_REQ = json.dumps({
    "type": "request", "seq": 1,
    "params": {"preview": {"audio": ["default"], "channels": [0], "resolutions": ["HD"]}}
}, separators=(",",":")).encode()

def log(msg): print(f"  {msg}", flush=True)

async def main():
    print("\n── STEP 1: connecting to camera ──────────────────────────────")
    try:
        r, w = await asyncio.wait_for(asyncio.open_connection(IP, PORT), timeout=6)
        log("✓ TCP connected")
    except Exception as e:
        log(f"✗ Can't connect: {e}")
        log("  Camera is asleep or unreachable. Trigger motion/ring first.")
        return

    print("\n── STEP 2: first POST (expecting 401) ────────────────────────")
    w.write(
        b"POST /stream HTTP/1.1\r\n"
        b"Content-Type: multipart/mixed;boundary=client-stream-boundary--\r\n"
        b"Connection: keep-alive\r\n"
        b"Content-Length: -1\r\n\r\n"
    )
    await w.drain()
    try:
        raw = await asyncio.wait_for(r.readuntil(b"\r\n\r\n"), timeout=6)
        log(f"Response: {raw[:80].decode(errors='replace').strip()}")
    except Exception as e:
        log(f"✗ No response: {e}")
        return

    if b"401" not in raw:
        log(f"✗ Expected 401, got something else. Full response:\n{raw.decode(errors='replace')}")
        return
    log("✓ Got 401 Digest challenge")

    realm  = re.search(rb'realm="([^"]+)"',  raw).group(1).decode()
    nonce  = re.search(rb'nonce="([^"]+)"',  raw).group(1).decode()
    opaque = re.search(rb'opaque="([^"]+)"', raw).group(1).decode()
    log(f"  realm={realm}  nonce={nonce[:12]}...")

    print("\n── STEP 3: authenticated POST (expecting 200) ────────────────")
    hashed_pw = hashlib.sha256(CLOUD_PASS.encode()).hexdigest().upper()
    cnonce = "aabb1122ccdd3344"; nc = "00000001"
    ha1  = hashlib.md5(f"{USER}:{realm}:{hashed_pw}".encode()).hexdigest()
    ha2  = hashlib.md5(b"POST:/stream").hexdigest()
    resp = hashlib.md5(f"{ha1}:{nonce}:{nc}:{cnonce}:auth:{ha2}".encode()).hexdigest()
    auth = (
        f'Digest username="{USER}",realm="{realm}",uri="/stream",algorithm=MD5,'
        f'nonce="{nonce}",nc={nc},cnonce="{cnonce}",qop=auth,'
        f'response="{resp}",opaque="{opaque}"'
    ).encode()

    w.write(
        b"POST /stream HTTP/1.1\r\n"
        b"Content-Type: multipart/mixed;boundary=client-stream-boundary--\r\n"
        b"Connection: keep-alive\r\n"
        b"Content-Length: -1\r\n"
        b"Authorization: " + auth + b"\r\n\r\n"
    )
    await w.drain()
    try:
        raw2 = await asyncio.wait_for(r.readuntil(b"\r\n\r\n"), timeout=6)
        log(f"Response: {raw2[:120].decode(errors='replace').strip()}")
    except Exception as e:
        log(f"✗ No response: {e}")
        return

    if b"200" not in raw2:
        log(f"✗ Auth failed. Full response:\n{raw2.decode(errors='replace')}")
        return
    log("✓ Authenticated — camera streaming")

    print("\n── STEP 3.5: raw read immediately after 200 OK ───────────────")
    log("Reading raw bytes for 2s before sending anything...")
    raw_buf = bytearray()
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        try:
            chunk = await asyncio.wait_for(r.read(4096), timeout=0.5)
            if chunk:
                raw_buf.extend(chunk)
        except asyncio.TimeoutError:
            pass
        except Exception as e:
            log(f"  read error: {e}")
            break
    if raw_buf:
        log(f"  GOT {len(raw_buf)} bytes immediately after 200 OK!")
        log(f"  first 200 bytes: {bytes(raw_buf[:200])}")
    else:
        log("  nothing came before preview request")

    print("\n── STEP 4+5: trying preview format variations ─────────────────")
    # Try different JSON formats — D210 fw 1.1.0 rejects the standard one
    formats = [
        ("no-audio",    {"type":"request","seq":1,"params":{"preview":{"channels":[0],"resolutions":["HD"]}}}),
        ("pcm-audio",   {"type":"request","seq":1,"params":{"preview":{"channels":[0],"resolutions":["HD"],"audio":["pcm"]}}}),
        ("bare",        {"type":"request","seq":1,"params":{"preview":{"channels":[0]}}}),
        ("avStream",    {"type":"request","seq":1,"params":{"avStream":{"channel":0,"resolution":"HD"}}}),
        ("original",    {"type":"request","seq":1,"params":{"preview":{"audio":["default"],"channels":[0],"resolutions":["HD"]}}}),
    ]

    video_bytes = bytearray()
    chunks_seen = 0
    chosen_fmt  = None

    for fmt_name, fmt_json in formats:
        req = json.dumps(fmt_json, separators=(",",":")).encode()
        chunk = (
            b"--" + C_BOUNDARY + b"\r\n"
            b"Content-Type: application/json\r\n"
            b"Content-Length: " + str(len(req)).encode() + b"\r\n"
            b"\r\n" + req + b"\r\n"
        )
        w.write(chunk)
        await w.drain()
        log(f"  trying [{fmt_name}]: {req.decode()}")
        # Read response
        try:
            resp = await asyncio.wait_for(r.read(4096), timeout=2)
            log(f"  → {resp[:100]}")
            if b"400" not in resp and b"error" not in resp.lower():
                log(f"  ✓ format [{fmt_name}] accepted!")
                chosen_fmt = fmt_name
                video_bytes.extend(resp)
                break
            else:
                log(f"  ✗ rejected")
        except asyncio.TimeoutError:
            log(f"  → (timeout — possibly accepted, reading...)")
            chosen_fmt = fmt_name
            break

    log(f"\nchosen format: {chosen_fmt}")

    async def send_preview():
        await asyncio.sleep(15)  # hold connection open

    async def read_chunks():
        nonlocal chunks_seen
        deadline = time.monotonic() + 12
        # Try both boundary variants (Tapo boundary declaration is non-standard)
        bound_variants = [b"----device-stream-boundary--", b"--device-stream-boundary--"]
        while time.monotonic() < deadline:
            try:
                # Just read raw bytes and look manually
                chunk = await asyncio.wait_for(r.read(65536), timeout=3)
                if not chunk:
                    break
                log(f"  raw bytes: {len(chunk)}  hex prefix: {chunk[:32].hex()}")
                log(f"  text preview: {chunk[:80]}")
                for bv in bound_variants:
                    if bv in chunk:
                        log(f"  ✓ found boundary: {bv}")
            except asyncio.TimeoutError:
                log("  (no data in 3s — camera stopped)")
                break
            hdrs    = await asyncio.wait_for(r.readuntil(b"\r\n\r\n"), timeout=3)
            ctype_m = re.search(rb"Content-Type:\s*(\S+)", hdrs)
            clen_m  = re.search(rb"Content-Length:\s*(\d+)", hdrs)
            ctype   = ctype_m.group(1).decode(errors='replace') if ctype_m else "unknown"
            clen    = int(clen_m.group(1)) if clen_m else 0
            data    = await asyncio.wait_for(r.readexactly(clen), timeout=5) if clen else b""
            chunks_seen += 1
            if b"json" in ctype.encode():
                log(f"  chunk {chunks_seen}: JSON  {clen}b → {data[:80].decode(errors='replace')}")
            else:
                video_bytes.extend(data)
                log(f"  chunk {chunks_seen}: VIDEO {ctype}  {clen}b  (total: {len(video_bytes)//1024}KB)")
                if len(video_bytes) <= clen and len(data) >= 8:
                    log(f"    first bytes: {data[:16].hex()}")

    try:
        await asyncio.wait_for(
            asyncio.gather(send_preview(), read_chunks()),
            timeout=14
        )
    except (asyncio.TimeoutError, Exception) as e:
        if "TimeoutError" not in type(e).__name__:
            log(f"  ended: {e}")

    print(f"\n── RESULT ────────────────────────────────────────────────────")
    log(f"chunks seen:  {chunks_seen}")
    log(f"video data:   {len(video_bytes)//1024} KB")
    if video_bytes:
        out = Path("/tmp/doorbell_raw.bin")
        out.write_bytes(video_bytes)
        log(f"raw video saved to {out}")
        log(f"first 16 bytes: {bytes(video_bytes[:16]).hex()}")
        # Detect format
        magic = bytes(video_bytes[:8])
        if magic[:4] == b'\x00\x00\x00\x01' or magic[:3] == b'\x00\x00\x01':
            log("format hint: looks like raw H.264 (start codes detected)")
        elif magic[4:8] in (b'ftyp', b'mdat', b'moov'):
            log("format hint: looks like MP4/fMP4")
        elif magic[:4] == b'GIF8':
            log("format hint: GIF??")
        elif magic[:2] == b'\xff\xd8':
            log("format hint: JPEG")
        else:
            log(f"format hint: unknown — need to check hex")
        log("\ntry running:")
        log(f"  ffprobe /tmp/doorbell_raw.bin")
    else:
        log("✗ No video data received")

asyncio.run(main())
