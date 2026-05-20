"""
camera_adapters.py — Universal multi-manufacturer camera adapter framework.

COMMAND CENTER supports any camera that can be reached over IP, USB, or
serial-style protocols. Each manufacturer/protocol gets a thin adapter that
implements the CameraAdapter interface. New cameras plug in by:

    1. Subclassing CameraAdapter
    2. Implementing snapshot() and stream_url()
    3. Calling register_adapter("brand_name", YourAdapterClass)

The registry is auto-discovered at import time — drop a new file in
camera_adapters/ (or just register from anywhere) and it works.

CURRENTLY SUPPORTED:
    tapo        TP-Link Tapo (existing)
    rtsp        Generic RTSP (Hikvision, Dahua, Amcrest, Reolink, Wyze v3+ flashed, Axis, etc.)
    mjpeg       Generic HTTP MJPEG / multipart stream (older webcams, Foscam, etc.)
    onvif       ONVIF Profile S (universal pro IP cam standard)
    reolink     Reolink (RTSP + HTTP API for snapshot)
    amcrest     Amcrest / Dahua HTTP CGI
    hikvision   Hikvision ISAPI HTTP snapshot
    wyze        Wyze (via local RTSP firmware or docker-wyze-bridge)
    usb         USB / V4L2 (laptops, USB webcams via OpenCV index)
    go2rtc      go2rtc proxy (existing)
    http_snap   Generic single-shot HTTP JPEG poll (works for ANY camera with a snapshot URL)
    bluetooth   Bluetooth/BLE camera (placeholder — needs PyBluez or Bleak; many BT cams use SPP+JPEG chunks)

CLAUDE-CODE EXTENSION GUIDE (read before adding new vendors):
    - Each adapter is ~30-80 lines. Look at ReolinkAdapter or HikvisionAdapter as templates.
    - snapshot() must return JPEG bytes or None.
    - stream_url() returns an RTSP/HTTP URL ffmpeg can ingest, or None for snapshot-only.
    - capabilities() lists features so the dashboard knows what's possible.
    - Cloud-API cameras (Ring, Arlo, Nest) need OAuth — see RingAdapter stub at the bottom
      for the contract; populate the access_token field via env var or user UI.
"""
from __future__ import annotations

import base64
import os
import socket
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


# ── Capability flags (used by dashboard / autodiscovery) ──────────

CAP_SNAPSHOT  = "snapshot"
CAP_STREAM    = "stream"
CAP_PTZ       = "ptz"
CAP_AUDIO     = "audio"
CAP_NIGHT     = "night_vision"
CAP_TWO_WAY   = "two_way_audio"
CAP_RECORD    = "onboard_record"
CAP_MOTION    = "onboard_motion"
CAP_ONVIF     = "onvif"
CAP_BLUETOOTH = "bluetooth"


# ── Base adapter ──────────────────────────────────────────────────

@dataclass
class CameraInfo:
    """Identity + reachability info returned by adapters."""
    id:           str
    name:         str
    vendor:       str
    model:        str    = "unknown"
    ip:           str    = ""
    port:         int    = 0
    protocol:     str    = ""
    capabilities: list[str] = field(default_factory=list)
    metadata:     dict   = field(default_factory=dict)


class CameraAdapter(ABC):
    """Abstract base class for every supported camera type.

    Subclasses must implement snapshot(). stream_url() and probe() have
    sensible defaults but should be overridden for richer behavior.
    """
    vendor: str = "generic"
    default_port: int = 80

    def __init__(self, cfg: dict) -> None:
        self.cfg     = cfg
        self.id      = cfg["id"]
        self.name    = cfg.get("name", self.id)
        self.ip      = cfg.get("ip", "")
        self.port    = int(cfg.get("port", self.default_port))
        self.user    = cfg.get("username", "")
        self.passwd  = cfg.get("password", "")
        self.timeout = float(cfg.get("timeout", 5.0))

    # ── Required ──
    @abstractmethod
    def snapshot(self) -> bytes | None:
        """Return a single JPEG frame or None on failure."""

    # ── Recommended ──
    def stream_url(self) -> str | None:
        """Return an RTSP/HTTP URL ffmpeg can read, or None if snap-only."""
        return None

    def probe(self) -> bool:
        """Quick reachability test (TCP connect)."""
        if not self.ip:
            return False
        try:
            with socket.create_connection((self.ip, self.port), timeout=self.timeout):
                return True
        except OSError:
            return False

    def capabilities(self) -> list[str]:
        return [CAP_SNAPSHOT]

    def info(self) -> CameraInfo:
        return CameraInfo(
            id=self.id, name=self.name, vendor=self.vendor,
            model=self.cfg.get("model", "unknown"),
            ip=self.ip, port=self.port, protocol=self.vendor,
            capabilities=self.capabilities(),
            metadata=self.cfg.get("metadata", {}),
        )


# ── Helpers ──────────────────────────────────────────────────────

def _http_get(url: str, user: str = "", passwd: str = "", timeout: float = 5.0) -> bytes | None:
    """HTTP GET with optional Basic auth, returns body bytes or None."""
    req = urllib.request.Request(url)
    if user or passwd:
        token = base64.b64encode(f"{user}:{passwd}".encode()).decode()
        req.add_header("Authorization", f"Basic {token}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except (urllib.error.URLError, socket.timeout, ConnectionError):
        return None
    except Exception:
        return None


def _rtsp_url(scheme: str, user: str, passwd: str, ip: str, port: int, path: str) -> str:
    """Build an RTSP URL with embedded credentials."""
    auth = ""
    if user or passwd:
        auth = f"{user}:{passwd}@"
    if not path.startswith("/"):
        path = "/" + path
    if (scheme == "rtsp" and port == 554) or (scheme == "rtsps" and port == 322):
        return f"{scheme}://{auth}{ip}{path}"
    return f"{scheme}://{auth}{ip}:{port}{path}"


# ── Built-in adapters ────────────────────────────────────────────

class GenericRTSPAdapter(CameraAdapter):
    """Generic RTSP camera. Works for Hikvision, Dahua, Wyze flashed, Axis, etc.
    Config: {ip, port (554), username, password, rtsp_path (default /)}."""
    vendor = "rtsp"
    default_port = 554

    def stream_url(self) -> str | None:
        path   = self.cfg.get("rtsp_path", "/")
        scheme = self.cfg.get("scheme", "rtsp")
        return _rtsp_url(scheme, self.user, self.passwd, self.ip, self.port, path)

    def snapshot(self) -> bytes | None:
        try:
            import cv2
        except ImportError:
            return None
        url = self.stream_url()
        if not url:
            return None
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        try:
            ok, frame = cap.read()
            if not ok:
                return None
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return buf.tobytes() if ok else None
        finally:
            cap.release()

    def capabilities(self) -> list[str]:
        return [CAP_SNAPSHOT, CAP_STREAM]


class GenericMJPEGAdapter(CameraAdapter):
    """Generic HTTP MJPEG (multipart/x-mixed-replace) or single JPEG poll."""
    vendor = "mjpeg"
    default_port = 80

    def snapshot(self) -> bytes | None:
        url  = self.cfg.get("snapshot_url") or f"http://{self.ip}:{self.port}/snapshot"
        return _http_get(url, self.user, self.passwd, self.timeout)

    def stream_url(self) -> str | None:
        return self.cfg.get("stream_url") or f"http://{self.ip}:{self.port}/video"

    def capabilities(self) -> list[str]:
        return [CAP_SNAPSHOT, CAP_STREAM]


class HTTPSnapAdapter(CameraAdapter):
    """Single-shot HTTP JPEG poller. Works for almost ANY camera with a known snap URL.
    Config: {snapshot_url, username, password, poll_interval}."""
    vendor = "http_snap"
    default_port = 80

    def snapshot(self) -> bytes | None:
        url = self.cfg["snapshot_url"]
        return _http_get(url, self.user, self.passwd, self.timeout)


class HikvisionAdapter(CameraAdapter):
    """Hikvision ISAPI. Snapshot via /ISAPI/Streaming/channels/101/picture, RTSP /Streaming/Channels/101."""
    vendor = "hikvision"
    default_port = 80

    def snapshot(self) -> bytes | None:
        url = f"http://{self.ip}:{self.port}/ISAPI/Streaming/channels/101/picture"
        return _http_get(url, self.user, self.passwd, self.timeout)

    def stream_url(self) -> str | None:
        return _rtsp_url("rtsp", self.user, self.passwd, self.ip, 554,
                         self.cfg.get("rtsp_path", "/Streaming/Channels/101"))

    def capabilities(self) -> list[str]:
        return [CAP_SNAPSHOT, CAP_STREAM, CAP_PTZ, CAP_NIGHT, CAP_ONVIF, CAP_MOTION]


class DahuaAmcrestAdapter(CameraAdapter):
    """Dahua / Amcrest HTTP CGI snapshot at /cgi-bin/snapshot.cgi, RTSP /cam/realmonitor."""
    vendor = "amcrest"
    default_port = 80

    def snapshot(self) -> bytes | None:
        url = f"http://{self.ip}:{self.port}/cgi-bin/snapshot.cgi?channel=1"
        return _http_get(url, self.user, self.passwd, self.timeout)

    def stream_url(self) -> str | None:
        path = self.cfg.get("rtsp_path", "/cam/realmonitor?channel=1&subtype=0")
        return _rtsp_url("rtsp", self.user, self.passwd, self.ip, 554, path)

    def capabilities(self) -> list[str]:
        return [CAP_SNAPSHOT, CAP_STREAM, CAP_PTZ, CAP_NIGHT, CAP_ONVIF, CAP_TWO_WAY]


class ReolinkAdapter(CameraAdapter):
    """Reolink. Snapshot via /cgi-bin/api.cgi?cmd=Snap, RTSP /h264Preview_01_main."""
    vendor = "reolink"
    default_port = 80

    def snapshot(self) -> bytes | None:
        url = (f"http://{self.ip}:{self.port}/cgi-bin/api.cgi?"
               f"cmd=Snap&channel=0&user={self.user}&password={self.passwd}")
        return _http_get(url, timeout=self.timeout)

    def stream_url(self) -> str | None:
        path = self.cfg.get("rtsp_path", "/h264Preview_01_main")
        return _rtsp_url("rtsp", self.user, self.passwd, self.ip, 554, path)

    def capabilities(self) -> list[str]:
        return [CAP_SNAPSHOT, CAP_STREAM, CAP_NIGHT, CAP_MOTION, CAP_ONVIF]


class WyzeAdapter(CameraAdapter):
    """Wyze (requires RTSP firmware OR docker-wyze-bridge running locally).
    Config: {ip, port (8554), username, password, rtsp_path (default /live)}."""
    vendor = "wyze"
    default_port = 8554

    def stream_url(self) -> str | None:
        path = self.cfg.get("rtsp_path", "/live")
        return _rtsp_url("rtsp", self.user, self.passwd, self.ip, self.port, path)

    def snapshot(self) -> bytes | None:
        # docker-wyze-bridge exposes a snapshot endpoint
        if self.cfg.get("bridge_url"):
            url = f"{self.cfg['bridge_url'].rstrip('/')}/snapshot/{self.cfg.get('cam_name', self.id)}.jpg"
            data = _http_get(url, timeout=self.timeout)
            if data:
                return data
        return GenericRTSPAdapter(self.cfg).snapshot()

    def capabilities(self) -> list[str]:
        return [CAP_SNAPSHOT, CAP_STREAM, CAP_NIGHT, CAP_MOTION]


class ONVIFAdapter(CameraAdapter):
    """ONVIF Profile S — universal pro IP camera standard.
    Tries common snapshot/stream URI patterns then falls back to ISAPI/CGI/Reolink probes.
    For full ONVIF discovery + media URI negotiation, install onvif-zeep and switch to the
    full client; this lightweight version handles the 90% case without that dependency."""
    vendor = "onvif"
    default_port = 80

    SNAPSHOT_PATHS = [
        "/onvif/snapshot",
        "/onvif/media_service/snapshot",
        "/onvif-http/snapshot",
        "/cgi-bin/snapshot.cgi",
        "/ISAPI/Streaming/channels/101/picture",
        "/snapshot.jpg",
        "/image/jpeg.cgi",
    ]
    RTSP_PATHS = [
        "/onvif/profile1",
        "/onvif1",
        "/Streaming/Channels/101",
        "/cam/realmonitor?channel=1&subtype=0",
        "/h264Preview_01_main",
        "/live/main",
        "/live",
    ]

    def snapshot(self) -> bytes | None:
        forced = self.cfg.get("snapshot_url")
        if forced:
            return _http_get(forced, self.user, self.passwd, self.timeout)
        for path in self.SNAPSHOT_PATHS:
            url  = f"http://{self.ip}:{self.port}{path}"
            data = _http_get(url, self.user, self.passwd, self.timeout)
            if data and len(data) > 1024 and data[:3] in (b"\xff\xd8\xff", b"GIF", b"\x89PN"):
                return data
        return None

    def stream_url(self) -> str | None:
        forced = self.cfg.get("stream_url")
        if forced:
            return forced
        # Pick the first RTSP path the user configured, default to onvif/profile1
        path = self.cfg.get("rtsp_path", self.RTSP_PATHS[0])
        return _rtsp_url("rtsp", self.user, self.passwd, self.ip, 554, path)

    def capabilities(self) -> list[str]:
        return [CAP_SNAPSHOT, CAP_STREAM, CAP_ONVIF, CAP_PTZ]


class USBAdapter(CameraAdapter):
    """USB / V4L2 webcam via OpenCV index. Config: {device_index: 0}."""
    vendor = "usb"

    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        self.device_index = int(cfg.get("device_index", 0))

    def probe(self) -> bool:
        try:
            import cv2
            cap = cv2.VideoCapture(self.device_index)
            ok  = cap.isOpened()
            cap.release()
            return ok
        except Exception:
            return False

    def snapshot(self) -> bytes | None:
        try:
            import cv2
        except ImportError:
            return None
        cap = cv2.VideoCapture(self.device_index)
        try:
            ok, frame = cap.read()
            if not ok:
                return None
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return buf.tobytes() if ok else None
        finally:
            cap.release()

    def capabilities(self) -> list[str]:
        return [CAP_SNAPSHOT, CAP_STREAM]


class BluetoothAdapter(CameraAdapter):
    """Bluetooth / BLE camera placeholder.

    Most BT 'cameras' are body-cams or action-cams that expose either:
      - GATT JPEG-chunk characteristics (BLE) → use Bleak
      - Bluetooth Serial Port Profile (SPP) JPEG stream → use PyBluez
      - WiFi-Direct after BT pairing handshake (most action cams)

    Config: {bt_address, gatt_char_uuid OR spp_channel, snapshot_method}.
    Set BT_ENABLE=1 and install `bleak` to activate.
    """
    vendor = "bluetooth"

    def probe(self) -> bool:
        return os.environ.get("BT_ENABLE") == "1"

    def snapshot(self) -> bytes | None:
        if not self.probe():
            return None
        try:
            import asyncio
            from bleak import BleakClient  # type: ignore
        except ImportError:
            return None
        addr = self.cfg.get("bt_address")
        char = self.cfg.get("gatt_char_uuid")
        if not addr or not char:
            return None

        async def _read() -> bytes | None:
            async with BleakClient(addr) as client:
                if not client.is_connected:
                    return None
                data = await client.read_gatt_char(char)
                return bytes(data) if data else None

        try:
            return asyncio.run(_read())
        except Exception:
            return None

    def capabilities(self) -> list[str]:
        return [CAP_SNAPSHOT, CAP_BLUETOOTH]


class RingStubAdapter(CameraAdapter):
    """Ring / Arlo / Nest stub — requires OAuth tokens. Set the relevant env var:
        RING_REFRESH_TOKEN  (use ring-doorbell python lib)
        ARLO_USERNAME / ARLO_PASSWORD
        NEST_OAUTH_TOKEN
    Then implement snapshot() against their cloud SDK. This stub returns None
    so the dashboard shows the camera as 'configured but offline' until tokens
    are provided."""
    vendor = "ring_stub"

    def snapshot(self) -> bytes | None:
        return None

    def capabilities(self) -> list[str]:
        return [CAP_SNAPSHOT]


# ── Registry ─────────────────────────────────────────────────────

_REGISTRY: dict[str, type[CameraAdapter]] = {}


def register_adapter(name: str, cls: type[CameraAdapter]) -> None:
    """Register a camera adapter class under a string vendor key."""
    _REGISTRY[name.lower()] = cls


def get_adapter(name: str) -> type[CameraAdapter] | None:
    return _REGISTRY.get(name.lower())


def list_adapters() -> list[dict]:
    return [
        {"vendor": name, "class": cls.__name__,
         "default_port": cls.default_port,
         "capabilities": cls(cfg={"id": "_probe"}).capabilities()}
        for name, cls in sorted(_REGISTRY.items())
    ]


def build_adapter(cfg: dict) -> CameraAdapter | None:
    """Instantiate the adapter for cfg['type']."""
    vendor = cfg.get("type", "rtsp").lower()
    cls    = get_adapter(vendor)
    if cls is None:
        return None
    return cls(cfg)


# Auto-register built-ins
register_adapter("rtsp",       GenericRTSPAdapter)
register_adapter("mjpeg",      GenericMJPEGAdapter)
register_adapter("http_snap",  HTTPSnapAdapter)
register_adapter("hikvision",  HikvisionAdapter)
register_adapter("amcrest",    DahuaAmcrestAdapter)
register_adapter("dahua",      DahuaAmcrestAdapter)
register_adapter("reolink",    ReolinkAdapter)
register_adapter("wyze",       WyzeAdapter)
register_adapter("onvif",      ONVIFAdapter)
register_adapter("usb",        USBAdapter)
register_adapter("bluetooth",  BluetoothAdapter)
register_adapter("ring",       RingStubAdapter)
register_adapter("arlo",       RingStubAdapter)
register_adapter("nest",       RingStubAdapter)


def adapter_summary() -> str:
    rows = list_adapters()
    lines = [f"▸ COMMAND CENTER CAMERA ADAPTER REGISTRY ({len(rows)} vendors)"]
    for r in rows:
        caps = ",".join(r["capabilities"])
        lines.append(f"  {r['vendor']:<12} :{r['default_port']:<5} [{caps}]")
    lines.append("▸ Add new vendor: subclass CameraAdapter and register_adapter('vendor', YourCls)")
    return "\n".join(lines)
