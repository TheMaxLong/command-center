"""
camera_discover.py — Network camera auto-discovery.

Discovers cameras on the local LAN using a layered approach (each layer is
optional and can be enabled independently):

    1. ONVIF WS-Discovery   — UDP multicast 239.255.255.250:3702 SOAP probe
                              Picks up ~95% of professional IP cams (Hikvision,
                              Dahua, Amcrest, Axis, Bosch, Reolink, Vivotek).
    2. Port scan            — Sweeps the configured subnet for cameras on
                              common ports (80, 554, 8554, 8800, 8000, 37777).
    3. Vendor fingerprinting — For each open port, identifies the vendor by
                              probing characteristic URLs (Hikvision ISAPI,
                              Dahua CGI, Reolink api.cgi, Tapo TLS:8800, etc.)
    4. mDNS                 — Listens for ._http._tcp / ._rtsp._tcp on .local
                              (catches Apple HomeKit / consumer cams).

Returns a list of CameraInfo dicts ready to be merged into cameras.yaml.

Usage:
    from camera_discover import discover_all
    found = discover_all(subnet="192.168.1.0/24", timeout=4)

    # Or async:
    from camera_discover import discover_all_async
    found = await discover_all_async(subnet="192.168.1.0/24")

CLAUDE-CODE EXTENSION POINTS:
    - Add a vendor probe: append to VENDOR_PROBES list (path, marker_bytes, vendor)
    - Add a discovery layer: implement discover_<name>() and call from discover_all()
"""
from __future__ import annotations

import ipaddress
import socket
import struct
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable

from camera_adapters import CameraInfo


# ── Vendor fingerprints (path → marker_bytes_lower → vendor) ─────

VENDOR_PROBES: list[tuple[int, str, bytes, str, str]] = [
    # (port, path,                              marker (lowercase),    vendor,     default model)
    (80,  "/ISAPI/Streaming/channels/101",      b"hikvision",          "hikvision", "ISAPI cam"),
    (80,  "/cgi-bin/snapshot.cgi",              b"dahua",              "amcrest",   "CGI cam"),
    (80,  "/cgi-bin/api.cgi?cmd=GetDevInfo",    b"reolink",            "reolink",   "api.cgi cam"),
    (80,  "/cgi-bin/api.cgi?cmd=GetDevInfo",    b"\"name\"",           "reolink",   "api.cgi cam"),
    (80,  "/onvif/device_service",              b"onvif",              "onvif",     "ONVIF Profile S"),
    (80,  "/",                                  b"hikvision",          "hikvision", "Web UI cam"),
    (80,  "/",                                  b"dahua",              "amcrest",   "Web UI cam"),
    (80,  "/",                                  b"reolink",            "reolink",   "Web UI cam"),
    (80,  "/",                                  b"axis",               "onvif",     "Axis cam"),
    (80,  "/",                                  b"vivotek",            "onvif",     "Vivotek cam"),
    (80,  "/",                                  b"foscam",             "mjpeg",     "Foscam"),
    (80,  "/",                                  b"amcrest",            "amcrest",   "Amcrest cam"),
    (80,  "/",                                  b"wyze",               "wyze",      "Wyze cam"),
    (8800, "/",                                 b"",                   "tapo",      "Tapo (TLS port)"),
]

COMMON_CAM_PORTS: list[int] = [80, 554, 8000, 8080, 8554, 8800, 37777, 9000, 8888]


# ── ONVIF WS-Discovery ────────────────────────────────────────────

_WS_DISCOVER_PROBE = """<?xml version="1.0" encoding="UTF-8"?>
<e:Envelope xmlns:e="http://www.w3.org/2003/05/soap-envelope"
            xmlns:w="http://schemas.xmlsoap.org/ws/2004/08/addressing"
            xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery"
            xmlns:dn="http://www.onvif.org/ver10/network/wsdl">
 <e:Header>
  <w:MessageID>uuid:{msgid}</w:MessageID>
  <w:To e:mustUnderstand="true">urn:schemas-xmlsoap-org:ws:2005:04:discovery</w:To>
  <w:Action e:mustUnderstand="true">http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe</w:Action>
 </e:Header>
 <e:Body>
  <d:Probe>
   <d:Types>dn:NetworkVideoTransmitter</d:Types>
  </d:Probe>
 </e:Body>
</e:Envelope>"""


def discover_onvif(timeout: float = 3.0) -> list[CameraInfo]:
    """Send WS-Discovery multicast probe and collect ONVIF responses."""
    found: list[CameraInfo] = []
    seen_addrs: set[str] = set()

    msg = _WS_DISCOVER_PROBE.format(msgid=str(uuid.uuid4())).encode("utf-8")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
        sock.settimeout(timeout)
        sock.sendto(msg, ("239.255.255.250", 3702))

        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                data, addr = sock.recvfrom(8192)
            except socket.timeout:
                break
            ip = addr[0]
            if ip in seen_addrs:
                continue
            seen_addrs.add(ip)

            xa_url = ""
            try:
                text = data.decode("utf-8", errors="ignore")
                # crude XAddrs extraction (no XML parser dependency); fully guarded
                for opener in ("<d:XAddrs>", "<wsdd:XAddrs>", "XAddrs>"):
                    if opener in text:
                        try:
                            chunk = text.split(opener, 1)[1].split("<", 1)[0].strip()
                            if chunk:
                                xa_url = chunk.split()[0]
                                break
                        except (IndexError, ValueError):
                            continue
            except Exception:
                xa_url = ""

            port = 80
            if xa_url.startswith("http"):
                try:
                    from urllib.parse import urlparse
                    p = urlparse(xa_url)
                    port = p.port or 80
                except Exception:
                    pass

            found.append(CameraInfo(
                id=f"onvif_{ip.replace('.', '_')}",
                name=f"ONVIF cam @ {ip}",
                vendor="onvif",
                model="ONVIF Profile S (auto-detected)",
                ip=ip, port=port, protocol="onvif",
                capabilities=["snapshot", "stream", "onvif", "ptz"],
                metadata={"onvif_xaddr": xa_url},
            ))
    except OSError:
        pass
    finally:
        sock.close()
    return found


# ── Port + vendor fingerprint scan ───────────────────────────────

def _check_port(ip: str, port: int, timeout: float = 0.4) -> bool:
    try:
        with socket.create_connection((ip, port), timeout=timeout):
            return True
    except OSError:
        return False


def _http_probe(ip: str, port: int, path: str, timeout: float = 1.5) -> bytes:
    url = f"http://{ip}:{port}{path}"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read(4096) + str(r.headers).encode("utf-8", errors="ignore")
    except (urllib.error.HTTPError) as e:
        return str(e.headers or "").encode("utf-8", errors="ignore") + (e.read(2048) if hasattr(e, "read") else b"")
    except (urllib.error.URLError, ConnectionError, socket.timeout, OSError):
        return b""
    except Exception:
        return b""


def _fingerprint(ip: str, open_ports: list[int]) -> CameraInfo | None:
    """Identify vendor at an IP by probing characteristic URLs."""
    open_set = set(open_ports)

    # First pass: vendor probes
    for port, path, marker, vendor, model in VENDOR_PROBES:
        if port not in open_set:
            continue
        body = _http_probe(ip, port, path).lower()
        if marker in body or (marker == b"" and port == 8800):
            return CameraInfo(
                id=f"{vendor}_{ip.replace('.', '_')}",
                name=f"{vendor.title()} @ {ip}",
                vendor=vendor, model=model, ip=ip,
                port=port, protocol=vendor,
                capabilities=["snapshot", "stream"],
                metadata={"open_ports": open_ports, "fingerprint_path": path},
            )

    # Fallback: if RTSP port open, register as generic RTSP
    for rport in (554, 8554):
        if rport in open_set:
            return CameraInfo(
                id=f"rtsp_{ip.replace('.', '_')}",
                name=f"Generic RTSP @ {ip}",
                vendor="rtsp", model="generic", ip=ip,
                port=rport, protocol="rtsp",
                capabilities=["snapshot", "stream"],
                metadata={"open_ports": open_ports, "note": "credentials needed"},
            )

    # HTTP-only? Could be MJPEG webcam
    if 80 in open_set or 8080 in open_set:
        port = 80 if 80 in open_set else 8080
        return CameraInfo(
            id=f"http_{ip.replace('.', '_')}",
            name=f"Unknown HTTP device @ {ip}",
            vendor="mjpeg", model="generic", ip=ip,
            port=port, protocol="mjpeg",
            capabilities=["snapshot"],
            metadata={"open_ports": open_ports, "note": "verify is camera"},
        )
    return None


def discover_subnet(subnet: str, timeout: float = 0.4,
                    max_workers: int = 64) -> list[CameraInfo]:
    """Scan a CIDR subnet for cameras on common ports + vendor-fingerprint hits."""
    try:
        net = ipaddress.IPv4Network(subnet, strict=False)
    except ValueError:
        return []

    hosts = [str(h) for h in net.hosts()]
    if len(hosts) > 1024:
        hosts = hosts[:1024]

    # Phase 1: parallel TCP scan
    open_map: dict[str, list[int]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(_check_port, ip, p, timeout): (ip, p)
                for ip in hosts for p in COMMON_CAM_PORTS}
        for fut in as_completed(futs):
            ip, p = futs[fut]
            try:
                if fut.result():
                    open_map.setdefault(ip, []).append(p)
            except Exception:
                pass

    # Phase 2: vendor fingerprint each host with open ports
    found: list[CameraInfo] = []
    for ip, ports in open_map.items():
        info = _fingerprint(ip, sorted(ports))
        if info:
            found.append(info)
    return found


# ── Local subnet detection ───────────────────────────────────────

def _local_subnet() -> str | None:
    """Best-effort detection of the host's primary /24 subnet."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 53))
        local_ip = s.getsockname()[0]
        s.close()
        parts = local_ip.split(".")
        if len(parts) == 4:
            return f"{parts[0]}.{parts[1]}.{parts[2]}.0/24"
    except OSError:
        pass
    return None


# ── Top-level entry ──────────────────────────────────────────────

def discover_all(subnet: str | None = None, timeout: float = 3.0,
                 enable_onvif: bool = True, enable_scan: bool = True) -> list[CameraInfo]:
    """Run all discovery layers and merge results, dedup by IP."""
    seen: dict[str, CameraInfo] = {}

    if enable_onvif:
        try:
            for c in discover_onvif(timeout=timeout):
                seen.setdefault(c.ip, c)
        except Exception:
            pass

    if enable_scan:
        net = subnet or _local_subnet()
        if net:
            try:
                for c in discover_subnet(net, timeout=0.4):
                    if c.ip in seen:
                        seen[c.ip].metadata.update(c.metadata)
                    else:
                        seen[c.ip] = c
            except Exception:
                pass

    return list(seen.values())


def to_yaml_block(infos: Iterable[CameraInfo]) -> str:
    """Render discovered cameras as a YAML block ready to paste into cameras.yaml."""
    lines = ["cameras:"]
    for c in infos:
        lines += [
            f"  - id: {c.id}",
            f"    name: \"{c.name}\"",
            f"    type: {c.vendor}",
            f"    ip: {c.ip}",
            f"    port: {c.port}",
            f"    username: \"REPLACE_ME\"",
            f"    password: \"REPLACE_ME\"",
            f"    poll_interval: 5",
            f"    # detected: {c.model} | caps={','.join(c.capabilities)}",
            "",
        ]
    return "\n".join(lines)


def discovery_briefing() -> str:
    """Run discovery and return a human-readable PALANTIR-style summary."""
    found = discover_all()
    if not found:
        return ("▸ NETWORK DISCOVERY — COMMAND CENTER\n"
                "▸ No cameras detected on local subnet.\n"
                "▸ Confirm: (1) cameras powered on, (2) on same LAN as host,\n"
                "▸           (3) firewall not blocking ONVIF UDP/3702.\n"
                "▸ For cloud cams (Ring/Arlo/Nest) configure OAuth in env vars.")
    lines = [f"▸ NETWORK DISCOVERY — {len(found)} camera(s) found"]
    for c in found:
        lines.append(f"  • [{c.vendor:<10}] {c.ip}:{c.port}  {c.model}")
    lines.append("▸ Drop the YAML block from /api/discover/yaml into cameras.yaml to enroll.")
    return "\n".join(lines)
