"""OpenEXR I/O.

Reading uses the official `OpenEXR` Python bindings when installed (handles
every compression type) and otherwise falls back to a small pure-NumPy reader
for the uncompressed files the renderer writes. Writing is always the pure-NumPy
uncompressed writer, so no native dependency is required.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

try:  # OpenEXR >= 3.3 exposes OpenEXR.File
    import OpenEXR as _openexr

    _HAS_OPENEXR = hasattr(_openexr, "File")
except ImportError:  # pragma: no cover - depends on environment
    _HAS_OPENEXR = False

_PIXEL_TYPES = {0: np.uint32, 1: np.float16, 2: np.float32}


def _read_cstr(buf: bytes, pos: int) -> tuple[str, int]:
    end = buf.index(b"\0", pos)
    return buf[pos:end].decode("latin-1"), end + 1


def _read_exr_numpy(path: Path) -> dict[str, np.ndarray]:
    buf = path.read_bytes()
    magic, version = struct.unpack_from("<ii", buf, 0)
    if magic != 20000630:
        raise ValueError(f"{path}: not an OpenEXR file")
    if version & 0x200 or version & 0x1000:
        raise ValueError(f"{path}: tiled/multipart EXR not supported without the OpenEXR package")
    pos = 8
    channels: list[tuple[str, int]] = []
    compression = 0
    data_window = None
    while True:
        name, pos = _read_cstr(buf, pos)
        if name == "":
            break
        _type, pos = _read_cstr(buf, pos)
        (size,) = struct.unpack_from("<i", buf, pos)
        pos += 4
        value = buf[pos:pos + size]
        pos += size
        if name == "channels":
            p = 0
            while value[p] != 0:
                ch, p = _read_cstr(value, p)
                (ptype,) = struct.unpack_from("<i", value, p)
                channels.append((ch, ptype))
                p += 16
        elif name == "compression":
            compression = value[0]
        elif name == "dataWindow":
            data_window = struct.unpack("<iiii", value)
    if compression != 0:
        raise ValueError(f"{path}: compressed EXR needs the OpenEXR package (pip install OpenEXR)")
    xmin, ymin, xmax, ymax = data_window
    w, h = xmax - xmin + 1, ymax - ymin + 1
    offsets = np.frombuffer(buf, dtype="<u8", count=h, offset=pos)
    out = {name: np.empty((h, w), dtype=_PIXEL_TYPES[pt]) for name, pt in channels}
    for off in offsets:
        y, _size = struct.unpack_from("<ii", buf, int(off))
        p = int(off) + 8
        row = y - ymin
        for name, ptype in channels:
            dt = np.dtype(_PIXEL_TYPES[ptype]).newbyteorder("<")
            out[name][row] = np.frombuffer(buf, dtype=dt, count=w, offset=p)
            p += w * dt.itemsize
    return out


def read_exr_channels(path: str | Path) -> dict[str, np.ndarray]:
    """Returns {channel_name: (H, W) array}. Grouped layers are flattened."""
    path = Path(path)
    if _HAS_OPENEXR:
        with _openexr.File(str(path), separate_channels=True) as f:
            return {k: np.array(v.pixels) for k, v in f.channels().items()}
    return _read_exr_numpy(path)


def read_exr(path: str | Path) -> np.ndarray:
    """Reads an EXR as float32 (H, W, C). RGB files return channels in R, G, B
    order; single-channel files return (H, W, 1)."""
    ch = read_exr_channels(path)
    if all(k in ch for k in ("R", "G", "B")):
        arr = np.stack([ch["R"], ch["G"], ch["B"]], axis=-1)
    elif len(ch) == 1:
        arr = next(iter(ch.values()))[..., None]
    else:
        arr = np.stack([ch[k] for k in sorted(ch)], axis=-1)
    return arr.astype(np.float32, copy=False)


def write_exr(path: str | Path, image: np.ndarray, channel_names: list[str] | None = None) -> None:
    """Writes a float32 uncompressed scanline EXR. `image` is (H, W) or (H, W, C)."""
    img = np.asarray(image, dtype=np.float32)
    if img.ndim == 2:
        img = img[..., None]
    h, w, c = img.shape
    if channel_names is None:
        channel_names = {1: ["Y"], 3: ["R", "G", "B"], 4: ["R", "G", "B", "A"]}.get(c)
        if channel_names is None:
            raise ValueError("channel_names required for this channel count")
    order = sorted(range(c), key=lambda i: channel_names[i])

    def attr(name: str, typ: str, payload: bytes) -> bytes:
        return name.encode() + b"\0" + typ.encode() + b"\0" + struct.pack("<i", len(payload)) + payload

    chlist = b"".join(channel_names[i].encode() + b"\0" + struct.pack("<iBBBBii", 2, 0, 0, 0, 0, 1, 1) for i in order) + b"\0"
    window = struct.pack("<iiii", 0, 0, w - 1, h - 1)
    header = struct.pack("<ii", 20000630, 2)
    header += attr("channels", "chlist", chlist)
    header += attr("compression", "compression", b"\0")
    header += attr("dataWindow", "box2i", window)
    header += attr("displayWindow", "box2i", window)
    header += attr("lineOrder", "lineOrder", b"\0")
    header += attr("pixelAspectRatio", "float", struct.pack("<f", 1.0))
    header += attr("screenWindowCenter", "v2f", struct.pack("<ff", 0.0, 0.0))
    header += attr("screenWindowWidth", "float", struct.pack("<f", 1.0))
    header += b"\0"

    line_bytes = c * w * 4
    start = len(header) + 8 * h
    offsets = start + np.arange(h, dtype="<u8") * (8 + line_bytes)
    planar = np.ascontiguousarray(img[:, :, order].transpose(0, 2, 1)).astype("<f4")  # (H, C, W)
    rows = np.empty((h, 8 + line_bytes), dtype=np.uint8)
    rows[:, 0:4] = np.arange(h, dtype="<i4").view(np.uint8).reshape(h, 4)
    rows[:, 4:8] = np.frombuffer(struct.pack("<i", line_bytes), dtype=np.uint8)
    rows[:, 8:] = planar.reshape(h, -1).view(np.uint8)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(header)
        f.write(offsets.tobytes())
        f.write(rows.tobytes())
