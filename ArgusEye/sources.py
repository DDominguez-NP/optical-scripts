# Argus data sources: camera frames plus the measurements that go with them.
#
# Every image enters the analysis as a Frame. Today a Frame is built from image files on disk and
# carries no wavelength or environment data. The read_*/capture_* functions below are the hooks
# for the live station: when the Arena camera, Bristol 771 and T/P/RH logger are wired in, they
# fill the same Frame fields and the analysis code does not change.

import os
import json
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import tifffile
from PIL import Image

IMAGE_EXTENSIONS = ('.tif', '.tiff', '.png', '.npy', '.raw')

# GenICam mono pixel formats: (bits per stored pixel, significant bits, LSB-packed)
RAW_PIXEL_FORMATS = {
    'mono8': (8, 8, False),
    'mono10': (16, 10, False),
    'mono10p': (10, 10, True),
    'mono12': (16, 12, False),
    'mono12p': (12, 12, True),
    'mono16': (16, 16, False),
    'mono24': (24, 24, False),
    'mono32': (32, 32, False),
}
# Display-converted colour formats Arena can also save: (channels, bytes per channel). The camera is
# mono, so the colour channels are copies and are averaged; these are only 8 bits deep.
RAW_COLOUR_FORMATS = {
    'rgb8': (3, 1), 'bgr8': (3, 1), 'rgba8': (4, 1), 'bgra8': (4, 1),
}


@dataclass
class RawFormat:
    """How to read a headerless .raw file saved by Arena."""
    width: int = 1280
    height: int = 1024
    pixel_format: str = 'auto'        # a RAW_PIXEL_FORMATS key, or auto = guess from file size


@dataclass
class Environment:
    """Air conditions at the beam path, for the air-index correction."""
    temperature_c: float
    pressure_hpa: float
    humidity_pct: float


@dataclass
class Frame:
    """An averaged camera image and everything recorded with it."""
    image: np.ndarray                 # float64, dark/background not yet removed unless dark_subtracted
    paths: list
    bit_depth: int                    # bits per pixel of the source files
    n_frames: int
    timestamp: datetime = None
    wavelength_nm: float = None       # vacuum wavelength from the Bristol 771, when recorded
    env: Environment = None
    dark_subtracted: bool = False
    notes: list = field(default_factory=list)

    @property
    def full_scale(self):
        return 2 ** self.bit_depth - 1


def _unpack_lsb(raw_bytes, bits, count):
    """Unpacks GenICam 'p' formats: pixels stored back to back, least significant bit first."""
    bitstream = np.unpackbits(raw_bytes, bitorder='little')[:count * bits]
    return bitstream.reshape(count, bits).astype(np.uint32) @ (1 << np.arange(bits, dtype=np.uint32))


def read_raw(path, fmt):
    """
    Reads a headerless .raw file as (frames, rows, cols) plus its significant bit depth.
    Several frames back to back in one file are read as a stack.
    """
    raw = np.fromfile(path, dtype=np.uint8)
    # Arena writes <file>.raw.json with the image size and pixel format; it overrides config.ini
    sidecar = path + '.json'
    if os.path.exists(sidecar):
        with open(sidecar) as f:
            meta = json.load(f)
        fmt = RawFormat(width=int(meta.get('width', fmt.width)), height=int(meta.get('height', fmt.height)),
                        pixel_format=meta.get('pixelformat', fmt.pixel_format))
    pixels = fmt.width * fmt.height
    name = fmt.pixel_format.strip().lower()
    note = None

    if name in RAW_COLOUR_FORMATS:
        channels, depth = RAW_COLOUR_FORMATS[name]
        frame_bytes = pixels * channels * depth
        if raw.size % frame_bytes:
            raise ValueError(f"{path}: {raw.size} bytes is not a whole number of {fmt.width} x "
                             f"{fmt.height} {fmt.pixel_format} frames ({frame_bytes} bytes each)")
        data = raw.reshape(-1, fmt.height, fmt.width, channels)
        # Colour order does not matter (the channels are copies); alpha, if any, is dropped
        first = 1 if name.startswith('a') else 0
        note = (f"{fmt.pixel_format} raw file: display-converted 8-bit colour, not the camera's "
                "mono data; set PixelFormat to Mono12/Mono16 in Arena before saving")
        return data[..., first:first + 3].mean(axis=-1), 8, note
    if name == 'auto':
        # A format that makes the file exactly one frame wins; otherwise the first that divides it
        # into whole frames (ambiguous: two Mono8 frames are the size of one Mono16 frame).
        # 16-bit data is assumed to be Mono16.
        candidates = ('mono16', 'mono8', 'mono12p', 'mono10p', 'mono24', 'mono32')
        sizes = {c: pixels * RAW_PIXEL_FORMATS[c][0] for c in candidates}
        single = [c for c in candidates if raw.size * 8 == sizes[c]]
        whole = [c for c in candidates if (raw.size * 8) % sizes[c] == 0]
        if not (single or whole):
            raise ValueError(f"{path}: {raw.size} bytes is not a whole number of "
                             f"{fmt.width} x {fmt.height} frames in any known pixel format")
        name = (single or whole)[0]
        note = (f"raw pixel format guessed as {name} from the file size; set raw_pixel_format "
                "in config.ini to be sure (it sets the saturation level)")
    if name not in RAW_PIXEL_FORMATS:
        raise ValueError(f"Unknown raw pixel format '{fmt.pixel_format}'; use one of "
                         f"{', '.join(list(RAW_PIXEL_FORMATS) + list(RAW_COLOUR_FORMATS))} or auto")

    stored, significant, packed = RAW_PIXEL_FORMATS[name]
    frame_bytes = pixels * stored // 8
    if raw.size % frame_bytes:
        raise ValueError(f"{path}: {raw.size} bytes is not a whole number of {fmt.width} x "
                         f"{fmt.height} {name} frames ({frame_bytes} bytes each)")
    n_frames = raw.size // frame_bytes
    if packed:
        data = _unpack_lsb(raw, stored, pixels * n_frames)
    elif stored == 24:
        b = raw.reshape(-1, 3).astype(np.uint32)
        data = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
    else:
        data = raw.view({8: '<u1', 16: '<u2', 32: '<u4'}[stored])
    return data.reshape(n_frames, fmt.height, fmt.width).astype(np.float64), significant, note


def _read_image_stack(path, raw_format=None):
    """Reads one file as a (frames, rows, cols) float array, its bit depth and any note."""
    ext = os.path.splitext(path)[1].lower()
    if ext == '.raw':
        return read_raw(path, raw_format or RawFormat())
    if ext in ('.tif', '.tiff'):
        data = tifffile.imread(path)
        bit_depth = data.dtype.itemsize * 8
    elif ext == '.npy':
        data = np.load(path)
        bit_depth = data.dtype.itemsize * 8 if data.dtype.kind in 'ui' else 16
    elif ext == '.png':
        img = Image.open(path)
        data = np.asarray(img)
        bit_depth = 16 if data.dtype == np.uint16 or img.mode.startswith('I') else 8
    else:
        raise ValueError(f"Unsupported image format: {path}")

    data = np.asarray(data, dtype=np.float64)
    # Colour images from a mono camera: the channels are copies, so average them
    if data.ndim == 3 and data.shape[-1] in (3, 4):
        data = data[..., :3].mean(axis=-1)
    if data.ndim == 2:
        data = data[np.newaxis]
    return data, bit_depth, None


def load_frame(paths, dark_paths=None, raw_format=None):
    """
    Loads and averages one or more image files (multi-page TIFFs and multi-frame .raw files
    count every frame). If dark frames are given they are averaged and subtracted; otherwise the
    background is estimated from the image itself later, in spots.py.
    """
    if isinstance(paths, str):
        paths = [paths]
    stacks, depths, notes = zip(*(_read_image_stack(p, raw_format) for p in paths))
    stack = np.concatenate(stacks)
    frame = Frame(image=stack.mean(axis=0), paths=list(paths), bit_depth=max(depths),
                  n_frames=len(stack),
                  timestamp=datetime.fromtimestamp(os.path.getmtime(paths[0])))
    frame.notes.extend(sorted({n for n in notes if n}))
    if frame.bit_depth < 12:
        frame.notes.append(f"{frame.bit_depth}-bit input: limited dynamic range; "
                           "save Mono12/Mono16 (.raw or 16-bit TIFF) from Arena for precision work")

    if dark_paths:
        dark = load_frame(dark_paths, raw_format=raw_format)
        if dark.image.shape != frame.image.shape:
            raise ValueError("Dark frame size does not match the image")
        frame.image = frame.image - dark.image
        frame.dark_subtracted = True

    frame.wavelength_nm = read_wavelength_nm(frame.timestamp)
    frame.env = read_environment(frame.timestamp)
    return frame


# --- Hooks for the live station -------------------------------------------------------------

def capture_frame(n_frames=1, exposure_us=None, gain_db=None):
    """
    Hook: grab n_frames from the LUCID Triton through arena_api (Mono16, fixed gain/exposure,
    auto features off) and return a Frame stamped with the capture time, the Bristol wavelength
    and the T/P/RH reading.
    """
    raise NotImplementedError("Camera acquisition is not wired in yet; load saved images instead")


def capture_dark(n_frames=1, exposure_us=None, gain_db=None):
    """Hook: grab dark frames (beam blocked) at the same exposure and gain as the measurement."""
    raise NotImplementedError("Camera acquisition is not wired in yet")


def read_wavelength_nm(timestamp):
    """
    Hook: the Bristol 771 vacuum wavelength at the given time, from its live interface or log.
    Returns None while the wavemeter is not connected.
    """
    return None


def read_environment(timestamp):
    """
    Hook: temperature, pressure and humidity at the given time from the T/P/RH logger.
    Returns None while no logger is connected.
    """
    return None


def air_index(env, wavelength_nm):
    """
    Hook: refractive index of air (Ciddor) for converting between vacuum and air wavelengths.
    With no environment data the wavelength is treated as already in air (n = 1).
    """
    if env is None:
        return 1.0
    raise NotImplementedError("Ciddor air-index correction is not implemented yet")
