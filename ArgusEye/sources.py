# Argus data sources: camera frames plus the measurements that go with them.
#
# Every image enters the analysis as a Frame. Today a Frame is built from image files on disk and
# carries no wavelength or environment data. The read_*/capture_* functions below are the hooks
# for the live station: when the Arena camera, Bristol 771 and T/P/RH logger are wired in, they
# fill the same Frame fields and the analysis code does not change.

import os
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import tifffile
from PIL import Image

IMAGE_EXTENSIONS = ('.tif', '.tiff', '.png', '.npy')


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


def _read_image_stack(path):
    """Reads one file as a (frames, rows, cols) float array plus its bit depth."""
    ext = os.path.splitext(path)[1].lower()
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
    return data, bit_depth


def load_frame(paths, dark_paths=None):
    """
    Loads and averages one or more image files (multi-page TIFFs count every page as a frame).
    If dark frames are given they are averaged and subtracted; otherwise the background is
    estimated from the image itself later, in spots.py.
    """
    if isinstance(paths, str):
        paths = [paths]
    stacks, depths = zip(*(_read_image_stack(p) for p in paths))
    stack = np.concatenate(stacks)
    frame = Frame(image=stack.mean(axis=0), paths=list(paths), bit_depth=max(depths),
                  n_frames=len(stack),
                  timestamp=datetime.fromtimestamp(os.path.getmtime(paths[0])))
    if frame.bit_depth < 12:
        frame.notes.append(f"{frame.bit_depth}-bit input: limited dynamic range; "
                           "use 16-bit mono TIFF from Arena for precision work")

    if dark_paths:
        dark = load_frame(dark_paths)
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
