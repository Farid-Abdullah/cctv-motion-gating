"""Small helpers for reading video frames with OpenCV (no display needed).

Background, if video is new to you:
- A video is a sequence of images ("frames") shown at a fixed rate, e.g. 30 frames per second (fps).
- So frame number i is shown at time i / fps seconds. We always compute timestamps this way, because
  OpenCV's own "current time" property (CAP_PROP_POS_MSEC) is unreliable for some files.
- Videos are compressed: most frames are stored only as *differences* from earlier frames. Only every
  few seconds is there a complete "keyframe" (I-frame). To read frame 5000, the decoder must start at the
  previous I-frame and decode forward. That's why jumping around ("seeking") is slower than reading on.
- OpenCV images are NumPy arrays of shape (height, width, 3) with colour channels in **BGR** order
  (not RGB), values 0-255 (uint8).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    path: str
    fps: float
    frame_count: int
    width: int
    height: int
    duration_s: float

    def to_dict(self) -> dict:
        return asdict(self)


def open_video(path: str | Path) -> cv2.VideoCapture:
    """Open a video file, failing loudly if OpenCV can't read it."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    return cap


def video_info(path: str | Path) -> VideoInfo:
    """Frame rate, number of frames, resolution and duration of a video."""
    cap = open_video(path)
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    return VideoInfo(Path(path).name, fps, n, w, h, n / fps if fps else 0.0)


def format_hms(seconds: float, sep: str = ":") -> str:
    """1234.5 -> '00:20:34.500'. Use sep='-' for file names (':' isn't allowed on Windows)."""
    ms_total = int(round(max(seconds, 0.0) * 1000))
    h, rem = divmod(ms_total, 3_600_000)  # divmod(a, b) returns (a // b, a % b)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}{sep}{m:02d}{sep}{s:02d}.{ms:03d}"


def iter_sampled_frames(
    path: str | Path, step: int, start_frame: int = 0, end_frame: int | None = None
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (frame_idx, frame) for every `step`-th frame, e.g. step=6 turns 30 fps into 5 samples/s.

    cap.read() = cap.grab() + cap.retrieve():
      grab()     decodes the next frame but keeps it in the decoder's internal format (cheap)
      retrieve() converts that frame into a BGR NumPy array (the expensive part we want to skip)
    Every frame must be grabbed (the decoder can't skip ahead), but we retrieve only the ones we keep.
    """
    if step < 1:
        raise ValueError("step must be >= 1")
    cap = open_video(path)
    try:
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)  # seek
        idx = start_frame
        while end_frame is None or idx < end_frame:
            if not cap.grab():  # False at the end of the video
                break
            if (idx - start_frame) % step == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                yield idx, frame  # `yield` makes this a generator: frames are produced one at a time
            idx += 1
    finally:
        cap.release()


def read_frames(
    path: str | Path, frame_indices: Iterable[int], seek_threshold: int = 300
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (frame_idx, frame) for specific frame numbers, in ascending order.

    For small forward jumps we just grab() our way there (exact and cheap). For big jumps (more than
    `seek_threshold` frames, about 10 s at 30 fps) we seek, which costs one partial decode.
    """
    wanted = sorted({int(i) for i in frame_indices})
    cap = open_video(path)
    pos = 0  # number of the frame the next grab()/read() will return
    try:
        for target in wanted:
            if target < pos or target - pos > seek_threshold:
                cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                pos = target
            while pos < target:
                if not cap.grab():
                    return
                pos += 1
            ok, frame = cap.read()
            if not ok:
                return
            pos += 1
            yield target, frame
    finally:
        cap.release()
