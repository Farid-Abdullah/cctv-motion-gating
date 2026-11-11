"""Drawing helpers for explore.ipynb: thumbnails, image grids, mask overlays, timeline plot, GIFs.

Nothing here opens a window (no cv2.imshow), so it also works on a machine without a screen.
Images are NumPy arrays in OpenCV's BGR colour order.
"""

from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from motion_gating import smooth
from video_utils import format_hms, read_frames

# Colour per keyframe reason (hex, for matplotlib)
REASON_COLORS = {"first": "#2b8cbe", "peak": "#e34a33", "change": "#31a354", "maxgap": "#756bb1", "heartbeat": "#636363"}


def hex_to_bgr(h: str) -> tuple[int, int, int]:
    r, g, b = (int(h.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    return b, g, r


# ---------------------------------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------------------------------
def thumb(img: np.ndarray, width: int) -> np.ndarray:
    """Resize to `width` pixels wide, keeping the aspect ratio."""
    h, w = img.shape[:2]
    return cv2.resize(img, (width, max(1, round(h * width / w))), interpolation=cv2.INTER_AREA)


def label(img, text, color=(255, 255, 255), bg=(0, 0, 0), scale=0.4, pos="top"):
    """Write `text` on a filled strip at the top or bottom of a copy of the image."""
    out = img.copy()
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    h = th + base + 4
    y0 = 0 if pos == "top" else out.shape[0] - h
    cv2.rectangle(out, (0, y0), (min(out.shape[1], tw + 6), y0 + h), bg, -1)
    cv2.putText(out, text, (3, y0 + th + 2), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
    return out


def border(img, color, px=3):
    return cv2.copyMakeBorder(img, px, px, px, px, cv2.BORDER_CONSTANT, value=color)


def grid(images, cols: int, pad: int = 4, bg: int = 255) -> np.ndarray:
    """Paste images into a grid with `cols` columns on a white canvas."""
    if not images:
        return np.full((10, 10, 3), bg, np.uint8)
    ch = max(i.shape[0] for i in images)
    cw = max(i.shape[1] for i in images)
    rows = (len(images) + cols - 1) // cols  # ceiling division
    canvas = np.full((rows * (ch + pad) + pad, cols * (cw + pad) + pad, 3), bg, np.uint8)
    for k, im in enumerate(images):
        r, c = divmod(k, cols)
        y, x = pad + r * (ch + pad), pad + c * (cw + pad)
        canvas[y:y + im.shape[0], x:x + im.shape[1]] = im
    return canvas


def jpeg_bytes(img, quality=80) -> bytes:
    """Encode as JPEG in memory (used to embed small images in the notebook)."""
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()


def overlay_mask(frame, mask, color=(0, 0, 255), alpha=0.55, removed=None, removed_color=(0, 220, 255)):
    """Tint the foreground pixels of a full-size frame: `mask` in red, optional `removed` in yellow.
    Masks are at processing size (320 px wide) and get scaled up first."""
    out = frame.copy()
    for mk, col in ((removed, removed_color), (mask, color)):
        if mk is None:
            continue
        # INTER_NEAREST: copy the nearest pixel when enlarging, so the mask stays strictly 0/255
        m = cv2.resize(mk, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        tint = np.zeros_like(frame)
        tint[:] = col
        out[m] = cv2.addWeighted(frame, 1 - alpha, tint, alpha, 0)[m]  # blend frame and colour
        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, col, 1)  # outline each blob
    return out


def frames_at(video_path, frame_indices) -> dict[int, np.ndarray]:
    """{frame_idx: frame} for the requested frames."""
    return dict(read_frames(video_path, frame_indices))


def contact_sheet(video_path, fps, times_s, width=200, cols=6) -> np.ndarray:
    """One labelled thumbnail per timestamp, in a grid."""
    idx = [int(round(t * fps)) for t in times_s]
    frames = frames_at(video_path, idx)
    return grid([label(thumb(frames[i], width), format_hms(i / fps)[:8]) for i in idx if i in frames], cols)


# ---------------------------------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------------------------------
def plot_timeline(ax, scores: pd.DataFrame, score_col, segments, start_threshold, stop_threshold,
                  smooth_window=1, keyframes=None, title=""):
    """Motion score over time (x in minutes, y in % of pixels), thresholds as horizontal lines,
    segments shaded, keyframes as coloured ticks under the axis."""
    t = scores["timestamp_s"].to_numpy() / 60
    y = scores[score_col].to_numpy() * 100
    ax.plot(t, y, lw=0.4, color="#9ecae1", label="raw score")
    if smooth_window > 1:
        ax.plot(t, smooth(y, smooth_window), lw=0.8, color="#08519c", label="smoothed")
    for seg in segments.itertuples(index=False):
        ax.axvspan(seg.start_s / 60, seg.end_s / 60, color="#fdae6b", alpha=0.35, lw=0)
    ax.axhline(start_threshold * 100, color="#d62728", lw=0.9, ls="--", label=f"start {start_threshold * 100:.2f}%")
    ax.axhline(stop_threshold * 100, color="#ff7f0e", lw=0.9, ls=":", label=f"stop {stop_threshold * 100:.2f}%")
    top = max(np.percentile(y, 99.8) * 1.15, start_threshold * 200)
    if keyframes is not None and len(keyframes):
        for reason, g in keyframes.groupby("reason"):
            ax.scatter(g["timestamp_s"] / 60, np.full(len(g), -0.06 * top), marker="|", s=60,
                       color=REASON_COLORS.get(reason, "k"), label=f"kf: {reason} ({len(g)})", clip_on=False)
    ax.set_ylim(-0.1 * top, top)
    ax.set_xlim(t.min(), t.max())
    ax.set_ylabel("moving pixels (%)")
    ax.set_xlabel("video time (min)")
    ax.set_title(title, loc="left", fontsize=10)
    ax.grid(alpha=0.25, lw=0.5)


def fig_png_bytes(fig, dpi=72) -> bytes:
    """Render a matplotlib figure to PNG bytes (small dpi keeps the notebook file small)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    return buf.getvalue()


# ---------------------------------------------------------------------------------------------------
# GIFs
# ---------------------------------------------------------------------------------------------------
def segment_gif(video_path, fps, start_s, end_s, out_path, width=256, sample_fps=2.0, play_fps=10.0,
                max_frames=40, colors=64, caption=""):
    """Save a small sped-up GIF of [start_s, end_s] (at most `max_frames` frames around the middle).
    2 frames per second of video played at 10 fps = 5x speed. Returns (path, size_bytes, speedup)."""
    from PIL import Image

    if (end_s - start_s) * sample_fps > max_frames:
        mid = (start_s + end_s) / 2
        start_s, end_s = mid - max_frames / sample_fps / 2, mid + max_frames / sample_fps / 2
    idx = [int(round(t * fps)) for t in np.arange(start_s, end_s, 1 / sample_fps)]
    frames = frames_at(video_path, idx)
    pil = [Image.fromarray(cv2.cvtColor(label(thumb(frames[i], width), f"{caption} {format_hms(i / fps)[:8]}".strip(),
                                              scale=0.35), cv2.COLOR_BGR2RGB))  # PIL wants RGB
           for i in idx if i in frames]
    # GIFs can only use a palette of <= 256 colours. One shared palette for all frames keeps the file
    # small and avoids flickering colours.
    palette = pil[len(pil) // 2].quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
    q = [p.quantize(palette=palette, dither=Image.Dither.NONE) for p in pil]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    q[0].save(out_path, save_all=True, append_images=q[1:], duration=int(1000 / play_fps), loop=0, optimize=True)
    return out_path, out_path.stat().st_size, play_fps / sample_fps
