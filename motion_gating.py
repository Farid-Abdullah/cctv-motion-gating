"""Stage 1: motion gating. Find the time ranges of a fixed-camera video where something moves.

Run:  python motion_gating.py [video]          (default: video/meva_30min.mp4)
Writes outputs/<video name>/motion_scores.csv and segments.csv (+ a cache of the slow decode pass).

Steps:
  1. scan_video     - decode 5 frames per second of video; for each, compute two motion scores:
                        score_diff : frame differencing (what changed since the previous sample)
                        score_mog2 : background subtraction with OpenCV's MOG2 (what differs from the
                                     learned "empty scene"), with a filter that removes lighting changes
                      Each score = fraction of pixels marked as moving ("foreground").
                      The result is cached, so changing thresholds later costs no decoding.
  2. thresholds     - fixed number, or adaptive: median + k * MAD of the scores
  3. find_segments  - smooth -> hysteresis -> merge short gaps -> drop short segments -> pad
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from keyframes import gray_hist, tile_phash
from video_utils import format_hms, iter_sampled_frames, video_info

ROOT = Path(__file__).resolve().parent  # folder containing this file
DEFAULT_VIDEO = ROOT / "video" / "meva_30min.mp4"


def output_dir(video_path: str | Path) -> Path:
    """outputs/<video name without extension>/, so several videos don't overwrite each other."""
    return ROOT / "outputs" / Path(video_path).stem


# Default parameters (the notebook's config cell uses the same values)
SCAN_PARAMS = dict(
    sample_fps=5.0,           # frames analysed per second of video (the video has 30)
    width=320,                # shrink frames to this width before analysis
    blur_ksize=5,             # Gaussian blur kernel size (px), smooths out pixel noise
    diff_threshold=15,        # frame differencing: a pixel "moved" if it changed by > 15 grey levels (of 255)
    mog2_history=500,         # MOG2 remembers the last 500 samples (= 100 s) as "normal background"
    mog2_var_threshold=16.0,  # MOG2 sensitivity: lower = more pixels count as foreground
    mog2_learning_rate=-1.0,  # how fast the background adapts; -1 = automatic (1 / history)
    mog2_illum_ncc=0.7,       # lighting filter: drop blobs whose texture still matches the background (None = off)
    morph_kernel=3,           # size of the clean-up filter that removes isolated noise pixels
    ignore_polygons=[],       # regions to ignore, e.g. [[(0,0),(1,0),(1,0.22),(0,0.22)]] = top 22% (sky)
    hash_grid=4,              # keyframe fingerprints: 4x4 tiles (see keyframes.tile_phash)
    hist_bins=32,
)
SEGMENT_PARAMS = dict(
    method="mog2",            # which score drives segments and keyframes: "mog2" or "diff"
    threshold_mode="adaptive",  # "adaptive" (median + k*MAD) or "fixed"
    k=3.0,                    # adaptive: how many robust standard deviations above normal
    floor=0.001,              # adaptive threshold never below 0.1% of pixels
    fixed_threshold={"mog2": 0.004, "diff": 0.006},  # used when threshold_mode="fixed"
    stop_ratio=0.5,           # hysteresis: stop threshold = 0.5 x start threshold
    smooth_s=1.0,             # rolling median over 1 s before thresholding
    merge_gap_s=3.0,          # join segments separated by <= 3 s
    min_duration_s=1.0,       # drop segments shorter than 1 s
    pad_s=2.0,                # add 2 s before and after each segment
)


# ---------------------------------------------------------------------------------------------------
# Per-frame processing
# ---------------------------------------------------------------------------------------------------
def preprocess(frame: np.ndarray, width: int = 320, blur_ksize: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Shrink, convert to greyscale, blur. Returns (gray, blurred).

    Why: motion detection doesn't need colour or full resolution. 320x180 has 16x fewer pixels than
    1280x720, so it's much faster, and averaging pixels also averages away sensor/compression noise.
    The Gaussian blur (a weighted average over a 5x5 neighbourhood) removes more of that noise, so
    single flickering pixels don't count as motion.
    """
    h, w = frame.shape[:2]
    height = max(2, int(round(h * width / w / 2)) * 2)  # keep aspect ratio, even number of rows
    small = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)  # INTER_AREA = good for shrinking
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0) if blur_ksize > 1 else gray
    return gray, blurred


def make_ignore_mask(width: int, height: int, polygons: list | None) -> np.ndarray | None:
    """Mask of pixels to use (255) vs ignore (0), from polygons in 0..1 coordinates
    ((0,0) = top-left, (1,1) = bottom-right), so the same polygon works at any resolution."""
    if not polygons:
        return None
    mask = np.full((height, width), 255, np.uint8)
    for poly in polygons:
        pts = np.array([[round(x * (width - 1)), round(y * (height - 1))] for x, y in poly], np.int32)
        cv2.fillPoly(mask, [pts], 0)
    return mask


def remove_illumination_blobs(fg: np.ndarray, gray: np.ndarray, background: np.ndarray,
                              ncc_threshold: float = 0.7, min_pixels: int = 8) -> np.ndarray:
    """Remove foreground blobs that are only lighting changes (cloud shadows, sunlight).

    Idea: when a cloud passes, a patch of grass gets darker but still looks like the *same grass*
    (same texture). When a person walks in, the texture is replaced. So for each blob we compare the
    current pixels with the background model's pixels using normalised cross-correlation (NCC):
      - subtract each patch's mean (removes the overall brighter/darker shift)
      - NCC = sum(a*b) / sqrt(sum(a^2) * sum(b^2)), between -1 and 1; 1 = same pattern
    NCC above the threshold -> same texture -> lighting -> remove the blob.
    On the MEVA video this cut a sunlit-lawn false alarm from 9.3% to 0.3% of pixels while people
    kept ~90-100% of their foreground.
    """
    if background.ndim == 3:
        background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    # Connected components: give every separate white blob in the mask its own label number 1..n-1
    # (0 = background); stats has each blob's bounding box (x, y, w, h) and area.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    keep = np.ones(n, dtype=bool)
    keep[0] = False
    for k in range(1, n):
        x, y, w, h, area = stats[k]
        if area < min_pixels:  # too small to judge texture; keep it
            continue
        box = (slice(y, y + h), slice(x, x + w))
        m = labels[box] == k  # this blob's pixels inside its bounding box
        a = gray[box][m].astype(np.float32)
        b = background[box][m].astype(np.float32)
        a -= a.mean()
        b -= b.mean()
        den = float(np.sqrt((a * a).sum() * (b * b).sum()))
        ncc = float((a * b).sum()) / den if den > 1e-6 else 1.0  # both perfectly flat -> treat as lighting
        if ncc > ncc_threshold:
            keep[k] = False
    # keep[labels] looks up each pixel's blob label in the keep table -> boolean image
    return np.where(keep[labels], 255, 0).astype(np.uint8)


class MotionScorer:
    """Turns a stream of preprocessed frames into two foreground masks (0 = still, 255 = moving).

    A class (rather than a function) because both methods need memory of earlier frames.
    """

    def __init__(self, diff_threshold=15, mog2_history=500, mog2_var_threshold=16.0, mog2_learning_rate=-1.0,
                 morph_kernel=3, ignore_mask=None, mog2_illum_ncc=0.7):
        self.diff_threshold = diff_threshold
        # MOG2 = "Mixture of Gaussians" background subtraction. For every pixel it keeps a few Gaussian
        # (bell-curve) models of the values that pixel usually has, learned over the last `history`
        # frames. A new value that fits none of them -> foreground. Because it learns, slow changes
        # (sun moving) are absorbed into the background, and so are objects that stop moving long enough.
        # detectShadows=True marks likely shadows with the value 127 instead of 255.
        self.mog2 = cv2.createBackgroundSubtractorMOG2(history=mog2_history, varThreshold=mog2_var_threshold,
                                                       detectShadows=True)
        self.learning_rate = mog2_learning_rate
        self.illum_ncc = mog2_illum_ncc
        # Morphological "opening" = erosion (shrink white areas, so tiny specks vanish) followed by
        # dilation (grow back what survived). Removes isolated noise pixels but keeps real blobs.
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel)) if morph_kernel > 1 else None
        self.ignore_mask = ignore_mask
        self.prev = None                 # previous frame, for frame differencing
        self.last_illum_removed = None   # blobs the lighting filter removed (for the notebook overlays)

    def _clean(self, fg):
        if self.kernel is not None:
            fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self.kernel)
        if self.ignore_mask is not None:
            fg = cv2.bitwise_and(fg, self.ignore_mask)
        return fg

    def update(self, blurred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Feed the next frame; returns (fg_diff, fg_mog2)."""
        # Frame differencing: pixels whose brightness changed by more than the threshold since the
        # previous sample. Simple and fast, but only sees things that are moving *right now*.
        if self.prev is None:
            fg_diff = np.zeros_like(blurred)
        else:
            _, fg_diff = cv2.threshold(cv2.absdiff(blurred, self.prev), self.diff_threshold, 255, cv2.THRESH_BINARY)
        self.prev = blurred

        raw = self.mog2.apply(blurred, learningRate=self.learning_rate)
        _, fg_mog2 = cv2.threshold(raw, 200, 255, cv2.THRESH_BINARY)  # 255 stays, shadows (127) -> 0
        fg_mog2 = self._clean(fg_mog2)
        self.last_illum_removed = None
        if self.illum_ncc is not None and cv2.countNonZero(fg_mog2):
            kept = remove_illumination_blobs(fg_mog2, blurred, self.mog2.getBackgroundImage(), self.illum_ncc)
            self.last_illum_removed = cv2.subtract(fg_mog2, kept)
            fg_mog2 = kept
        return self._clean(fg_diff), fg_mog2

    def valid_pixels(self, shape) -> int:
        return int(np.count_nonzero(self.ignore_mask)) if self.ignore_mask is not None else shape[0] * shape[1]


# ---------------------------------------------------------------------------------------------------
# The slow pass (decode the whole video once) + cache
# ---------------------------------------------------------------------------------------------------
def _cache_key(video_path: Path, params: dict) -> str:
    """Short fingerprint of (video, parameters): change any parameter and you get a new cache file."""
    ident = {"video": video_path.name, "size": video_path.stat().st_size, **params}
    return hashlib.sha1(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:12]


def scan_video(video_path: str | Path, *, sample_fps=5.0, width=320, blur_ksize=5, diff_threshold=15,
               mog2_history=500, mog2_var_threshold=16.0, mog2_learning_rate=-1.0, mog2_illum_ncc=0.7,
               morph_kernel=3, ignore_polygons=None, hash_grid=4, hist_bins=32,
               cache_dir: str | Path | None = None) -> dict:
    """Decode the video once and compute, for every sample: both motion scores, mean brightness, and
    the image fingerprints keyframes.py needs. Cached in cache_dir/scan_<key>.npz.

    Returns a dict:
      scores  DataFrame: sample, frame_idx, timestamp_s, score_diff, score_mog2, mean_luma
      sig     {"phash": (N, 16) uint64, "hist": (N, 32) float32} fingerprints per sample
      params  the scan parameters;  info  video info;  stats  timing (from_cache = was it loaded?)
    """
    video_path = Path(video_path)
    info = video_info(video_path)
    params = dict(sample_fps=sample_fps, width=width, blur_ksize=blur_ksize, diff_threshold=diff_threshold,
                  mog2_history=mog2_history, mog2_var_threshold=mog2_var_threshold,
                  mog2_learning_rate=mog2_learning_rate, mog2_illum_ncc=mog2_illum_ncc, morph_kernel=morph_kernel,
                  ignore_polygons=[[list(map(float, p)) for p in poly] for poly in (ignore_polygons or [])],
                  hash_grid=hash_grid, hist_bins=hist_bins)
    cols = ["sample", "frame_idx", "timestamp_s", "score_diff", "score_mog2", "mean_luma"]

    cache_file = Path(cache_dir) / f"scan_{_cache_key(video_path, params)}.npz" if cache_dir else None
    if cache_file and cache_file.exists():
        t0 = time.perf_counter()
        z = np.load(cache_file, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        return {"scores": pd.DataFrame({c: z[c] for c in cols}), "sig": {"phash": z["phash"], "hist": z["hist"]},
                "params": meta["params"], "info": meta["info"],
                "stats": {**meta["stats"], "from_cache": True, "load_s": time.perf_counter() - t0}}

    step = max(1, int(round(info.fps / sample_fps)))  # 30 fps / 5 = every 6th frame
    frame_idx, diff, mog, luma, phash, hist = [], [], [], [], [], []
    scorer = None
    t0 = time.perf_counter()
    for idx, frame in iter_sampled_frames(video_path, step):
        gray, blurred = preprocess(frame, width, blur_ksize)
        if scorer is None:  # created on the first frame, once we know the processed size
            mask = make_ignore_mask(gray.shape[1], gray.shape[0], params["ignore_polygons"])
            scorer = MotionScorer(diff_threshold, mog2_history, mog2_var_threshold, mog2_learning_rate,
                                  morph_kernel, mask, mog2_illum_ncc)
            n_valid = scorer.valid_pixels(gray.shape)
        fg_d, fg_m = scorer.update(blurred)
        frame_idx.append(idx)
        diff.append(np.count_nonzero(fg_d) / n_valid)  # fraction of pixels moving
        mog.append(np.count_nonzero(fg_m) / n_valid)
        luma.append(float(gray.mean()))
        phash.append(tile_phash(gray, hash_grid))
        hist.append(gray_hist(gray, hist_bins))
    elapsed = time.perf_counter() - t0

    fi = np.asarray(frame_idx, np.int64)
    scores = pd.DataFrame({"sample": np.arange(len(fi)), "frame_idx": fi, "timestamp_s": fi / info.fps,
                           "score_diff": diff, "score_mog2": mog, "mean_luma": luma})
    sig = {"phash": np.stack(phash), "hist": np.stack(hist)}
    decoded = int(fi[-1]) + 1  # every frame up to the last sample had to be decoded
    stats = {"from_cache": False, "scan_s": elapsed, "step": step, "effective_sample_fps": info.fps / step,
             "frames_decoded": decoded, "samples": len(fi),
             "decoded_fps": decoded / elapsed, "samples_per_s": len(fi) / elapsed}
    if cache_file:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        meta = json.dumps({"params": params, "stats": stats, "info": info.to_dict()})
        np.savez_compressed(cache_file, meta=np.array(meta), phash=sig["phash"], hist=sig["hist"],
                            **{c: scores[c].to_numpy() for c in cols})
    return {"scores": scores, "sig": sig, "params": params, "info": info.to_dict(), "stats": stats}


# ---------------------------------------------------------------------------------------------------
# Thresholds and segments
# ---------------------------------------------------------------------------------------------------
def robust_threshold(scores, k: float = 3.0, floor: float = 0.0) -> float:
    """Adaptive threshold = median + k * 1.4826 * MAD  (never below `floor`).

    MAD (median absolute deviation) = median(|x - median(x)|), a measure of spread like the standard
    deviation, but big spikes (the motion we're looking for) barely affect it. 1.4826 * MAD equals
    the standard deviation for normally distributed data, so k=3 means "3 sigma above normal".
    Assumes most of the video is quiet, so the median describes "nothing happening".
    """
    x = np.asarray(scores, dtype=float)
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    return max(floor, med + k * 1.4826 * mad)


def start_stop_thresholds(scores: pd.DataFrame, which: str, *, threshold_mode="adaptive", k=3.0, floor=0.001,
                          fixed_threshold=None, stop_ratio=0.5, **_ignored) -> tuple[float, float]:
    """(start, stop) thresholds for the score column score_<which> ("diff" or "mog2").
    **_ignored swallows extra keys, so you can pass the whole SEGMENT_PARAMS dict."""
    if threshold_mode == "adaptive":
        start = robust_threshold(scores[f"score_{which}"], k, floor)
    elif threshold_mode == "fixed":
        start = (fixed_threshold or {})[which]
    else:
        raise ValueError(threshold_mode)
    return start, start * stop_ratio


def smooth(scores, window: int) -> np.ndarray:
    """Rolling median over `window` samples: each value becomes the median of its neighbourhood, which
    removes single-sample spikes but keeps real rises and falls sharp (unlike a mean)."""
    x = pd.Series(np.asarray(scores, dtype=float))
    return x.to_numpy().copy() if window <= 1 else x.rolling(window, center=True, min_periods=1).median().to_numpy()


def hysteresis(scores: np.ndarray, start_threshold: float, stop_threshold: float) -> np.ndarray:
    """Active/inactive per sample, with two thresholds (like a thermostat):
    switch ON when the score rises above `start`, switch OFF only when it falls below the lower `stop`.
    With a single threshold, a score hovering around it would flicker on/off many times."""
    active = np.zeros(len(scores), dtype=bool)
    on = False
    for i, v in enumerate(scores):
        on = v >= stop_threshold if on else v >= start_threshold
        active[i] = on
    return active


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[(first, last), ...] index ranges where mask is True (found where the 0/1 sequence steps up/down)."""
    if not mask.any():
        return []
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1).tolist(), (np.flatnonzero(d == -1) - 1).tolist()))


def _merge(intervals: list[list[float]], gap: float) -> list[list[float]]:
    """Join intervals that are at most `gap` seconds apart."""
    out: list[list[float]] = []
    for a, b in sorted(intervals):
        if out and a - out[-1][1] <= gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def find_segments(scores: pd.DataFrame, score_col: str, *, start_threshold: float, stop_threshold: float,
                  duration_s: float, smooth_s=1.0, merge_gap_s=3.0, min_duration_s=1.0, pad_s=2.0) -> pd.DataFrame:
    """Turn per-sample scores into active time ranges:
    smooth -> hysteresis -> merge gaps <= merge_gap_s -> drop shorter than min_duration_s -> pad -> merge overlaps.
    Returns segment_id, start_s, end_s, duration_s, start_hms, end_hms, peak_score, peak_time_s,
    peak_frame_idx, mean_score (peak/mean use the unsmoothed score)."""
    t = scores["timestamp_s"].to_numpy()
    raw = scores[score_col].to_numpy()
    period = float(np.median(np.diff(t)))  # time between samples (0.2 s)
    active = hysteresis(smooth(raw, max(1, round(smooth_s / period))), start_threshold, stop_threshold)

    ivs = [[t[a], t[b] + period] for a, b in _runs(active)]
    ivs = _merge(ivs, merge_gap_s)
    ivs = [iv for iv in ivs if iv[1] - iv[0] >= min_duration_s]
    ivs = _merge([[max(0.0, a - pad_s), min(duration_s, b + pad_s)] for a, b in ivs], 0.0)

    rows = []
    for a, b in ivs:
        sel = np.flatnonzero((t >= a) & (t < b))
        p = sel[np.argmax(raw[sel])]
        rows.append({"segment_id": len(rows), "start_s": round(a, 3), "end_s": round(b, 3),
                     "duration_s": round(b - a, 3), "start_hms": format_hms(a), "end_hms": format_hms(b),
                     "peak_score": float(raw[p]), "peak_time_s": round(float(t[p]), 3),
                     "peak_frame_idx": int(scores["frame_idx"].iat[p]), "mean_score": float(raw[sel].mean())})
    cols = ["segment_id", "start_s", "end_s", "duration_s", "start_hms", "end_hms", "peak_score", "peak_time_s",
            "peak_frame_idx", "mean_score"]
    return pd.DataFrame(rows, columns=cols)


def segment_both_methods(scores: pd.DataFrame, duration_s: float, **segment_params) -> dict:
    """Thresholds + segments for both scores: {"diff": {...}, "mog2": {...}}, each with start, stop, segments."""
    kw = {k: segment_params[k] for k in ("smooth_s", "merge_gap_s", "min_duration_s", "pad_s")}
    out = {}
    for m in ("diff", "mog2"):
        start, stop = start_stop_thresholds(scores, m, **segment_params)
        segs = find_segments(scores, f"score_{m}", start_threshold=start, stop_threshold=stop,
                             duration_s=duration_s, **kw)
        out[m] = {"col": f"score_{m}", "start": start, "stop": stop, "segments": segs}
    return out


def active_mask(times: np.ndarray, segments: pd.DataFrame) -> np.ndarray:
    """True for each time that falls inside a segment."""
    m = np.zeros(len(times), dtype=bool)
    for seg in segments.itertuples(index=False):
        m |= (times >= seg.start_s) & (times < seg.end_s)
    return m


def active_fraction(segments: pd.DataFrame, duration_s: float) -> float:
    return float(segments["duration_s"].sum() / duration_s) if len(segments) else 0.0


def save_results(out_dir: Path, scores: pd.DataFrame, results: dict, method: str) -> None:
    """Write motion_scores.csv (every sample, both scores, active flags) and segments.csv (chosen method)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ms = scores.copy()
    ms.insert(3, "timestamp_hms", ms["timestamp_s"].map(format_hms))
    for m, r in results.items():
        ms[f"active_{m}"] = active_mask(ms["timestamp_s"].to_numpy(), r["segments"])
    ms.to_csv(out_dir / "motion_scores.csv", index=False, float_format="%.6g")
    results[method]["segments"].assign(method=method).to_csv(out_dir / "segments.csv", index=False)


# ---------------------------------------------------------------------------------------------------
# For the notebook: show what triggered a segment
# ---------------------------------------------------------------------------------------------------
def foreground_mask_at(video_path, frame_idx: int, scan_params: dict, method="mog2", warmup_s=60.0):
    """Recompute the foreground mask at one frame. MOG2 needs history, so it is first fed the previous
    `warmup_s` seconds (the result is close to, not identical with, the full pass).
    Returns (full-res frame, mask, removed_by_lighting_filter or None); masks are at processing size."""
    info = video_info(video_path)
    p = scan_params
    step = max(1, int(round(info.fps / p["sample_fps"])))
    start = max(0, frame_idx - (step if method == "diff" else int(warmup_s * info.fps)))
    start -= start % step  # stay on the same 1-in-6 frame grid as the scan
    scorer, frame, fg, removed = None, None, None, None
    for _, frame in iter_sampled_frames(video_path, step, start_frame=start, end_frame=frame_idx + 1):
        gray, blurred = preprocess(frame, p["width"], p["blur_ksize"])
        if scorer is None:
            mask = make_ignore_mask(gray.shape[1], gray.shape[0], p.get("ignore_polygons"))
            scorer = MotionScorer(p["diff_threshold"], p["mog2_history"], p["mog2_var_threshold"],
                                  p["mog2_learning_rate"], p["morph_kernel"], mask, p.get("mog2_illum_ncc"))
        fg_d, fg_m = scorer.update(blurred)
        fg, removed = (fg_d, None) if method == "diff" else (fg_m, scorer.last_illum_removed)
    return frame, fg, removed


# ---------------------------------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 1: motion gating.")
    ap.add_argument("video", nargs="?", default=str(DEFAULT_VIDEO))
    a = ap.parse_args()
    video = Path(a.video)
    out = output_dir(video)

    scan = scan_video(video, **SCAN_PARAMS, cache_dir=out / "cache")
    st = scan["stats"]
    print("loaded cached scan" if st["from_cache"] else
          f"scanned {st['frames_decoded']} frames in {st['scan_s']:.1f} s ({st['decoded_fps']:.0f} frames/s)")
    results = segment_both_methods(scan["scores"], scan["info"]["duration_s"], **SEGMENT_PARAMS)
    method = SEGMENT_PARAMS["method"]
    save_results(out, scan["scores"], results, method)
    for m, r in results.items():
        print(f"{m:5s}: {len(r['segments'])} segments, {active_fraction(r['segments'], scan['info']['duration_s']):.1%} "
              f"of video active" + ("  <- used" if m == method else ""))
    print(f"wrote {out / 'motion_scores.csv'} and {out / 'segments.csv'}")


if __name__ == "__main__":
    main()
