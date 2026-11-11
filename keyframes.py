"""Stage 2: keyframe extraction. Pick a few informative frames from each active segment.

Run:  python keyframes.py [video]      (run motion_gating.py first)
Writes outputs/<video name>/keyframes/*.jpg and keyframes.csv.

Rules, applied to the 5-samples/s frames inside each segment:
  first     - the first frame of the segment
  peak      - the frame with the highest motion score
  change    - the scene changed enough since the last kept keyframe (and at least `min_gap_s` passed)
  maxgap    - nothing changed enough, but `max_gap_s` passed, so keep one anyway
  heartbeat - outside segments (idle time): one frame every `heartbeat_s`, so slow changes
              (a parked car, lighting) still get a frame
Then near-duplicates (almost the same image as the previous keyframe) are dropped.

To measure "how much did the scene change" without comparing millions of pixels we use a perceptual
hash: a tiny fingerprint of an image's coarse structure. Fingerprints are computed once per sample in
motion_gating.scan_video (during the single decode pass) and cached, so this stage doesn't decode the
video again except to save the chosen frames.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from video_utils import format_hms, read_frames

# Default parameters (the notebook's config cell uses the same values)
KEYFRAME_PARAMS = dict(
    metric="phash",          # "phash" (tile hashes) or "hist" (grey-level histogram)
    change_threshold=6.0,    # phash: mean differing bits per tile (out of 64) to count as "changed"
    dedup_threshold=2.5,     # closer than this to the previous keyframe = duplicate
    min_gap_s=3.0,           # at least this long between keyframes in a segment
    max_gap_s=30.0,          # at most this long without a keyframe in a segment
    heartbeat_s=300.0,       # one keyframe per 5 min of idle time (0 = off)
)

# When two candidate keyframes are duplicates, keep the one with the more important reason
REASON_PRIORITY = {"peak": 0, "first": 1, "change": 2, "maxgap": 3, "heartbeat": 4}
IDLE_SEGMENT_ID = -1  # segment_id used for heartbeat frames


# ---------------------------------------------------------------------------------------------------
# Image fingerprints
# ---------------------------------------------------------------------------------------------------
def tile_phash(gray: np.ndarray, grid: int = 4) -> np.ndarray:
    """Perceptual hash (pHash) of each tile in a grid x grid split of a greyscale image.

    pHash in 4 steps:
      1. shrink the tile to 32x32 (throws away fine detail and noise)
      2. DCT (discrete cosine transform): re-express the image as a sum of wave patterns, from
         smooth/coarse (top-left of the result) to fine detail (bottom-right). JPEG uses the same idea.
      3. keep the 8x8 coarsest coefficients = the "overall structure" of the tile
      4. bit = 1 where a coefficient is above the median, else 0  -> 64 bits = one uint64 number
    Similar-looking images get hashes that differ in only a few bits.

    Why tiles? In a wide shot, a few people walking barely change a *whole-frame* hash. With 16
    tiles, the tile they walk through changes clearly.
    """
    h, w = gray.shape[:2]
    ys = np.linspace(0, h, grid + 1, dtype=int)  # tile borders
    xs = np.linspace(0, w, grid + 1, dtype=int)
    out = np.empty(grid * grid, dtype=np.uint64)
    k = 0
    for i in range(grid):
        for j in range(grid):
            tile = gray[ys[i]:ys[i + 1], xs[j]:xs[j + 1]]
            small = cv2.resize(tile, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
            low = cv2.dct(small)[:8, :8].flatten()
            bits = low > np.median(low[1:])  # low[0] is the average brightness, left out of the median
            # pack 64 booleans into 8 bytes, then read those as one big-endian 64-bit integer
            out[k] = np.packbits(bits).view(">u8")[0]
            k += 1
    return out


def gray_hist(gray: np.ndarray, bins: int = 32) -> np.ndarray:
    """Histogram of grey levels (how many pixels are dark, medium, bright...), normalised to sum to 1.
    Alternative change measure: ignores *where* things are, only the overall brightness distribution."""
    hist = cv2.calcHist([gray], [0], None, [bins], [0, 256]).ravel().astype(np.float32)
    return hist / max(hist.sum(), 1.0)


def phash_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Mean Hamming distance per tile (Hamming distance = number of bits that differ), 0..64.

    XOR gives 1 exactly where two bit patterns differ; we then count the 1s. We average over tiles
    rather than taking the max, because flat tiles (sky, grass) flip a few bits from compression noise
    alone and the max would pick that up. On the MEVA video: 0.2 s apart ~1 bit, a quiet 5 s gap
    ~2-3.5 bits, a busy 5 s gap ~6-8 bits.
    """
    x = np.bitwise_xor(np.asarray(a, np.uint64), np.asarray(b, np.uint64))
    bits = np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1)  # 1-bits per tile
    return float(bits.mean())


def hist_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Bhattacharyya distance between two histograms: 0 = identical, 1 = no overlap."""
    return float(cv2.compareHist(np.float32(a), np.float32(b), cv2.HISTCMP_BHATTACHARYYA))


def signature_distance(sig: dict, i: int, j: int, metric: str = "phash") -> float:
    """How different samples i and j look, using the cached fingerprints in `sig`."""
    if metric == "phash":
        return phash_distance(sig["phash"][i], sig["phash"][j])
    if metric == "hist":
        return hist_distance(sig["hist"][i], sig["hist"][j])
    raise ValueError(f"unknown metric {metric!r} (use 'phash' or 'hist')")


# ---------------------------------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------------------------------
def _dedup(cands: list[dict], sig: dict, metric: str, threshold: float) -> list[dict]:
    """Walk candidates in time order; if one is (nearly) the same image as the previously kept one in
    the same segment, keep only the one with the higher-priority reason."""
    kept: list[dict] = []
    for c in sorted(cands, key=lambda c: c["sample"]):
        if kept and kept[-1]["segment_id"] == c["segment_id"]:
            prev = kept[-1]
            same = c["sample"] == prev["sample"] or signature_distance(sig, c["sample"], prev["sample"], metric) < threshold
            if same:
                if REASON_PRIORITY[c["reason"]] < REASON_PRIORITY[prev["reason"]]:
                    kept[-1] = c
                continue
        kept.append(c)
    return kept


def select_keyframes(
    scores: pd.DataFrame,
    segments: pd.DataFrame,
    sig: dict,
    *,
    score_col: str,
    metric: str = "phash",
    change_threshold: float = 6.0,
    dedup_threshold: float = 2.5,
    min_gap_s: float = 3.0,
    max_gap_s: float = 30.0,
    heartbeat_s: float = 300.0,
) -> pd.DataFrame:
    """Choose keyframes from the per-sample table (`scores`), the segments and the fingerprints.

    scores:    from motion_gating.scan_video (columns frame_idx, timestamp_s, score_diff, score_mog2, ...)
    segments:  from motion_gating.find_segments
    score_col: which motion score decides the "peak" frame ("score_mog2" or "score_diff")
    Returns one row per keyframe: frame_idx, timestamp_s, timestamp_hms, segment_id, motion_score,
    reason, sample (row number in `scores`).
    """
    t = scores["timestamp_s"].to_numpy()
    s = scores[score_col].to_numpy()
    cands: list[dict] = []

    for seg in segments.itertuples(index=False):
        idx = np.flatnonzero((t >= seg.start_s) & (t < seg.end_s))  # sample rows inside this segment
        if idx.size == 0:
            continue
        sid = int(seg.segment_id)
        first = int(idx[0])
        cands.append({"sample": first, "segment_id": sid, "reason": "first"})
        last = first  # last kept keyframe; "change" is measured against it
        for i in idx[1:]:
            dt = t[i] - t[last]
            if dt < min_gap_s:
                continue
            if signature_distance(sig, int(i), last, metric) >= change_threshold:
                cands.append({"sample": int(i), "segment_id": sid, "reason": "change"})
                last = int(i)
            elif dt >= max_gap_s:
                cands.append({"sample": int(i), "segment_id": sid, "reason": "maxgap"})
                last = int(i)
        peak = int(idx[np.argmax(s[idx])])
        cands.append({"sample": peak, "segment_id": sid, "reason": "peak"})

    kept = _dedup(cands, sig, metric, dedup_threshold)

    # Heartbeat: walk through idle samples; whenever `heartbeat_s` has passed since the last keyframe
    # of any kind, keep one.
    if heartbeat_s and heartbeat_s > 0:
        active = np.zeros(len(t), dtype=bool)
        for seg in segments.itertuples(index=False):
            active |= (t >= seg.start_s) & (t < seg.end_s)
        kf_times = sorted(t[c["sample"]] for c in kept)
        last_t, ptr = -np.inf, 0
        for i in np.flatnonzero(~active):
            while ptr < len(kf_times) and kf_times[ptr] <= t[i]:  # advance to the latest keyframe before t[i]
                last_t = kf_times[ptr]
                ptr += 1
            if t[i] - last_t >= heartbeat_s:
                kept.append({"sample": int(i), "segment_id": IDLE_SEGMENT_ID, "reason": "heartbeat"})
                last_t = t[i]

    kept.sort(key=lambda c: c["sample"])
    rows = [{
        "frame_idx": int(scores["frame_idx"].iat[c["sample"]]),
        "timestamp_s": round(float(t[c["sample"]]), 3),
        "timestamp_hms": format_hms(t[c["sample"]]),
        "segment_id": c["segment_id"],
        "motion_score": float(s[c["sample"]]),
        "reason": c["reason"],
        "sample": c["sample"],
    } for c in kept]
    cols = ["frame_idx", "timestamp_s", "timestamp_hms", "segment_id", "motion_score", "reason", "sample"]
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------------------------------
def keyframe_filename(timestamp_s: float, segment_id: int) -> str:
    """'kf_00-12-34.500_seg007.jpg' ('_idle' for heartbeat frames)."""
    seg = "idle" if segment_id == IDLE_SEGMENT_ID else f"seg{segment_id:03d}"
    return f"kf_{format_hms(timestamp_s, sep='-')}_{seg}.jpg"


def export_keyframes(video_path: str | Path, keyframes: pd.DataFrame, out_dir: str | Path,
                     jpeg_quality: int = 90) -> pd.DataFrame:
    """Read the chosen frames at full resolution, save them as JPGs in out_dir/keyframes/, and write
    out_dir/keyframes.csv. Old kf_*.jpg files are removed first so folder and CSV always match."""
    out_dir = Path(out_dir)
    kf_dir = out_dir / "keyframes"
    kf_dir.mkdir(parents=True, exist_ok=True)
    for old in kf_dir.glob("kf_*.jpg"):
        old.unlink()
    kf = keyframes.copy()
    kf.insert(0, "filename", [keyframe_filename(r.timestamp_s, r.segment_id) for r in kf.itertuples()])
    names = dict(zip(kf["frame_idx"], kf["filename"]))
    written = 0
    for frame_idx, frame in read_frames(video_path, names.keys()):
        cv2.imwrite(str(kf_dir / names[frame_idx]), frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        written += 1
    if written != len(names):
        raise RuntimeError(f"wrote {written} of {len(names)} keyframes")
    kf.drop(columns="sample").to_csv(out_dir / "keyframes.csv", index=False)
    return kf


# ---------------------------------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------------------------------
def main() -> None:
    # Imported here, not at the top: motion_gating imports this file (for tile_phash), and importing
    # each other at the top would be a circular import.
    import motion_gating as mg

    ap = argparse.ArgumentParser(description="Stage 2: keyframe extraction (run motion_gating.py first).")
    ap.add_argument("video", nargs="?", default=str(mg.DEFAULT_VIDEO))
    a = ap.parse_args()

    video = Path(a.video)
    out = mg.output_dir(video)
    seg_csv = out / "segments.csv"
    if not seg_csv.exists():
        raise SystemExit(f"{seg_csv} not found, run: python motion_gating.py {video}")
    segments = pd.read_csv(seg_csv)
    method = segments["method"].iat[0] if len(segments) else "mog2"
    scan = mg.scan_video(video, **mg.SCAN_PARAMS, cache_dir=out / "cache")  # loads from cache
    kf = select_keyframes(scan["scores"], segments, scan["sig"], score_col=f"score_{method}", **KEYFRAME_PARAMS)
    kf = export_keyframes(video, kf, out)
    print(f"{len(kf)} keyframes {kf['reason'].value_counts().to_dict()} -> {out / 'keyframes'}")


if __name__ == "__main__":  # only runs when started as a script, not when imported
    main()
