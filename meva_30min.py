#!/usr/bin/env python3
"""
meva_30min.py - grab a ~30-minute continuous surveillance video from the
public MEVA dataset (CC BY 4.0) and stitch it into one MP4 in video/.

Usage (defaults are fine):
    python meva_30min.py
Options:
    --minutes 30        length of the final video (5-min clips, so 30 -> 6 clips)
    --camera G330       force a specific camera ID (otherwise auto-picked)
    --out video/meva_30min.mp4
    --full-res          keep original resolution (default scales to 720p)
    --dry-run           only show which clips would be used

Needs: Python 3.8+. Uses ffmpeg if installed, otherwise auto-installs
the 'imageio-ffmpeg' pip package (which bundles an ffmpeg binary).
No AWS account or AWS CLI needed - the bucket is public.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict

BUCKET_URL = "https://mevadata-public-01.s3.amazonaws.com"
NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
# e.g. 2018-03-11.11-15-00.11-20-00.school.G330.r13.avi
NAME_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2})\.(\d{2})-(\d{2})-(\d{2})\.(\d{2})-(\d{2})-(\d{2})\.([^./]+)\.([^./]+)"
)
GAP_TOLERANCE_S = 15  # clips don't always start exactly on 5-min boundaries


def list_bucket(prefix="drops-123-r13/"):
    """Return [(key, size_bytes)] for every .avi under prefix."""
    out, token, page = [], None, 0
    while True:
        params = {"list-type": "2", "max-keys": "1000", "prefix": prefix}
        if token:
            params["continuation-token"] = token
        url = f"{BUCKET_URL}/?{urllib.parse.urlencode(params)}"
        with urllib.request.urlopen(url, timeout=60) as r:
            root = ET.fromstring(r.read())
        for c in root.findall("s3:Contents", NS):
            key = c.find("s3:Key", NS).text
            size = int(c.find("s3:Size", NS).text)
            if key.lower().endswith(".avi"):
                out.append((key, size))
        page += 1
        print(f"\r  listed {page} page(s), {len(out)} video files so far...", end="", flush=True)
        if root.find("s3:IsTruncated", NS).text != "true":
            break
        token = root.find("s3:NextContinuationToken", NS).text
    print()
    return out


def parse(objects):
    """Group clips by (date, site, camera); dedupe repeated releases of a clip."""
    groups = defaultdict(dict)
    unparsed = []
    for key, size in objects:
        if "uav" in key.lower():
            continue
        m = NAME_RE.search(os.path.basename(key))
        if not m:
            unparsed.append(key)
            continue
        date, h1, m1, s1, h2, m2, s2, site, cam = m.groups()
        start = int(h1) * 3600 + int(m1) * 60 + int(s1)
        end = int(h2) * 3600 + int(m2) * 60 + int(s2)
        if end <= start:
            continue
        clip = {"key": key, "size": size, "start": start, "end": end}
        g = groups[(date, site, cam)]
        if start not in g or size > g[start]["size"]:
            g[start] = clip
    return groups, unparsed


def find_runs(groups, n, camera=None):
    """Every run of n back-to-back clips from one camera, best first."""
    runs = []
    for (date, site, cam), clips in groups.items():
        if camera and cam.lower() != camera.lower():
            continue
        seq = sorted(clips.values(), key=lambda c: c["start"])
        for i in range(len(seq) - n + 1):
            window = seq[i : i + n]
            ok = all(
                abs(window[j + 1]["start"] - window[j]["end"]) <= GAP_TOLERANCE_S
                and (window[j]["end"] - window[j]["start"]) >= 240  # skip truncated clips
                for j in range(n - 1)
            ) and (window[-1]["end"] - window[-1]["start"]) >= 240
            if ok:
                # Bigger compressed files ~ more motion/detail -> more interesting footage
                runs.append((sum(c["size"] for c in window), date, site, cam, window))
    runs.sort(key=lambda r: r[0], reverse=True)
    return runs


def hhmmss(sec):
    return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def download(key, size, dest):
    if os.path.exists(dest) and os.path.getsize(dest) == size:
        print(f"  already have {os.path.basename(dest)}")
        return
    url = f"{BUCKET_URL}/{urllib.parse.quote(key)}"
    tmp = dest + ".part"
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            print(f"\r  {os.path.basename(dest)}  {done / 1e6:7.1f} / {size / 1e6:.1f} MB", end="", flush=True)
    print()
    os.replace(tmp, dest)


def get_ffmpeg():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
    except ImportError:
        print("ffmpeg not found - installing imageio-ffmpeg (bundled ffmpeg)...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "imageio-ffmpeg"])
        import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=int, default=30)
    ap.add_argument("--camera", default=None)
    video_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "video")
    ap.add_argument("--out", default=os.path.join(video_dir, "meva_30min.mp4"))
    ap.add_argument("--clips-dir", default=os.path.join(video_dir, "meva_clips"))
    ap.add_argument("--full-res", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    n = max(1, round(a.minutes / 5))

    print("1/4  Listing the public MEVA bucket (takes ~30 s)...")
    groups, unparsed = parse(list_bucket())
    if not groups:
        print("Could not recognise any MEVA clip names. Sample keys:")
        for k in unparsed[:10]:
            print("   ", k)
        sys.exit(1)

    print(f"2/4  Finding {n} back-to-back clips from one camera...")
    runs = find_runs(groups, n, a.camera)
    if not runs:
        sys.exit(f"No run of {n} consecutive clips found" + (f" for camera {a.camera}" if a.camera else "") + ".")
    _, date, site, cam, window = runs[0]
    print(f"     picked: {site} camera {cam}, {date} {hhmmss(window[0]['start'])} -> {hhmmss(window[-1]['end'])}"
          f"  ({sum(c['size'] for c in window) / 1e9:.2f} GB download)")
    if a.dry_run:
        for c in window:
            print("    ", c["key"])
        return

    print("3/4  Downloading clips...")
    os.makedirs(a.clips_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    paths = []
    for c in window:
        dest = os.path.join(a.clips_dir, os.path.basename(c["key"]))
        download(c["key"], c["size"], dest)
        paths.append(os.path.abspath(dest))

    print("4/4  Stitching with ffmpeg...")
    list_file = os.path.join(a.clips_dir, "list.txt")
    with open(list_file, "w", encoding="ascii") as f:
        for p in paths:
            f.write("file '" + p.replace("\\", "/").replace("'", "'\\''") + "'\n")
    cmd = [get_ffmpeg(), "-y", "-hide_banner", "-loglevel", "warning", "-stats",
           "-f", "concat", "-safe", "0", "-i", list_file,
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-an"]
    if not a.full_res:
        cmd += ["-vf", "scale=-2:720"]
    cmd.append(a.out)
    subprocess.check_call(cmd)

    with open(os.path.splitext(a.out)[0] + "_source.txt", "w") as f:
        f.write(f"MEVA (Kitware/IARPA, CC BY 4.0) - {site} camera {cam}, {date} "
                f"{hhmmss(window[0]['start'])}-{hhmmss(window[-1]['end'])}\n")
        f.write("Video time 00:00:00 = wall clock " + hhmmss(window[0]["start"]) + "\n")
        for c in window:
            f.write(c["key"] + "\n")
    print(f"\nDone -> {os.path.abspath(a.out)}")
    print(f"You can delete {a.clips_dir} afterwards to free space.")


if __name__ == "__main__":
    main()
