#!/usr/bin/env python3
"""
Download IIIF images of Diophantus Arithmetica, Paris. gr. 2485
from Gallica BnF.

Usage:
    python fetch_manuscript.py                  # download all 214 folios (thumbnails)
    python fetch_manuscript.py --full           # download full resolution
    python fetch_manuscript.py --folios 1-20    # download folios 1-20 only
    python fetch_manuscript.py --manifest       # download and save the IIIF manifest
"""

import argparse
import json
import time
import urllib.request
from pathlib import Path

ARK = "ark:/12148/btv1b10722263g"
BASE_IIIF = f"https://gallica.bnf.fr/iiif/{ARK}"
MANIFEST_URL = f"{BASE_IIIF}/manifest.json"
TOTAL_FOLIOS = 214
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

OUT_DIR = Path(__file__).parent / "images"


def image_url(folio: int, full: bool = False) -> str:
    size = "full" if full else "500,"
    return f"{BASE_IIIF}/f{folio}/full/{size}/0/native.jpg"


def download_file(url: str, dest: Path) -> bool:
    if dest.exists():
        print(f"  [skip] {dest.name} already exists")
        return True
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=30) as resp:
            dest.write_bytes(resp.read())
        print(f"  [ok]   {dest.name}")
        return True
    except Exception as exc:
        print(f"  [err]  {dest.name}: {exc}")
        return False


def parse_folio_range(spec: str, total: int) -> range:
    if "-" in spec:
        lo, hi = spec.split("-", 1)
        return range(int(lo), int(hi) + 1)
    return range(1, total + 1)


def download_manifest(out_dir: Path) -> dict:
    dest = out_dir / "manifest.json"
    if dest.exists():
        print(f"[manifest] Loading cached {dest}")
        return json.loads(dest.read_text())
    print(f"[manifest] Downloading from {MANIFEST_URL}")
    req = urllib.request.Request(MANIFEST_URL, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = resp.read()
    dest.write_bytes(data)
    print(f"[manifest] Saved to {dest}")
    return json.loads(data)


def main():
    parser = argparse.ArgumentParser(
        description="Download Diophantus Arithmetica manuscript images from Gallica BnF"
    )
    parser.add_argument(
        "--folios",
        default=f"1-{TOTAL_FOLIOS}",
        help=f"Folio range to download, e.g. '1-20' (default: all 1-{TOTAL_FOLIOS})",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Download full resolution (~5900×4200px) instead of thumbnails (500px wide)",
    )
    parser.add_argument(
        "--manifest",
        action="store_true",
        help="Download and save IIIF manifest JSON only",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay in seconds between requests (default: 0.5)",
    )
    parser.add_argument(
        "--out",
        default=str(OUT_DIR),
        help="Output directory (default: ./images/)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    size_label = "full" if args.full else "thumb"
    img_dir = out_dir / size_label
    img_dir.mkdir(parents=True, exist_ok=True)

    if args.manifest:
        download_manifest(out_dir)
        return

    folio_range = parse_folio_range(args.folios, TOTAL_FOLIOS)
    total = len(folio_range)
    print(
        f"Downloading {total} folios of Paris. gr. 2485 (Diophantus Arithmetica)\n"
        f"  Resolution: {'full (~5900px)' if args.full else 'thumbnail (500px wide)'}\n"
        f"  Output dir: {img_dir}\n"
        f"  Delay: {args.delay}s\n"
        f"  Gallica ARK: {ARK}\n"
    )

    ok = 0
    for i, folio in enumerate(folio_range, 1):
        url = image_url(folio, full=args.full)
        dest = img_dir / f"f{folio:03d}.jpg"
        if download_file(url, dest):
            ok += 1
        if i < total:
            time.sleep(args.delay)

    print(f"\nDone: {ok}/{total} folios downloaded to {img_dir}")
    if ok < total:
        print("Re-run the script to retry failed downloads (existing files are skipped).")


if __name__ == "__main__":
    main()
