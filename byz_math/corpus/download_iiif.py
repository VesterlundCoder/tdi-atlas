"""
IIIF Manuscript Downloader — DigiVatLib & compatible IIIF v2 servers.

Usage:
    python -m byz_math.corpus.download_iiif \
        --manifest https://digi.vatlib.it/iiif/MSS_Vat.gr.1777/manifest.json \
        --out byz_math/data/vat_gr_1777 \
        --width 2000 \
        --workers 3
"""

import argparse
import json
import time
import re
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; ByzMathResearch/1.0; "
        "+https://github.com/VesterlundCoder/tdi-atlas)"
    )
}
RETRY_DELAYS = [2, 5, 15]


# ---------------------------------------------------------------------------
# Manifest parsing
# ---------------------------------------------------------------------------

def fetch_manifest(url: str) -> dict:
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()


def extract_canvases(manifest: dict) -> list[dict]:
    """Return list of {label, image_url, thumb_url, canvas_id} dicts."""
    canvases = []
    sequences = manifest.get("sequences", [])
    for seq in sequences:
        for canvas in seq.get("canvases", []):
            label = canvas.get("label", "unlabeled")
            canvas_id = canvas.get("@id", "")
            # Extract page number from canvas id (e.g. p0007 → 7)
            m = re.search(r"p(\d+)$", canvas_id)
            page_num = int(m.group(1)) if m else 0

            thumb_url = None
            if "thumbnail" in canvas:
                thumb_url = canvas["thumbnail"].get("@id")

            service_id = None
            for img_anno in canvas.get("images", []):
                resource = img_anno.get("resource", {})
                service = resource.get("service", {})
                service_id = service.get("@id")
                if service_id:
                    break

            canvases.append({
                "page_num": page_num,
                "label": label,
                "canvas_id": canvas_id,
                "service_id": service_id,
                "thumb_url": thumb_url,
            })
    return sorted(canvases, key=lambda c: c["page_num"])


def build_image_url(service_id: str, width: int) -> str:
    size = f"{width}," if width > 0 else "full"
    return f"{service_id.rstrip('/')}/full/{size}/0/default.jpg"


# ---------------------------------------------------------------------------
# Downloading
# ---------------------------------------------------------------------------

def safe_filename(label: str, page_num: int) -> str:
    clean = re.sub(r"[^\w\-.]", "_", label)
    return f"{page_num:04d}_{clean}.jpg"


def download_image(url: str, dest: Path, retries: int = 3) -> bool:
    if dest.exists():
        return True
    for attempt, delay in enumerate(RETRY_DELAYS[:retries]):
        try:
            r = requests.get(url, headers=HEADERS, timeout=60, stream=True)
            if r.status_code == 200:
                dest.write_bytes(r.content)
                return True
            if r.status_code in (429, 503):
                time.sleep(delay)
        except requests.RequestException:
            time.sleep(delay)
    return False


def download_manifest(
    manifest_url: str,
    out_dir: Path,
    width: int = 2000,
    workers: int = 3,
    delay: float = 0.4,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching manifest: {manifest_url}")
    manifest = fetch_manifest(manifest_url)

    shelfmark = manifest.get("label", "unknown")
    meta = {
        "shelfmark": shelfmark,
        "manifest_url": manifest_url,
        "metadata": manifest.get("metadata", []),
        "canvases": [],
    }

    canvases = extract_canvases(manifest)
    print(f"  {shelfmark}: {len(canvases)} pages")

    tasks = []
    for c in canvases:
        fname = safe_filename(c["label"], c["page_num"])
        dest = out_dir / fname
        img_url = build_image_url(c["service_id"], width) if c["service_id"] else None
        meta["canvases"].append({
            "page_num": c["page_num"],
            "label": c["label"],
            "filename": fname,
            "image_url": img_url,
            "thumb_url": c["thumb_url"],
        })
        if img_url:
            tasks.append((img_url, dest))

    # Save metadata before download so it's always available
    meta_path = out_dir / "manifest_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"  Metadata saved → {meta_path}")

    ok = fail = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for url, dest in tasks:
            time.sleep(delay / workers)
            futures[pool.submit(download_image, url, dest)] = dest

        with tqdm(total=len(futures), unit="img", desc=shelfmark) as bar:
            for fut in as_completed(futures):
                if fut.result():
                    ok += 1
                else:
                    fail += 1
                    tqdm.write(f"  FAILED: {futures[fut].name}")
                bar.update(1)

    print(f"  Done: {ok} downloaded, {fail} failed → {out_dir}/")
    return meta


# ---------------------------------------------------------------------------
# HTML index for phone/browser access
# ---------------------------------------------------------------------------

def write_html_index(meta: dict, out_dir: Path) -> None:
    shelfmark = meta["shelfmark"]

    meta_rows = ""
    for m in meta.get("metadata", []):
        meta_rows += (
            f'<tr><td class="key">{m.get("label","")}</td>'
            f'<td>{m.get("value","")}</td></tr>\n'
        )

    thumbs_html = ""
    for c in meta["canvases"]:
        local_img = c["filename"]
        thumb = c["thumb_url"] or c["image_url"] or ""
        label = c["label"]
        thumbs_html += (
            f'<div class="thumb">'
            f'<a href="{local_img}" target="_blank">'
            f'<img src="{thumb}" loading="lazy" alt="{label}" onerror="this.style.display=\'none\'">'
            f'</a>'
            f'<span>{label}</span>'
            f'</div>\n'
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{shelfmark} — Byzantine Manuscripts</title>
<style>
  body {{ font-family: Georgia, serif; max-width: 1200px; margin: 0 auto; padding: 1rem; background: #1a1a2e; color: #e0e0e0; }}
  h1 {{ color: #c9a84c; border-bottom: 1px solid #c9a84c; padding-bottom: .5rem; }}
  table {{ border-collapse: collapse; margin-bottom: 2rem; width: 100%; }}
  td {{ padding: .4rem .8rem; border: 1px solid #333; }}
  td.key {{ font-weight: bold; color: #c9a84c; width: 30%; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(140px,1fr)); gap: 12px; }}
  .thumb {{ text-align: center; }}
  .thumb img {{ width: 100%; border-radius: 4px; border: 1px solid #444; }}
  .thumb span {{ font-size: .75rem; color: #aaa; display: block; margin-top: 4px; }}
  a {{ color: #c9a84c; }}
  .manifest-link {{ margin-bottom: 1rem; display: block; }}
</style>
</head>
<body>
<h1>{shelfmark}</h1>
<a class="manifest-link" href="{meta['manifest_url']}" target="_blank">IIIF Manifest JSON</a>
<table>{meta_rows}</table>
<h2>{len(meta['canvases'])} Pages</h2>
<div class="grid">
{thumbs_html}
</div>
</body>
</html>"""

    index_path = out_dir / "index.html"
    index_path.write_text(html, encoding="utf-8")
    print(f"  HTML index → {index_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Download IIIF manuscript images")
    parser.add_argument(
        "--manifest",
        default="https://digi.vatlib.it/iiif/MSS_Vat.gr.1777/manifest.json",
        help="IIIF Presentation manifest URL",
    )
    parser.add_argument(
        "--out",
        default="byz_math/data/vat_gr_1777",
        help="Output directory",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=2000,
        help="Image width in pixels (0 = full resolution)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Parallel download threads (keep ≤3 for Vatican)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Seconds between requests per worker",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    meta = download_manifest(
        manifest_url=args.manifest,
        out_dir=out_dir,
        width=args.width,
        workers=args.workers,
        delay=args.delay,
    )
    write_html_index(meta, out_dir)


if __name__ == "__main__":
    main()
