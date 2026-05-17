#!/usr/bin/env python3
"""
Download Byzantine Astronomical Manuscript Corpus — batch IIIF downloader
==========================================================================
Downloads IIIF manifests and (optionally) images for all manuscripts in
manuscripts/config.json.

Usage:
  python download_astro_corpus.py                     # manifests only, all priorities
  python download_astro_corpus.py --images            # manifests + full images
  python download_astro_corpus.py --priority 1        # level-1 only
  python download_astro_corpus.py --priority 1 2 --images  # levels 1+2 with images
  python download_astro_corpus.py --signum Vat.gr.1291     # single manuscript
  python download_astro_corpus.py --images --width 1200    # smaller images (faster)
  python download_astro_corpus.py --table-pages-only       # images only for known table folios

After running, manifests are at:
  manuscripts/<dir>/manifest_meta.json
  manuscripts/<dir>/index.html

Images (if --images) at:
  manuscripts/<dir>/images/<####_label>.jpg
"""

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────

ASTRO_DIR    = Path(__file__).parent
CONFIG_PATH  = ASTRO_DIR / "manuscripts" / "config.json"
MS_BASE      = ASTRO_DIR / "manuscripts"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; ByzAstroOTR/1.0; "
        "+https://github.com/VesterlundCoder/tdi-atlas)"
    )
}
RETRY_DELAYS = [2, 5, 15]

# ── HTTP helpers ───────────────────────────────────────────────────────────────

def fetch_json(url: str, timeout: int = 30) -> dict:
    r = requests.get(url, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.json()


def download_image(url: str, dest: Path, retries: int = 3) -> bool:
    if dest.exists():
        return True
    for delay in RETRY_DELAYS[:retries]:
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


# ── Manifest parsing ───────────────────────────────────────────────────────────

def extract_canvases(manifest: dict) -> list:
    canvases = []
    for seq in manifest.get("sequences", []):
        for canvas in seq.get("canvases", []):
            label     = canvas.get("label", "unlabeled")
            canvas_id = canvas.get("@id", "")
            m         = re.search(r"p(\d+)$", canvas_id)
            page_num  = int(m.group(1)) if m else 0

            service_id = None
            for img_anno in canvas.get("images", []):
                svc = img_anno.get("resource", {}).get("service", {})
                service_id = svc.get("@id")
                if service_id:
                    break

            thumb_url = None
            if "thumbnail" in canvas:
                thumb_url = canvas["thumbnail"].get("@id")

            canvases.append({
                "page_num":   page_num,
                "label":      label,
                "canvas_id":  canvas_id,
                "service_id": service_id,
                "thumb_url":  thumb_url,
            })
    return sorted(canvases, key=lambda c: c["page_num"])


def build_image_url(service_id: str, width: int) -> str:
    size = f"{width}," if width > 0 else "full"
    return f"{service_id.rstrip('/')}/full/{size}/0/default.jpg"


def safe_filename(label: str, page_num: int) -> str:
    clean = re.sub(r"[^\w\-.]", "_", label)
    return f"{page_num:04d}_{clean}.jpg"


# ── Table folio detection ──────────────────────────────────────────────────────

TABLE_LABEL_PATTERNS = [
    re.compile(r"\b(ΑΝΑΦΟΡΑΙ|ΩΡΩΝ|ΧΡΟΝΩΝ|anaforai|anaphorai)\b", re.IGNORECASE),
    re.compile(r"\b(tabul|tabula|tableau)\b", re.IGNORECASE),
    re.compile(r"\bklima\b", re.IGNORECASE),
]

ZODIAC_LABEL_PATTERNS = re.compile(
    r"\b(ΚΡΙΟΥ|ΤΑΥΡΟΥ|ΔΙΔΥΜΩΝ|ΚΑΡΚΙΝΟΥ|ΛΕΟΝΤΟΣ|ΠΑΡΘΕΝΟΥ|"
    r"ΖΥΓΟΥ|ΣΚΟΡΠΙΟΥ|ΤΟΞΟΤΟΥ|ΑΙΓΟΚΕΡΩΤΟΣ|ΥΔΡΟΧΟΟΥ|ΙΧΘΥΩΝ|"
    r"aries|taurus|gemini|cancer|leo|virgo|libra|scorpio|sagittarius)\b",
    re.IGNORECASE,
)

# Known folio ranges from config "known_table_folios" field
def is_known_table_folio(canvas_label: str, ms_config: dict) -> bool:
    """Check if a folio label falls in the known table range for this manuscript."""
    ktf = ms_config.get("klima5_folio_range")
    if not ktf or ktf in (None, "unknown"):
        return False
    if isinstance(ktf, dict) and ktf.get("start") == "unknown":
        return False
    # Simple label match for now
    start = str(ktf.get("start", ""))
    end   = str(ktf.get("end",   ""))
    label = canvas_label.lower().replace(" ", "")
    return label in (start.lower(), end.lower())


# ── Per-manuscript download ────────────────────────────────────────────────────

def process_manuscript(ms: dict, ms_base: Path, args) -> dict:
    signum  = ms["signum"]
    ms_dir  = ms_base / ms["dir"]
    img_dir = ms_dir / "images"
    ms_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "signum":       signum,
        "dir":          ms["dir"],
        "manifest_ok":  False,
        "n_canvases":   0,
        "n_table_pages": 0,
        "images_downloaded": 0,
        "images_failed": 0,
        "error":        None,
    }

    # ── Manifest ──────────────────────────────────────────────────────────────
    meta_path = ms_dir / "manifest_meta.json"
    if meta_path.exists() and not args.force:
        print(f"  [{signum}] manifest already cached, skipping fetch")
        with open(meta_path) as fh:
            meta = json.load(fh)
        result["manifest_ok"] = True
    else:
        try:
            print(f"  [{signum}] fetching manifest …")
            manifest = fetch_json(ms["iiif_url"])
            canvases = extract_canvases(manifest)

            shelfmark = manifest.get("label", signum)
            meta = {
                "signum":       signum,
                "shelfmark":    shelfmark,
                "manifest_url": ms["iiif_url"],
                "date_range":   ms.get("date_range", ""),
                "script":       ms.get("script", ""),
                "content_summary": ms.get("content_summary", ""),
                "table_types":  ms.get("table_types", []),
                "model_confidence": ms.get("model_confidence", "unknown"),
                "iiif_metadata": manifest.get("metadata", []),
                "n_canvases":   len(canvases),
                "canvases":     [],
            }
            for c in canvases:
                fname    = safe_filename(c["label"], c["page_num"])
                img_url  = build_image_url(c["service_id"], args.width) if c["service_id"] else None
                is_table = is_known_table_folio(c["label"], ms)
                meta["canvases"].append({
                    "page_num":      c["page_num"],
                    "label":         c["label"],
                    "filename":      fname,
                    "image_url":     img_url,
                    "thumb_url":     c["thumb_url"],
                    "is_known_table_folio": is_table,
                })

            meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
            result["manifest_ok"] = True
            result["n_canvases"]  = len(canvases)
            print(f"    → {len(canvases)} pages, saved to {meta_path.name}")

        except Exception as exc:
            result["error"] = str(exc)
            print(f"  [{signum}] ERROR: {exc}", file=sys.stderr)
            return result

    result["n_canvases"] = meta.get("n_canvases", len(meta.get("canvases", [])))

    # ── Write HTML index ───────────────────────────────────────────────────────
    _write_html_index(meta, ms_dir, ms)

    # ── Image download ─────────────────────────────────────────────────────────
    if not args.images:
        return result

    img_dir.mkdir(exist_ok=True)
    canvases_to_dl = meta["canvases"]
    if args.table_pages_only:
        canvases_to_dl = [c for c in canvases_to_dl if c.get("is_known_table_folio")]
        print(f"    → {len(canvases_to_dl)} known table pages to download")

    tasks = [
        (c["image_url"], img_dir / c["filename"])
        for c in canvases_to_dl
        if c.get("image_url")
    ]

    ok = fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for url, dest in tasks:
            time.sleep(args.delay / args.workers)
            futures[pool.submit(download_image, url, dest)] = dest

        with tqdm(total=len(futures), desc=f"  {signum}", unit="img") as bar:
            for fut in as_completed(futures):
                if fut.result():
                    ok += 1
                else:
                    fail += 1
                    tqdm.write(f"    FAILED: {futures[fut].name}")
                bar.update(1)

    result["images_downloaded"] = ok
    result["images_failed"]     = fail
    return result


# ── HTML index ─────────────────────────────────────────────────────────────────

def _write_html_index(meta: dict, out_dir: Path, ms_cfg: dict) -> None:
    signum = meta["signum"]
    conf   = meta.get("model_confidence", "?")
    conf_note = ms_cfg.get("confidence_note", "")

    conf_colour = {"high": "#4caf50", "medium": "#ff9800",
                   "medium-low": "#ff6600", "low": "#f44336",
                   "none": "#888"}.get(conf, "#aaa")

    meta_rows = ""
    for entry in meta.get("iiif_metadata", []):
        meta_rows += (
            f'<tr><td class="key">{entry.get("label","")}</td>'
            f'<td>{entry.get("value","")}</td></tr>\n'
        )

    thumbs_html = ""
    for c in meta.get("canvases", []):
        thumb  = c.get("thumb_url") or c.get("image_url") or ""
        label  = c["label"]
        border = " style=\"border:2px solid #c9a84c\"" if c.get("is_known_table_folio") else ""
        thumbs_html += (
            f'<div class="thumb"{border}>'
            f'<a href="images/{c["filename"]}" target="_blank">'
            f'<img src="{thumb}" loading="lazy" alt="{label}" '
            f'onerror="this.style.display=\'none\'"></a>'
            f'<span>{label}</span></div>\n'
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{signum} — Byzantine Astro OTR</title>
<style>
  body {{font-family:Georgia,serif;max-width:1400px;margin:0 auto;padding:1rem;
         background:#1a1a2e;color:#e0e0e0}}
  h1 {{color:#c9a84c;border-bottom:1px solid #c9a84c;padding-bottom:.5rem}}
  h2 {{color:#c9a84c;margin-top:2rem}}
  table {{border-collapse:collapse;width:100%;margin-bottom:1.5rem}}
  td {{padding:.4rem .8rem;border:1px solid #333}}
  td.key {{font-weight:bold;color:#c9a84c;width:30%}}
  .conf {{display:inline-block;padding:.2rem .6rem;border-radius:4px;
          background:{conf_colour};color:#000;font-weight:bold}}
  .grid {{display:grid;grid-template-columns:repeat(auto-fill,minmax(120px,1fr));gap:10px}}
  .thumb {{text-align:center}}
  .thumb img {{width:100%;border-radius:3px;border:1px solid #444}}
  .thumb span {{font-size:.7rem;color:#aaa;display:block;margin-top:3px}}
  a {{color:#c9a84c}}
</style>
</head>
<body>
<h1>{signum}</h1>
<p><span class="conf">{conf.upper()} MODEL CONFIDENCE</span> — {conf_note}</p>
<a href="{meta['manifest_url']}" target="_blank">IIIF Manifest</a>
<h2>Manuscript Information</h2>
<table>{meta_rows}
  <tr><td class="key">Table types</td><td>{', '.join(meta.get('table_types',[]))}</td></tr>
  <tr><td class="key">Script</td><td>{meta.get('script','')}</td></tr>
  <tr><td class="key">Content</td><td>{meta.get('content_summary','')}</td></tr>
  <tr><td class="key">Total pages</td><td>{meta.get('n_canvases',0)}</td></tr>
</table>
<h2>Pages <small style="color:#888">(gold border = known table folio)</small></h2>
<div class="grid">{thumbs_html}</div>
</body></html>"""

    (out_dir / "index.html").write_text(html, encoding="utf-8")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--priority",  type=int, nargs="+", default=[1, 2, 3],
                    help="Priority levels to download (1=must-have, 2=cross-hand, 3=broader)")
    ap.add_argument("--signum",    type=str, default=None,
                    help="Download single manuscript by signum (e.g. 'Vat.gr.1291')")
    ap.add_argument("--images",    action="store_true",
                    help="Also download page images (slow; each ms may be 100-500 MB)")
    ap.add_argument("--table-pages-only", action="store_true",
                    help="Only download images for known table folios (much faster)")
    ap.add_argument("--width",     type=int, default=2000,
                    help="Image width in pixels (default 2000)")
    ap.add_argument("--workers",   type=int, default=2,
                    help="Parallel download threads per manuscript (keep ≤3 for Vatican)")
    ap.add_argument("--delay",     type=float, default=0.6,
                    help="Seconds between requests per worker")
    ap.add_argument("--force",     action="store_true",
                    help="Re-fetch manifests even if already cached")
    args = ap.parse_args()

    if not CONFIG_PATH.exists():
        print(f"ERROR: config not found at {CONFIG_PATH}", file=sys.stderr)
        sys.exit(1)

    with open(CONFIG_PATH) as fh:
        cfg = json.load(fh)

    manuscripts = cfg["manuscripts"]

    # Filter
    if args.signum:
        manuscripts = [m for m in manuscripts if m["signum"] == args.signum]
        if not manuscripts:
            print(f"ERROR: signum '{args.signum}' not found in config", file=sys.stderr)
            sys.exit(1)
    else:
        manuscripts = [m for m in manuscripts if m["priority"] in args.priority]

    print(f"Processing {len(manuscripts)} manuscript(s) …")
    print(f"  Images: {'YES, ' + ('table pages only' if args.table_pages_only else 'ALL pages') if args.images else 'NO (manifest only)'}")
    print()

    results = []
    for ms in manuscripts:
        print(f"── {ms['signum']} (priority {ms['priority']}, confidence: {ms.get('model_confidence','?')}) ──")
        r = process_manuscript(ms, MS_BASE, args)
        results.append(r)
        time.sleep(0.5)   # polite pause between manuscripts
        print()

    # Summary
    print("── Summary ─────────────────────────────────────────────────────────")
    ok_count   = sum(1 for r in results if r["manifest_ok"])
    fail_count = sum(1 for r in results if not r["manifest_ok"])
    print(f"  Manifests: {ok_count} ok, {fail_count} failed")
    if args.images:
        total_dl = sum(r["images_downloaded"] for r in results)
        total_fail = sum(r["images_failed"] for r in results)
        print(f"  Images: {total_dl} downloaded, {total_fail} failed")

    print(f"\n  Per manuscript:")
    for r in results:
        status = "✓" if r["manifest_ok"] else "✗"
        img_note = f", {r['images_downloaded']} images" if args.images else ""
        err_note = f"  ERROR: {r['error']}" if r["error"] else ""
        print(f"    {status} {r['signum']:<20} {r['n_canvases']:>4} pages{img_note}{err_note}")

    # Save index
    index = {
        "downloaded_manifests": [r for r in results if r["manifest_ok"]],
        "failed": [r for r in results if not r["manifest_ok"]],
    }
    idx_path = MS_BASE / "download_index.json"
    idx_path.write_text(json.dumps(index, indent=2, ensure_ascii=False))
    print(f"\n  Index saved → {idx_path}")
    print("\nNext: python scan_tables.py")


if __name__ == "__main__":
    main()
