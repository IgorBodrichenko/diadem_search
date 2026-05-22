import argparse
import csv
import json
import mimetypes
import os
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote

import requests


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
URL_RE = re.compile(r"https?://[^\s\"'<>]+|//[^\s\"'<>]+", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bulk upload slide images into Bubble Storage and create SlideAsset records.")
    p.add_argument("--archive", default="", help="Path to zip archive with slide images")
    p.add_argument("--dir", default="", help="Path to folder with slide images")
    p.add_argument("--bubble-base-url", required=True, help="Bubble app base URL, e.g. https://myapp.bubbleapps.io")
    p.add_argument("--bubble-api-token", required=True, help="Bubble API token")
    p.add_argument("--bubble-env", choices=["live", "test"], default="live", help="Use live or version-test API")
    p.add_argument("--data-type", default="SlideAsset", help="Bubble Data API type name")
    p.add_argument("--output-json", default="slide_image_urls.json", help="Output page->url mapping JSON")
    p.add_argument("--output-csv", default="slide_upload.csv", help="Output CSV with page,slide_id,image_url")
    p.add_argument("--start-page", type=int, default=1, help="Starting page number")
    p.add_argument("--skip-data-create", action="store_true", help="Skip creating Bubble Data API objects and only upload files + emit mapping")
    return p.parse_args()


def api_root(base_url: str, bubble_env: str) -> str:
    base = base_url.rstrip("/")
    if bubble_env == "test":
        return f"{base}/version-test/api/1.1"
    return f"{base}/api/1.1"


def fileupload_endpoints(base_url: str, bubble_env: str) -> List[str]:
    base = base_url.rstrip("/")
    if bubble_env == "test":
        return [
            f"{base}/version-test/api/1.1/fileupload",
            f"{base}/api/1.1/fileupload",
            f"{base}/version-test/fileupload",
            f"{base}/fileupload",
        ]
    return [
        f"{base}/api/1.1/fileupload",
        f"{base}/fileupload",
    ]


def _extract_url_from_obj(obj: object) -> str:
    if isinstance(obj, str):
        s = obj.strip().strip('"')
        if s.startswith("//"):
            return f"https:{s}"
        if s.startswith("http://") or s.startswith("https://"):
            return s
        m = URL_RE.search(s)
        if m:
            found = m.group(0)
            return f"https:{found}" if found.startswith("//") else found
        return ""

    if isinstance(obj, dict):
        # Prefer common keys first
        for key in ("url", "image_url", "file_url", "download_url", "response", "result", "data"):
            if key in obj:
                candidate = _extract_url_from_obj(obj.get(key))
                if candidate:
                    return candidate
        # Fallback: deep search all values
        for value in obj.values():
            candidate = _extract_url_from_obj(value)
            if candidate:
                return candidate
        return ""

    if isinstance(obj, list):
        for item in obj:
            candidate = _extract_url_from_obj(item)
            if candidate:
                return candidate
        return ""

    return ""


def _extract_upload_url(resp: requests.Response) -> str:
    # 1) JSON bodies
    try:
        data = resp.json()
        candidate = _extract_url_from_obj(data)
        if candidate:
            return candidate
    except Exception:
        pass

    # 2) Location header
    location = (resp.headers.get("location") or resp.headers.get("Location") or "").strip()
    if location:
        if location.startswith("//"):
            return f"https:{location}"
        if location.startswith("http://") or location.startswith("https://"):
            return location

    # 3) Raw text
    body = (resp.text or "").strip().strip('"')
    if body:
        if body.startswith("//"):
            return f"https:{body}"
        if body.startswith("http://") or body.startswith("https://"):
            return body
        m = URL_RE.search(body)
        if m:
            found = m.group(0)
            return f"https:{found}" if found.startswith("//") else found

    return ""


def collect_images(folder: Path) -> List[Path]:
    files = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS]
    return sorted(files, key=natural_sort_key)


def natural_sort_key(p: Path) -> Tuple:
    parts = re.split(r"(\d+)", p.name.lower())
    key: List[object] = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part)
    return tuple(key)


def maybe_extract_archive(archive_path: str) -> Tuple[Optional[tempfile.TemporaryDirectory], Path]:
    tmp = tempfile.TemporaryDirectory(prefix="slides_")
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(tmp.name)
    return tmp, Path(tmp.name)


def upload_file(
    session: requests.Session,
    base_url: str,
    bubble_env: str,
    token: str,
    file_path: Path,
) -> str:
    mime, _ = mimetypes.guess_type(str(file_path))
    mime = mime or "application/octet-stream"

    resp = None
    last_status = None
    for endpoint in fileupload_endpoints(base_url, bubble_env):
        for field_name in ("file", "contents", "upload"):
            with file_path.open("rb") as f:
                resp = session.post(
                    endpoint,
                    headers={"Authorization": f"Bearer {token}"},
                    files={field_name: (file_path.name, f, mime)},
                    timeout=60,
                )

            if resp.status_code == 404:
                last_status = 404
                continue

            resp.raise_for_status()
            url = _extract_upload_url(resp)
            if url:
                return url

    if resp is None or (resp.status_code == 404 and last_status == 404):
        raise requests.HTTPError(
            "Bubble file upload endpoint not found. Tried multiple /fileupload paths; got 404.",
            response=resp,
        )

    body_preview = (resp.text or "").strip().replace("\n", " ")[:300]
    raise ValueError(
        f"Upload returned empty URL for {file_path.name}. "
        f"status={resp.status_code}, content_type={resp.headers.get('content-type', '')}, "
        f"body={body_preview}"
    )


def create_slide_asset(
    session: requests.Session,
    root: str,
    token: str,
    data_type: str,
    page: int,
    slide_id: str,
    image_url: str,
    file_name: str,
) -> None:
    candidate_types: List[str] = [data_type]
    # Common mismatch: user passes plural but Bubble type is singular (or vice versa)
    if data_type.endswith("s") and len(data_type) > 1:
        candidate_types.append(data_type[:-1])
    else:
        candidate_types.append(f"{data_type}s")

    payload = {
        "page": page,
        "slide_id": slide_id,
        "image_url": image_url,
        "file_name": file_name,
    }

    last_resp: Optional[requests.Response] = None
    tried_endpoints: List[str] = []
    for t in candidate_types:
        endpoint = f"{root}/obj/{quote(t, safe='')}"
        tried_endpoints.append(endpoint)
        resp = session.post(
            endpoint,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60,
        )
        last_resp = resp

        if resp.status_code == 404:
            continue

        if resp.status_code in (401, 403):
            body_preview = (resp.text or "").strip().replace("\n", " ")[:400]
            raise requests.HTTPError(
                "Bubble Data API unauthorized while creating object. "
                "Check API token validity, Data API enabled, Privacy rules for this type, and exact --data-type name. "
                f"endpoint={endpoint}, status={resp.status_code}, body={body_preview}",
                response=resp,
            )

        resp.raise_for_status()
        return

    if last_resp is not None and last_resp.status_code == 404:
        body_preview = (last_resp.text or "").strip().replace("\n", " ")[:400]
        raise requests.HTTPError(
            "Bubble Data API type endpoint not found (404). "
            f"Tried: {', '.join(tried_endpoints)}. "
            "Verify Data type API name in Bubble and use it in --data-type.",
            response=last_resp,
        )

    if last_resp is not None:
        last_resp.raise_for_status()


def main() -> None:
    args = parse_args()
    if not args.archive and not args.dir:
        raise SystemExit("Provide either --archive or --dir")

    tmp_ctx: Optional[tempfile.TemporaryDirectory] = None
    image_root: Path

    if args.archive:
        if not os.path.exists(args.archive):
            raise SystemExit(f"Archive not found: {args.archive}")
        tmp_ctx, image_root = maybe_extract_archive(args.archive)
    else:
        image_root = Path(args.dir)
        if not image_root.exists():
            raise SystemExit(f"Directory not found: {args.dir}")

    try:
        images = collect_images(image_root)
        if not images:
            raise SystemExit("No image files found (.jpg/.jpeg/.png/.webp)")

        root = api_root(args.bubble_base_url, args.bubble_env)
        session = requests.Session()

        by_page: Dict[int, str] = {}
        rows: List[Dict[str, object]] = []

        page = args.start_page
        for img in images:
            slide_id = f"slide_{page:03d}"
            image_url = upload_file(
                session,
                args.bubble_base_url,
                args.bubble_env,
                args.bubble_api_token,
                img,
            )
            if not args.skip_data_create:
                create_slide_asset(
                    session,
                    root,
                    args.bubble_api_token,
                    args.data_type,
                    page,
                    slide_id,
                    image_url,
                    img.name,
                )

            by_page[page] = image_url
            rows.append({"page": page, "slide_id": slide_id, "image_url": image_url})
            print(f"[{page}] uploaded {img.name} -> {image_url}")
            page += 1

        with open(args.output_json, "w", encoding="utf-8") as jf:
            json.dump({"by_page": {str(k): v for k, v in by_page.items()}}, jf, ensure_ascii=False, indent=2)

        with open(args.output_csv, "w", newline="", encoding="utf-8") as cf:
            writer = csv.DictWriter(cf, fieldnames=["page", "slide_id", "image_url"])
            writer.writeheader()
            writer.writerows(rows)

        print(f"Done: {len(rows)} images uploaded")
        print(f"Mapping JSON: {args.output_json}")
        print(f"CSV: {args.output_csv}")
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()


if __name__ == "__main__":
    main()
