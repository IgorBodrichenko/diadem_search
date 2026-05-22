import argparse
import json
from typing import Any, Dict, List, Tuple

import requests


def _post_json(base_url: str, path: str, payload: Dict[str, Any], timeout: int = 60) -> Tuple[bool, str]:
    url = f"{base_url.rstrip('/')}{path}"
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return False, f"{path}: request/json error: {e}"

    assets = data.get("assets")
    if isinstance(assets, list):
        return True, f"{path}: PASS (assets list, len={len(assets)})"
    return False, f"{path}: FAIL (assets missing or not list)"


def _post_sse(base_url: str, path: str, payload: Dict[str, Any], timeout: int = 90) -> Tuple[bool, str]:
    url = f"{base_url.rstrip('/')}{path}"
    try:
        resp = requests.post(url, json=payload, stream=True, timeout=timeout)
        resp.raise_for_status()
    except Exception as e:
        return False, f"{path}: request error: {e}"

    saw_assets_event = False
    done_assets_ok = False

    event = ""
    data_lines: List[str] = []

    def flush_event(ev: str, payload_lines: List[str]) -> None:
        nonlocal saw_assets_event, done_assets_ok
        if not ev:
            return
        raw = "\n".join(payload_lines).strip()
        if not raw:
            return
        try:
            obj = json.loads(raw)
        except Exception:
            return

        if ev == "assets":
            assets = obj.get("assets")
            if isinstance(assets, list):
                saw_assets_event = True

        if ev == "done":
            assets = obj.get("assets")
            if isinstance(assets, list):
                done_assets_ok = True

    for line in resp.iter_lines(decode_unicode=True):
        if line is None:
            continue

        if line == "":
            flush_event(event, data_lines)
            if event == "done":
                break
            event = ""
            data_lines = []
            continue

        if line.startswith("event:"):
            event = line[len("event:") :].strip()
            continue

        if line.startswith("data:"):
            data_lines.append(line[len("data:") :].strip())

    if done_assets_ok:
        return True, f"{path}: PASS (done includes assets list, assets_event={saw_assets_event})"
    return False, f"{path}: FAIL (done has no assets list)"


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test assets contract across chat/master endpoints")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--query", default="How should I prepare my negotiation variables?")
    parser.add_argument("--template-id", default="smoke-template")
    args = parser.parse_args()

    payload_chat = {"query": args.query}
    payload_master = {"query": args.query, "template_id": args.template_id}

    checks = [
        (_post_json, "/chat", payload_chat),
        (_post_sse, "/chat/sse", payload_chat),
        (_post_json, "/master/template", payload_master),
        (_post_sse, "/master/template/sse", payload_master),
    ]

    failed = 0
    for fn, path, payload in checks:
        ok, msg = fn(args.base_url, path, payload)
        print(msg)
        if not ok:
            failed += 1

    if failed:
        print(f"RESULT: FAIL ({failed} checks failed)")
        return 1

    print("RESULT: PASS (all checks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
