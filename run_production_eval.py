"""
Run a lightweight production evaluation for Diadem AI endpoints.

This script is intentionally simple: it calls the configured backend, captures
responses, applies transparent heuristic checks, and writes a JSON report.
It does not require model keys when pointed at the live Render service.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import requests


DEFAULT_BASE_URL = "https://diadem-searchv3.onrender.com"
DEFAULT_EVAL_SET = "diadem_production_eval_set.json"


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _normalise(text: Any) -> str:
    return str(text or "").lower()


def _contains(text: str, term: str) -> bool:
    return term.lower() in text


def _coverage(text: str, terms: Iterable[str]) -> Dict[str, Any]:
    terms = list(terms or [])
    hits = [term for term in terms if _contains(text, term)]
    misses = [term for term in terms if term not in hits]
    return {
        "hits": hits,
        "misses": misses,
        "count": len(hits),
        "total": len(terms),
        "ratio": (len(hits) / len(terms)) if terms else 1.0,
    }


def _avoid_coverage(text: str, terms: Iterable[str]) -> Dict[str, Any]:
    terms = list(terms or [])
    hits: List[str] = []
    for term in terms:
        pattern = re.compile(r"\b" + re.escape(term.lower()) + r"\b")
        for match in pattern.finditer(text):
            prefix = text[max(0, match.start() - 18):match.start()]
            if re.search(r"(don't|do not|never|avoid|stop|without|rather than|instead of)\s+$", prefix):
                continue
            hits.append(term)
            break

    misses = [term for term in terms if term not in hits]
    return {
        "hits": hits,
        "misses": misses,
        "count": len(hits),
        "total": len(terms),
        "ratio": (len(hits) / len(terms)) if terms else 0.0,
    }


def _response_text(body: Mapping[str, Any]) -> str:
    return str(body.get("answer") or body.get("text") or "")


def _asset_sources(body: Mapping[str, Any]) -> List[str]:
    out: List[str] = []
    for asset in body.get("assets") or []:
        if isinstance(asset, Mapping):
            source = asset.get("source") or asset.get("file_name") or asset.get("file") or ""
            page = asset.get("page")
            out.append(f"{source} p{page}" if page else str(source))
    return out


def _asset_score(case: Mapping[str, Any], body: Mapping[str, Any]) -> Dict[str, Any]:
    preferred = list(case.get("asset_should_prefer") or [])
    if not preferred:
        return {"preferred": preferred, "sources": _asset_sources(body), "hits": [], "score": 1.0}

    sources = _asset_sources(body)
    source_text = " ".join(sources).lower()
    hits = [term for term in preferred if term.lower() in source_text]
    return {
        "preferred": preferred,
        "sources": sources,
        "hits": hits,
        "score": (len(hits) / len(preferred)) if preferred else 1.0,
    }


def _score_case(case: Mapping[str, Any], body: Mapping[str, Any]) -> Dict[str, Any]:
    text = _normalise(_response_text(body))
    must = _coverage(text, case.get("must_include") or [])
    should = _coverage(text, case.get("should_include") or [])
    avoid = _avoid_coverage(text, case.get("avoid") or [])
    assets = _asset_score(case, body)

    # Weighted, transparent heuristic. This is not pretending to be a human
    # judge; it is a quick tripwire for obvious regressions.
    score = 0.0
    score += must["ratio"] * 50.0
    score += should["ratio"] * 30.0
    score += assets["score"] * 10.0
    score += 10.0 if avoid["count"] == 0 else 0.0

    status = "pass"
    if must["ratio"] < 0.75:
        status = "fail"
    elif score < 75.0:
        status = "watch"

    return {
        "score": round(score, 1),
        "status": status,
        "must_include": must,
        "should_include": should,
        "avoid_terms_found": avoid["hits"],
        "assets": assets,
        "response_preview": _response_text(body)[:500],
    }


def _call_case(base_url: str, case: Mapping[str, Any], timeout: int) -> Dict[str, Any]:
    endpoint = str(case.get("endpoint") or "").strip()
    if not endpoint.startswith("/"):
        endpoint = "/" + endpoint
    url = base_url.rstrip("/") + endpoint
    payload = case.get("payload") or {}

    started = time.time()
    resp = requests.post(url, json=payload, timeout=timeout)
    elapsed_ms = int((time.time() - started) * 1000)

    try:
        body = resp.json()
    except Exception:
        body = {"raw_text": resp.text}

    return {
        "http_status": resp.status_code,
        "elapsed_ms": elapsed_ms,
        "body": body,
    }


def run_eval(base_url: str, eval_path: Path, timeout: int, limit: int = 0) -> Dict[str, Any]:
    eval_set = _load_json(eval_path)
    cases = list(eval_set.get("cases") or [])
    if limit > 0:
        cases = cases[:limit]

    results = []
    for idx, case in enumerate(cases, start=1):
        case_id = case.get("id") or f"case_{idx}"
        print(f"[{idx}/{len(cases)}] {case_id} -> {case.get('endpoint')}")
        try:
            called = _call_case(base_url, case, timeout)
            if called["http_status"] >= 400:
                scored = {
                    "score": 0.0,
                    "status": "error",
                    "error": f"HTTP {called['http_status']}",
                    "response_preview": str(called["body"])[:500],
                }
            else:
                scored = _score_case(case, called["body"])
        except Exception as exc:
            called = {"http_status": 0, "elapsed_ms": 0, "body": {}}
            scored = {
                "score": 0.0,
                "status": "error",
                "error": str(exc),
                "response_preview": "",
            }

        results.append(
            {
                "id": case_id,
                "module": case.get("module"),
                "endpoint": case.get("endpoint"),
                "http_status": called["http_status"],
                "elapsed_ms": called["elapsed_ms"],
                "score": scored["score"],
                "status": scored["status"],
                "details": scored,
            }
        )

    avg = round(sum(r["score"] for r in results) / len(results), 1) if results else 0.0
    status_counts: Dict[str, int] = {}
    for result in results:
        status_counts[result["status"]] = status_counts.get(result["status"], 0) + 1

    return {
        "run_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "base_url": base_url,
        "eval_set": str(eval_path),
        "case_count": len(results),
        "average_score": avg,
        "status_counts": status_counts,
        "results": results,
    }


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Run Diadem production AI evaluation.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Backend base URL.")
    parser.add_argument("--eval-set", default=DEFAULT_EVAL_SET, help="Path to evaluation JSON.")
    parser.add_argument("--timeout", type=int, default=75, help="Request timeout in seconds.")
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N cases.")
    parser.add_argument("--output", default="", help="Optional output JSON path.")
    args = parser.parse_args(argv)

    eval_path = Path(args.eval_set)
    report = run_eval(args.base_url, eval_path, args.timeout, args.limit)

    output = Path(args.output) if args.output else Path(
        "eval_results_" + dt.datetime.now().strftime("%Y%m%d_%H%M%S") + ".json"
    )
    with output.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("")
    print(f"Average score: {report['average_score']}")
    print(f"Status counts: {report['status_counts']}")
    print(f"Report written: {output}")

    if report["status_counts"].get("error") or report["status_counts"].get("fail"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
