# Diadem AI Production Evaluation

This repo now includes a lightweight evaluation loop for checking whether AI
responses are improving in a production-relevant way.

## Goal

The aim is not to force one fixed demo answer. The aim is to make the AI
consistently:

- retrieves the right Diadem methodology
- gives specific commercial guidance
- uses the right programme language
- avoids generic advice
- avoids exposing internal source details
- returns relevant supporting assets where available

## Files

- `diadem_production_eval_set.json`
  - A small set of representative MASTER Negotiator, STRONG Selling, Inspired Presenting, and document-review cases.

- `run_production_eval.py`
  - Calls the backend, captures responses, and applies transparent heuristic checks.

Generated reports are written as `eval_results_*.json` and are ignored by Git.

## Running Against Live Render

```powershell
.\.venv\Scripts\python.exe run_production_eval.py
```

This uses:

```text
https://diadem-searchv3.onrender.com
```

## Running Against Localhost

Start the local server first:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

Then run:

```powershell
.\.venv\Scripts\python.exe run_production_eval.py --base-url http://127.0.0.1:8000
```

## How Scoring Works

Each test case defines:

- `must_include`: terms or concepts that should usually appear
- `should_include`: useful supporting concepts
- `avoid`: phrases or behaviours that suggest weak output
- `asset_should_prefer`: preferred sources for visible supporting materials

The script produces:

- an overall score
- pass / watch / fail status
- missing expected concepts
- unwanted terms found
- visible asset sources
- a response preview

This is deliberately simple. It is a fast regression check, not a substitute for
human review.

## Production Use

Use this after prompt, retrieval, ingestion, or Bubble integration changes.

Recommended rhythm:

1. Run the eval before changes to get a baseline.
2. Make the change locally.
3. Run the eval locally or against a staging service.
4. Compare average score, failed cases, and missing concepts.
5. Push only if the overall trend improves and no important cases regress.

## Privacy Note

The evaluation set must not contain real confidential client documents. Use
synthetic or sanitised examples only.
