# User Testing Feedback Implementation

## Purpose

This change addresses the resource-selection and coaching issues recorded in `Master Negotiator AI testing (2).docx`. The existing answers were generally useful, but the application sometimes named or displayed the wrong Diadem tool, lost the topic during short follow-ups, or used advanced terminology where simpler course language was required.

## Implemented Behaviour

### Selling versus negotiating

- Defines selling as building value and earning agreement.
- Defines negotiating as trading after a proposal is on the table.
- Treats the other party asking for movement or changed terms as the transition signal.
- Prioritises the selling-versus-negotiating resource rather than a generic STRONG or MASTER slide.

### Negotiation anxiety

- Prioritises the Variables Planner and Low/High/Highest preparation.
- Retains Confident Mindset as supporting guidance.
- Treats a follow-up such as `the live conversation` as part of the preceding anxiety conversation.

### Rudeness, bullying and meeting domination

- Uses Five Elements as the primary tool.
- Uses Tactics Preparation/AIR as the supporting tool where present in retrieved content.
- Demotes unrelated DISC, Straightforwardness and Coal/Graphite/Diamond assets.
- Covers rudeness, bullying, belittling, undermining, interruption and meeting domination.

### CPI and perceived power imbalance

- Uses Low/High/Highest terminology.
- Prioritises the Variables Planner and Balanced Playing Field.
- Avoids Coal/Graphite/Diamond unless the user explicitly asks about those styles.

### Conversation persistence

Short or referential follow-ups use the last two user messages when selecting resources. Assistant answers are deliberately excluded so a previous AI mistake cannot reinforce itself.

This behaviour is active in:

- `POST /chat`
- `POST /chat/sse`
- MASTER template response endpoint
- MASTER template streaming endpoint

## Implementation Structure

`diadem_feedback_rules.py` contains deterministic, dependency-free rules:

- `reviewed_intent()` identifies only the scenarios found repeatedly in testing.
- `contextual_resource_query()` preserves intent for short follow-ups.
- `asset_search_queries()` defines targeted Pinecone searches.
- `asset_preference_score()` promotes the correct slides and demotes known mismatches.
- `response_instruction()` adds narrow answer requirements without replacing the main prompt.

`app.py` integrates these rules into retrieval, visible asset selection, fallback resources and response contracts.

## Local Verification

```powershell
.\.venv\Scripts\python.exe -m unittest test_diadem_feedback_rules.py
.\.venv\Scripts\python.exe -m py_compile app.py diadem_feedback_rules.py
git diff --check
```

## Production Acceptance Test

After deployment, use one new chat session per numbered scenario unless the step is explicitly a follow-up.

| Scenario | Expected answer | Expected visible resources |
| --- | --- | --- |
| Difference between selling and negotiating | Defines both and identifies a request for movement as the transition | Selling-versus-negotiating slide |
| Anxious about negotiating | Gives practical preparation using positions and variables | Variables Planner; Confident Mindset may also appear |
| `the live conversation` after anxiety | Continues the topic without repeating the first answer | Relevant live-conversation/preparation resource |
| Rude, belittling or bullying counterpart | Names Five Elements and gives a composed practical response | Five Elements and Tactics Preparation/AIR |
| Dominating an internal meeting, followed by `Internal` | Retains meeting-control intent and progresses the advice | Five Elements; no unrelated DISC slide |
| CPI with a powerful customer | Uses Low/High/Highest and multiple variables | Variables Planner and Balanced Playing Field |

For each response, confirm:

1. `assets` contains usable `image_url` values.
2. Bubble renders the returned assets rather than only mentioning them in text.
3. Suggested resource names match the displayed resources.
4. No duplicate or unrelated image is displayed.

## Operational Diagnostics

If the answer is correct but the image is wrong, inspect the API response before changing prompts:

- Correct `assets`, wrong Bubble output: fix the Bubble repeating group or image binding.
- Wrong `assets`, correct Pinecone match text: inspect page and source metadata.
- Missing `image_url`: re-index the affected slide with its image URL.
- Wrong source/page in Pinecone: correct the ingestion metadata and re-index that document.

Current index validation found that the Five Elements and CPI preparation slides have usable images. The selling-versus-negotiating search currently returns text-only `Selling.pdf` records, so its requested visual still requires the corresponding slide to be ingested with an `image_url`. The backend will continue to provide the correct written guidance until that asset is available.

## Deployment

This repository deploys through Render using `render.yaml`. Pushing the target branch updates GitHub; whether that automatically deploys depends on the Render service's connected branch and auto-deploy setting.

After Render reports a successful deploy:

1. Check the service health endpoint.
2. Run the production acceptance test above.
3. Inspect the raw JSON for any failed scenario before changing Bubble.
4. Confirm the Bubble app is calling the newly deployed service URL and not an older environment.

## Rollback

The feedback rules are isolated. To disable targeted asset lookups temporarily, set `ENABLE_ASSET_AUGMENT_RETRIEVAL=0`. Core semantic retrieval remains available, but the tested resource guarantees will no longer apply.
