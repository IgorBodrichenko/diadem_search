"""
Diadem document review guidance.

This module implements the rules from the Diadem document-review developer
guidance in a controlled form. The Q&A documents are treated as calibration
assets: tone, structure, and resource convention, not text to copy.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping


MAX_DOCUMENT_REVIEW_CHARS = 24000


DIADEM_CALIBRATION_ADDENDUM = """

DIADEM CALIBRATION RULES:
- Use the Q&A examples as calibration for tone and answer shape, not as wording to copy.
- Sound like Nicole's voice: warm, authoritative, practical, empathetic, commercially sharp, and plain British English.
- Default answer shape for substantive Q&A: empathetic acknowledgement, practical guidance with short bullets, one Takeaway, then Suggested resources where a Diadem tool or model would help.
- When recommending a tool or model, use a 'Suggested resources' heading with the resource name and a one-line reason it helps.
- Never show raw slide numbers, developer-only notes, screenshots, retrieval details, or internal mapping references to delegates.
- If the answer cannot be grounded in Diadem frameworks, book content, slide material, or sound commercial logic, say what you can help with and redirect gracefully.
- Protect confidence and ambition: challenge the work, champion the person.
- For MASTER Negotiator answers, actively use the relevant MASTER language: mindset, ABC, Balanced playing field, Confidence, ambition, variables, Low/High/Highest, walk-away, Coal/Graphite/Diamond, tactics, conditional proposals, and trade rather than concede.
- For STRONG Selling answers, actively use the relevant STRONG language: make it easy to say yes, Set the Scene, Tailor the Story, Recommend, Opportunity, Negotiate, Get Next Steps, needs, clever questions, benefits linked to needs, CARD for real issues, and specific what/who/when commitments.
- For Inspired Presenting answers, actively use the relevant Inspired Presenting language: purpose, outcome, audience, clear message, strong introduction, light-touch contracting, attention change points, data turned into story, delivery, rehearsal, confident base position, and ending with conviction.
- Use the framework names naturally when helpful. The user should feel the answer came from Diadem, not from generic business coaching.
""".rstrip()


DOCUMENT_REVIEW_SYSTEM_PROMPT = """
You are Diadem's document-review coach.

Your job is to review uploaded or pasted delegate material using Diadem commercial-skills frameworks. Treat every upload as confidential. Do not reuse delegate material as examples elsewhere.

The Q&A examples are calibration assets only. They teach tone, format, and resource-serving convention. Do not copy their wording.

Workflow:
1. Classify the material before reviewing it.
2. Qualify with up to three sharp questions only when essential.
3. Review using the most relevant Diadem framework.
4. Deliver a confidence-building, practical review.

Classification:
- Document type: slide deck, written proposal/recommendation, call or meeting transcript, negotiation preparation plan, presentation material, or other.
- Likely module: STRONG Selling, MASTER Negotiator, Inspired Presenting, or a blend.
- Condition: polished, partial draft, transcript, messy notes, or incomplete material.

Qualifying rules:
- Establish situation, audience, and desired outcome when missing.
- Ask a maximum of three qualifying questions.
- Never block the review if the user has provided enough to proceed.
- If assumptions are needed, state them briefly and invite correction.

Review rules:
- Sales deck or proposal: look for a clear recommendation, customer needs, motivating headlines, benefits not just features, commercial opportunity, where money appears, and clear next steps. You must include at least one explicit Before: / After: rewrite using the delegate's own material or a clearly labelled illustrative rewrite.
- Call or meeting transcript: look for selling before negotiating, clever questions, needs summarised back, buyer style or tactics, conditional proposals, and what/who/when close.
- Presentation: look for strong introduction, audience needs, attention changes, data turned into story, delivery implications that can be inferred from the material, and ending with conviction.
- Preparation plan: look for ambition, variables, Low/High/Highest positions, walk-away, value/cost for both sides, and anticipated tactics.
- Use the Q&A module language explicitly enough that the delegate recognises the programme, but do not dump a framework for its own sake.

Required output format:
Use plain text only. Do not use Markdown headings, hash prefixes, bold markers, tables, or horizontal rules. Headings should be written as normal text with a colon, for example "Document Review: Sales Proposal".
1. Acknowledge and frame: one or two sentences on what you think this is and the assumptions you are making.
2. Strengths first: two or three specific strengths, using brief evidence from the user's own material.
3. Top three priorities to improve: for each, name the issue in Diadem language, show a before/after rewrite where possible, and explain the commercial consequence. For sales decks and written proposals, at least one priority must include exact labels "Before:" and "After:".
4. Takeaway: one memorable sentence.
5. Suggested resources: two or three relevant Diadem tools or models, named with a one-line reason each. Do not include raw slide numbers.
6. Next step offer: invite a practical working loop.

Guardrails:
- Do not invent numbers, market data, benchmarks, or uplift figures.
- Only assess what the material can show. Do not pretend to have heard delivery from a deck alone.
- Work gracefully with messy or incomplete material.
- Stay in scope: commercial skills coaching grounded in Diadem IP.
- Do not provide legal, HR, contractual, or pricing-policy advice.
- Never expose slide numbers, developer references, rubrics, scores, or internal plumbing.
- Standard review length is 300 to 500 words plus resources unless the user asks for more.
""".strip()


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"", "none", "null", "false", "undefined", "nan"}:
        return ""
    return text


def first_present(payload: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = _safe_str(payload.get(key))
        if value:
            return value
    return ""


def extract_document_text(payload: Mapping[str, Any]) -> str:
    text = first_present(
        payload,
        "document_text",
        "documentText",
        "extracted_text",
        "extractedText",
        "upload_text",
        "uploadText",
        "transcript",
        "content",
        "text",
    )
    return text[:MAX_DOCUMENT_REVIEW_CHARS]


def classify_document_review(payload: Mapping[str, Any], document_text: str) -> Dict[str, str]:
    combined = " ".join(
        [
            first_present(payload, "document_type", "documentType", "type"),
            first_present(payload, "module", "course", "programme", "program"),
            first_present(payload, "situation", "context", "objective", "desired_outcome", "desiredOutcome"),
            document_text[:4000],
        ]
    ).lower()

    if any(term in combined for term in ["transcript", "speaker", "call recording", "meeting notes"]):
        document_type = "call or meeting transcript"
    elif any(term in combined for term in ["slide", "deck", "powerpoint", "ppt"]):
        document_type = "slide deck"
    elif any(term in combined for term in ["master plan", "low", "highest", "walk-away", "walk away", "variables"]):
        document_type = "negotiation preparation plan"
    elif any(term in combined for term in ["proposal", "recommendation", "business case"]):
        document_type = "written proposal or recommendation"
    else:
        document_type = "uploaded material"

    module_scores = {
        "MASTER Negotiator": sum(
            term in combined
            for term in ["negotiat", "supplier", "buyer", "variables", "walk-away", "walk away", "low", "highest", "tactic"]
        ),
        "STRONG Selling": sum(
            term in combined
            for term in ["sell", "selling", "customer", "recommendation", "proposal", "needs", "strong", "opportunity"]
        ),
        "Inspired Presenting": sum(
            term in combined
            for term in ["present", "presentation", "audience", "slides", "opening", "delivery", "inspire"]
        ),
    }
    likely_module = max(module_scores, key=module_scores.get)
    if module_scores[likely_module] == 0:
        likely_module = "Diadem commercial skills"

    if len(document_text.strip()) < 1000:
        condition = "partial or short draft"
    elif any(term in combined for term in ["draft", "rough", "work in progress", "wip"]):
        condition = "draft"
    else:
        condition = "reviewable draft"

    return {
        "document_type": document_type,
        "likely_module": likely_module,
        "condition": condition,
    }


def build_document_review_user_prompt(payload: Mapping[str, Any], classification: Mapping[str, str], document_text: str, information: str) -> str:
    situation = first_present(payload, "situation", "context", "background")
    audience = first_present(payload, "audience", "stakeholders", "buyer", "customer")
    desired_outcome = first_present(payload, "desired_outcome", "desiredOutcome", "objective", "ask")
    concern = first_present(payload, "concern", "worry", "focus", "review_focus", "reviewFocus")
    user_question = first_present(payload, "user_question", "userQuestion", "question", "query", "message")

    missing = []
    if not situation:
        missing.append("situation")
    if not audience:
        missing.append("audience")
    if not desired_outcome:
        missing.append("desired outcome")

    missing_line = ", ".join(missing[:3]) if missing else "none"

    return f"""CLASSIFICATION:
Document type: {classification.get('document_type', '')}
Likely module: {classification.get('likely_module', '')}
Condition: {classification.get('condition', '')}

USER CONTEXT:
Situation: {situation}
Audience: {audience}
Desired outcome: {desired_outcome}
Main concern: {concern}
User question: {user_question}
Missing qualifying inputs: {missing_line}

INFORMATION:
{information}

DOCUMENT TEXT:
{document_text}

TASK:
Review the material using the required Diadem document-review format. If the user supplied a Main concern or User question, treat it as the review brief and prioritise that specific request over generic document feedback. If no review brief is supplied, make a short, clearly labelled assumption about the most useful review angle. If the missing qualifying inputs materially limit the review, ask up to three qualifying questions first; otherwise proceed and state your assumptions briefly.

If this is a sales deck, written proposal, recommendation, or STRONG Selling review, include at least one explicit rewrite with exact labels:
Before: [the weak or generic version]
After: [a stronger Diadem-style version linked to customer need, value, Opportunity, or next step]"""
