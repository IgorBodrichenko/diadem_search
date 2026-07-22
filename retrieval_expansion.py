"""
Diadem retrieval expansion helpers.

These helpers improve Pinecone search by adding Diadem-specific retrieval
language to a user's question. The expanded text is only for retrieval; the
model still sees the user's original message separately.
"""

from __future__ import annotations

from typing import Iterable, List


def _has_any(text: str, terms: Iterable[str]) -> bool:
    t = text.lower()
    return any(term in t for term in terms)


def expand_diadem_retrieval_query(user_message: str, mode: str = "chat") -> str:
    """Expand a user message with Diadem-specific retrieval hints.

    The hints deliberately name the likely programme, framework, and practical
    concepts that should be retrieved. They are not shown to the user.
    """
    message = (user_message or "").strip()
    q = message.lower()
    additions: List[str] = []

    if mode.lower().startswith("document"):
        additions.append(
            "document review classify qualify review deliver strengths first top three priorities before after rewrite commercial why takeaway suggested resources delegate upload"
        )

    if mode.lower().startswith("master"):
        additions.append(
            "MASTER Negotiator master methodology negotiation template variables Low High Highest ambition walk-away trade concessions preparation confidence balanced playing field shopping list mindset"
        )

    if _has_any(q, ["supplier", "renewal", "incumbent", "long-term", "long term"]):
        additions.append(
            "MASTER Negotiator supplier renewal incumbent relationship leverage negotiation zone Graphite variables contract length payment terms service levels response times volume commitments implementation support trade variables not price only"
        )

    if _has_any(q, ["price", "increase", "discount", "expensive", "too high", "%", "margin"]):
        additions.append(
            "price pressure anchoring opening position discount concession value variables trade if you then I commercial control ambition price-only conversation challenge the number"
        )

    if _has_any(q, ["payment", "terms", "90 days", "60 days", "30 days"]):
        additions.append(
            "payment terms cash flow working capital Low High Highest ambition trade variable shorter terms"
        )

    if _has_any(q, ["scope", "extra request", "more work", "add", "additional", "free"]):
        additions.append(
            "scope creep variables trade exchange value timeline fee service levels commercial control"
        )

    if _has_any(q, ["procurement", "final offer", "deadline", "today", "take it or leave it"]):
        additions.append(
            "procurement pressure tactics deadline walk-away commercial control balanced playing field confidence difficult behaviour power"
        )

    if _has_any(q, ["tactic", "pressure", "power", "no budget", "do better", "tricky", "difficult"]):
        additions.append(
            "tactics difficult behaviour confidence balanced playing field self-control emotional intelligence get back to business"
        )

    if _has_any(q, ["prepare", "planning", "meeting tomorrow", "where to start", "not sure"]):
        additions.append(
            "preparation mindset ambition MASTER plan variables shopping list walk-away stakeholder support commercial objective confidence before every negotiation"
        )

    if _has_any(q, ["sell", "selling", "pitch", "recommendation", "proposal", "influence", "persuade"]):
        additions.append(
            "STRONG Selling storyboard Set the Scene Tailor the Story Recommend Opportunity Negotiate Get Next Steps needs opportunity value commercial benefit make it easy to say yes"
        )

    if _has_any(q, ["present", "presentation", "presenting", "audience", "slides", "deck", "delivery", "inspire"]):
        additions.append(
            "Inspired Presenting inner confidence audience needs strong introduction attention points data story clear message inspired delivery rehearse end with conviction"
        )

    if _has_any(q, ["review", "upload", "document", "transcript", "proposal", "deck", "feedback"]):
        additions.append(
            "document review strengths first top three priorities before after rewrite Takeaway Suggested resources classify qualify review deliver"
        )

    if _has_any(q, ["need", "needs", "discovery", "question", "questions", "customer wants"]):
        additions.append(
            "customer needs clever questions Set the Scene performance challenges pain points business needs ask listen emotional intelligence"
        )

    if _has_any(q, ["objection", "issue", "concern", "barrier", "won't", "will not", "no"]):
        additions.append(
            "CARD Clarify All Out Right Order Deal real issues objection handling close again commercial control"
        )

    if _has_any(q, ["benefit", "value", "commercial argument", "business case", "roi", "opportunity"]):
        additions.append(
            "Opportunity commercial benefit net impact robust commercial assumptions features benefits proof evidence so what"
        )

    if not additions and len(q.split()) <= 8:
        additions.append(
            "Diadem commercial coaching negotiation selling influencing variables needs confidence commercial control next step"
        )

    if not additions:
        return message

    return (message + "\n\nRETRIEVAL_HINTS:\n" + "\n".join(additions)).strip()
