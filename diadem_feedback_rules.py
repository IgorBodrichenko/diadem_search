"""Deterministic routing rules derived from Diadem user-testing feedback."""

import re
from typing import List


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def reviewed_intent(text: str) -> str:
    """Return a narrow intent only where testing identified a repeatable failure."""
    q = _normalise(text)

    if any(term in q for term in ("selling and negotiating", "selling and negotiation", "selling vs", "selling stops", "negotiation should begin")):
        return "selling_boundary"
    if any(term in q for term in ("cpi", "power sits with", "power is with", "all the power")):
        return "cpi_power"
    if any(term in q for term in (
        "rude", "bully", "belittle", "undermine", "incompetent", "dominating",
        "talk over", "talking over", "cut me off", "steamroll", "ignoring them",
        "ignore them", "difficult behaviour", "difficult behavior",
    )):
        return "difficult_behaviour"
    if any(term in q for term in ("anxious", "anxiety", "nervous", "live conversation")) and any(
        term in q for term in ("negotiat", "meeting", "live conversation")
    ):
        return "negotiation_anxiety"
    return ""


def contextual_resource_query(query: str, previous_user_messages: List[str]) -> str:
    """Carry the topic into short/referential follow-ups without adding assistant guesses."""
    q = (query or "").strip()
    q_lower = q.lower()
    referential = len(q.split()) <= 4 or any(
        marker in f" {q_lower} "
        for marker in (" they ", " them ", " it ", " this ", " that ", " internal ", " live conversation ")
    )
    prior = [str(item or "").strip() for item in (previous_user_messages or []) if str(item or "").strip()]
    if not referential or not prior:
        return q
    return "\n".join(prior[-2:] + [q])


def asset_search_queries(text: str) -> List[str]:
    """Return targeted searches for resources explicitly requested in testing."""
    intent = reviewed_intent(text)
    if intent == "selling_boundary":
        return [
            "difference between selling and negotiating proposal on table pushback request movement switch from selling to negotiation slide 11",
        ]
    if intent == "negotiation_anxiety":
        return [
            "Master Negotiator Slides Preparing The Negotiation Conversation prioritised list sliding scale positions understand values costs page 47",
            "Preparing A Confident Mindset slide 14 live conversation",
        ]
    if intent == "difficult_behaviour":
        return [
            "MASTER Five Elements tool difficult behaviour tactics pages 28 29 30 31 32 33",
            "MASTER tactics preparation toolkit page 34 AIR respond do not react",
        ]
    if intent == "cpi_power":
        return [
            "Master Negotiator Slides Preparing The Negotiation Conversation prioritised list sliding scale positions understand values costs page 47",
            "MASTER ABC balanced playing field confidence page 6",
        ]
    return []


def response_instruction(text: str) -> str:
    """Return the minimum extra contract needed to correct reviewed answers."""
    intent = reviewed_intent(text)
    if intent == "selling_boundary":
        return (
            "Reviewed-case requirement: define selling and negotiating explicitly. Explain that negotiation begins "
            "after a proposal is on the table when the other party asks for movement or a change to the terms. "
            "Name the selling-versus-negotiating resource in Suggested resources."
        )
    if intent == "negotiation_anxiety":
        return (
            "Reviewed-case requirement: connect the practical advice to the Variables Planner and Low/High/Highest "
            "preparation. For a live-conversation follow-up, add a new in-room move instead of repeating the prior answer."
        )
    if intent == "difficult_behaviour":
        return (
            "Reviewed-case requirement: use the Five Elements tool as the primary named resource and connect it "
            "to tactics preparation/AIR where supported by INFORMATION. Do not substitute DISC, Straightforwardness, "
            "or Coal/Graphite/Diamond unless the user explicitly asks about that model. If this is a follow-up, progress "
            "the coaching rather than repeating earlier wording."
        )
    if intent == "cpi_power":
        return (
            "Reviewed-case requirement: describe positions as Low/High/Highest and use the Variables Planner. "
            "Do not use Coal/Graphite/Diamond unless the user explicitly asks about those negotiation styles."
        )
    return ""


def asset_preference_score(text: str, page: object, preview: str) -> float:
    """Strongly prefer the reviewed resource and reject known unrelated choices."""
    intent = reviewed_intent(text)
    content = _normalise(preview)
    try:
        page_number = int(float(page))
    except (TypeError, ValueError):
        page_number = 0

    if intent == "selling_boundary":
        if page_number == 11 or ("selling" in content and "negotiat" in content):
            return 18.0
    elif intent == "negotiation_anxiety":
        if page_number == 47 or ("prioritised list" in content and "sliding scale" in content):
            return 20.0
        if "variable" in content and any(term in content for term in ("low", "high", "highest", "planner")):
            return 18.0
        if page_number == 14 or "confident mindset" in content:
            return 10.0
    elif intent == "difficult_behaviour":
        if page_number in {28, 29, 30, 31, 32, 33} or "five elements" in content:
            return 22.0
        if page_number == 34 or "tactics preparation" in content or "air" in content:
            return 16.0
        if "disc" in content or "straightforwardness" in content or any(term in content for term in ("coal", "graphite", "diamond")):
            return -20.0
    elif intent == "cpi_power":
        if page_number == 47 or ("prioritised list" in content and "sliding scale" in content):
            return 24.0
        if "variable" in content and any(term in content for term in ("low", "high", "highest", "planner")):
            return 20.0
        if page_number == 6 or "balanced playing field" in content:
            return 12.0
        if any(term in content for term in ("coal", "graphite", "diamond")):
            return -20.0
    return 0.0
