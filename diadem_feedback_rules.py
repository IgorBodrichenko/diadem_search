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
        "quarterly target", "quarter end", "end of the quarter", "agree to where we've got to",
        "agree to where we have got to", "close the deal now", "lengthy negotiation",
    )):
        return "deadline_close"
    if any(term in q for term in (
        "counter proposal", "counterproposal", "highest starting point", "they've said no",
        "they have said no", "without dropping", "without lowering", "if you, then i",
    )):
        return "conditional_proposal"
    if any(term in q for term in (
        "too expensive", "it's price", "it is price", "wiggle room", "move on price",
        "competitor pricing", "like for like", "price comparison", "issue with the price",
        "losing the deal", "price pressure",
    )):
        return "price_issue"
    if any(term in q for term in (
        "unclosed deals", "pipeline", "gone quiet", "got back to us", "get any momentum",
        "serious about doing a deal", "serious about the deal",
    )):
        return "pipeline_qualification"
    if any(term in q for term in (
        "rude", "bully", "belittle", "undermine", "incompetent", "dominating",
        "talk over", "talking over", "cut me off", "steamroll", "ignoring them",
        "ignore them", "difficult behaviour", "difficult behavior", "intimidating",
        "afraid to push back", "threat to go elsewhere", "go elsewhere", "time pressure",
        "tyme prsure", "deadline threat",
    )):
        return "difficult_behaviour"
    if any(term in q for term in ("anxious", "anxiety", "nervous", "live conversation")) and any(
        term in q for term in ("negotiat", "meeting", "live conversation")
    ):
        return "negotiation_anxiety"
    if any(term in q for term in (
        "better deal", "salary review", "salary negotiation", "variable planning",
        "ambition and positions", "highest", "high and low",
    )):
        return "master_toolkit"
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
            "Master Negotiator Slides Confidently Respond To Tactics Factics Or Tactical Behaviour Five Elements page 28",
            "MASTER tactics preparation toolkit page 34 AIR respond do not react",
        ]
    if intent == "cpi_power":
        return [
            "Master Negotiator Slides Preparing The Negotiation Conversation prioritised list sliding scale positions understand values costs page 47",
            "MASTER ABC balanced playing field confidence page 6",
        ]
    if intent == "deadline_close":
        return [
            "Master Negotiator Slides Before Every Negotiation You Need To Answer 4 Questions page 77",
            "Master Negotiator Slides Preparing The Negotiation Conversation prioritised list sliding scale positions page 47",
        ]
    if intent == "conditional_proposal":
        return [
            "Master Negotiator Slides Articulating Your Proposal alternatives to If You Then I page 60",
            "Master Negotiator Slides Preparing The Negotiation Conversation page 47",
        ]
    if intent == "price_issue":
        return [
            "CARD Clarify All Out Right Order Deal price objection",
            "Master Negotiator Slides Preparing The Negotiation Conversation page 47",
        ]
    if intent == "pipeline_qualification":
        return [
            "STRONG Get Next Steps specific what who when pipeline qualification",
        ]
    if intent == "master_toolkit":
        return [
            "Master Negotiator Slides Preparing The Negotiation Conversation prioritised list sliding scale positions page 47",
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
    if intent == "deadline_close":
        return (
            "Reviewed-case requirement: use the Four Questions to test whether closing now protects the deal, and "
            "name the MASTER Toolkit as the practical preparation resource. Do not recommend the same resource twice."
        )
    if intent == "conditional_proposal":
        return (
            "Reviewed-case requirement: keep the user's Highest position intact, introduce another variable, and use "
            "a conditional proposal. Name the MASTER Toolkit and the If you, then I proposal-language tool in Suggested resources."
        )
    if intent == "price_issue":
        return (
            "Reviewed-case requirement: use CARD to test whether price pressure is real before discussing movement. "
            "Name CARD and the MASTER Toolkit in Suggested resources, not MASTER Variables. If STRONG terminology is "
            "used, explain it in plain language because the user may not have studied STRONG."
        )
    if intent == "pipeline_qualification":
        return (
            "Reviewed-case requirement: qualify genuine momentum and secure a specific what, who and when. Avoid "
            "Coal/Graphite/Diamond as a generic pipeline-ranking device. Do not invent or name SCOTSMAN unless it is "
            "present in INFORMATION. Name the MASTER Toolkit where negotiation preparation is recommended."
        )
    if intent == "master_toolkit":
        return (
            "Reviewed-case requirement: name the MASTER Toolkit as the practical resource for ambition, variables, "
            "Low/High/Highest positions and preparation. Do not call the resource MASTER Variables."
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
    elif intent == "deadline_close":
        if page_number in {4, 77} or "answer 4 questions" in content:
            return 26.0
        if page_number == 47 or "sliding scale of positions" in content:
            return 16.0
    elif intent == "conditional_proposal":
        if page_number == 60 or "alternatives to" in content and "then i" in content:
            return 26.0
        if page_number == 47:
            return 14.0
    elif intent == "price_issue":
        if "card" in content or all(term in content for term in ("clarify", "right order", "deal")):
            return 28.0
        if page_number == 47 or "sliding scale of positions" in content:
            return 14.0
        if any(term in content for term in ("coal", "graphite", "diamond")):
            return -16.0
    elif intent == "pipeline_qualification":
        if "get next steps" in content or all(term in content for term in ("what", "who", "when")):
            return 20.0
        if any(term in content for term in ("coal", "graphite", "diamond")):
            return -18.0
    elif intent == "master_toolkit":
        if page_number == 47 or "sliding scale of positions" in content:
            return 24.0
    return 0.0


def naturalise_reviewed_phrasing(text: str) -> str:
    """Remove a repeated phrase that user testing identified as synthetic."""
    value = text or ""
    def _natural_question(match: re.Match) -> str:
        replacement = "what needs to happen for "
        if match.group(0)[:1].isupper():
            replacement = replacement[:1].upper() + replacement[1:]
        return replacement

    value = re.sub(r"(?i)what would need to be true for\s+", _natural_question, value)
    value = re.sub(r"(?i)\bMASTER Variables\b", "MASTER Toolkit", value)
    value = re.sub(
        r"(?i)\bMASTER Negotiation\s*[\-—:]\s*(Ambition and Variable Planning|Variable Planning|Ambition and Positions)\b",
        lambda match: f"MASTER Toolkit - {match.group(1)}",
        value,
    )
    return value
