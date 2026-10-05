import unittest

from diadem_feedback_rules import (
    asset_preference_score,
    asset_search_queries,
    contextual_resource_query,
    naturalise_reviewed_phrasing,
    response_instruction,
    reviewed_intent,
)


class FeedbackRulesTests(unittest.TestCase):
    def test_selling_boundary_is_explicit(self):
        text = "I'm not sure where the selling stops and the negotiation should begin"
        self.assertEqual(reviewed_intent(text), "selling_boundary")
        self.assertIn("asks for movement", response_instruction(text))
        self.assertTrue(asset_search_queries(text))

    def test_anxiety_prefers_variables_planner(self):
        text = "I feel anxious in a live negotiation conversation"
        self.assertEqual(reviewed_intent(text), "negotiation_anxiety")
        self.assertGreater(asset_preference_score(text, 40, "Variables Planner Low High Highest"), 15)
        self.assertGreater(asset_preference_score(text, 47, "Preparing The Negotiation Conversation"), 15)

    def test_difficult_behaviour_prefers_five_elements_and_rejects_disc(self):
        text = "What if they carry on being rude or bullying me?"
        self.assertEqual(reviewed_intent(text), "difficult_behaviour")
        self.assertGreater(asset_preference_score(text, 28, "Five Elements"), 20)
        self.assertLess(asset_preference_score(text, 15, "DISC personality styles"), 0)

    def test_meeting_domination_routes_to_difficult_behaviour(self):
        self.assertEqual(
            reviewed_intent("How do I stop them dominating the meeting and be more confident?"),
            "difficult_behaviour",
        )

    def test_cpi_uses_positions_not_advanced_styles(self):
        text = "We're going into a CPI and all the power sits with Tesco"
        self.assertEqual(reviewed_intent(text), "cpi_power")
        instruction = response_instruction(text)
        self.assertIn("Low/High/Highest", instruction)
        self.assertIn("Do not use Coal/Graphite/Diamond", instruction)

    def test_short_follow_up_retains_previous_user_topic(self):
        query = contextual_resource_query(
            "Internal",
            ["How do I stop them dominating the meeting and be more confident?"],
        )
        self.assertEqual(reviewed_intent(query), "difficult_behaviour")

    def test_long_standalone_query_does_not_inherit_old_topic(self):
        query = contextual_resource_query(
            "How should I structure a presentation for the board next week?",
            ["What if they are rude to me?"],
        )
        self.assertEqual(query, "How should I structure a presentation for the board next week?")

    def test_conditional_proposal_prefers_if_then_visual(self):
        text = "They've said no to my highest starting point. How do I make a counter proposal?"
        self.assertEqual(reviewed_intent(text), "conditional_proposal")
        self.assertGreater(asset_preference_score(text, 60, "Alternatives to If You Then I"), 20)

    def test_deadline_close_prefers_four_questions(self):
        text = "Can I agree now because I need this for my quarterly target?"
        self.assertEqual(reviewed_intent(text), "deadline_close")
        self.assertGreater(asset_preference_score(text, 77, "Before every negotiation answer 4 questions"), 20)

    def test_price_pressure_requires_card_and_toolkit(self):
        text = "The customer says we're too expensive and I have wiggle room"
        self.assertEqual(reviewed_intent(text), "price_issue")
        instruction = response_instruction(text)
        self.assertIn("CARD", instruction)
        self.assertIn("MASTER Toolkit", instruction)

    def test_pipeline_does_not_invent_unavailable_scotsman(self):
        text = "I have lots of unclosed deals in my pipeline"
        self.assertEqual(reviewed_intent(text), "pipeline_qualification")
        self.assertIn("Do not invent or name SCOTSMAN", response_instruction(text))

    def test_reviewed_language_is_naturalised(self):
        text = "What would need to be true for this deal? Use MASTER Variables."
        cleaned = naturalise_reviewed_phrasing(text)
        self.assertIn("What needs to happen for this deal?", cleaned)
        self.assertIn("MASTER Toolkit", cleaned)


if __name__ == "__main__":
    unittest.main()
