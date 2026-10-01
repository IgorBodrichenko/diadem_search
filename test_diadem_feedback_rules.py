import unittest

from diadem_feedback_rules import (
    asset_preference_score,
    asset_search_queries,
    contextual_resource_query,
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


if __name__ == "__main__":
    unittest.main()
