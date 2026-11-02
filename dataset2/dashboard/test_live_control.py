import unittest

from live_control import LiveControlError, validate_queue


class QueueValidationTests(unittest.TestCase):
    def test_computes_expected_color_and_transition_risk(self):
        rows = validate_queue([
            {"vial": 1, "recipe_id": 10, "t100_ml": 30, "t200_ml": 0, "t400_ml": 0},
            {"vial": 2, "recipe_id": 20, "t100_ml": 0, "t200_ml": 30, "t400_ml": 0},
        ], [[240, 120, 30], [30, 90, 220], [245, 245, 245]])
        self.assertEqual(rows[0]["expected_hex"], "#f5f5f5")
        self.assertEqual(rows[1]["expected_hex"], "#f0781e")
        self.assertEqual(rows[1]["transition_risk"], "HIGH")

    def test_rejects_fractional_ids_and_bad_totals(self):
        with self.assertRaisesRegex(LiveControlError, "whole numbers"):
            validate_queue([{"vial": 1.5, "recipe_id": 1, "t100_ml": 10}])
        with self.assertRaisesRegex(LiveControlError, "does not equal"):
            validate_queue([{
                "vial": 1, "recipe_id": 1, "t100_ml": 10,
                "t200_ml": 10, "t400_ml": 10, "total_ml": 25,
            }])

    def test_rejects_duplicate_or_empty_queue(self):
        with self.assertRaisesRegex(LiveControlError, "at least one"):
            validate_queue([])
        with self.assertRaisesRegex(LiveControlError, "duplicated"):
            validate_queue([
                {"vial": 1, "recipe_id": 1, "t100_ml": 10},
                {"vial": 1, "recipe_id": 2, "t200_ml": 10},
            ])


if __name__ == "__main__":
    unittest.main()
