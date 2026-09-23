"""Tests for canonical unit definitions and helpers."""

import unittest

from investigation import units


class CanonicalUnitsTests(unittest.TestCase):
    def test_stored_keys_map_to_canonical_fields(self):
        expected = {
            "typing_speed": "typing_speed_cpm",
            "dwell_mean": "dwell_mean_s",
            "flight_mean": "flight_mean_s",
            "correction_rate": "correction_rate",
            "rhythm_variability": "rhythm_variability_s",
            "pause_count": "pause_count",
        }
        for stored, canonical in expected.items():
            self.assertEqual(units.canonical_name(stored), canonical)

    def test_canonical_name_unknown_key_is_none(self):
        self.assertIsNone(units.canonical_name("nope"))

    def test_every_canonical_key_has_a_unit_and_label(self):
        for key in units.CANONICAL_FEATURE_KEYS:
            self.assertIn(key, units.CANONICAL_UNIT_NAMES)
            self.assertTrue(units.unit_for(key))
            self.assertTrue(units.label_for(key))

    def test_typing_speed_unit_is_cpm_not_wpm(self):
        self.assertEqual(units.unit_for("typing_speed_cpm"), "characters per minute")

    def test_dwell_unit_is_seconds(self):
        self.assertEqual(units.unit_for("dwell_mean_s"), "seconds")

    def test_canonical_units_returns_a_copy(self):
        snapshot = units.canonical_units()
        snapshot["typing_speed_cpm"] = "tampered"
        self.assertNotEqual(units.canonical_units()["typing_speed_cpm"], "tampered")

    def test_presentation_helpers(self):
        # Presentation-only conversions must not alter stored values.
        self.assertAlmostEqual(units.to_wpm(285.0), 57.0)
        self.assertAlmostEqual(units.to_milliseconds(0.12), 120.0)
        self.assertAlmostEqual(units.to_seconds(120.0), 0.12)
        self.assertIsNone(units.to_wpm(None))
        self.assertIsNone(units.to_milliseconds(None))


if __name__ == "__main__":
    unittest.main()
