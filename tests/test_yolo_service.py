import unittest

from services.label_mapping import get_display_label


class YOLOServiceLabelTests(unittest.TestCase):
    def test_swaps_bakso_and_sop_ayam(self):
        self.assertEqual(get_display_label("bakso"), "sop ayam")
        self.assertEqual(get_display_label("sop ayam"), "bakso")

    def test_swap_is_case_and_whitespace_insensitive(self):
        self.assertEqual(get_display_label(" Bakso "), "sop ayam")
        self.assertEqual(get_display_label("SOP   AYAM"), "bakso")

    def test_preserves_other_model_labels(self):
        self.assertEqual(get_display_label("Rawon"), "Rawon")


if __name__ == "__main__":
    unittest.main()
