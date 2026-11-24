import copy
import unittest
from unittest.mock import patch

import rag


class TranslationReviewTest(unittest.TestCase):
    def setUp(self):
        self.translations = rag.doc_map(rag.PREPARED / "ru")
        self.workbook = next(
            name for name in self.translations if name.endswith("/Assistant abonnement.xlsx")
        )

    def workbook_chunks(self, translations):
        original_doc_map = rag.doc_map
        with patch.object(
            rag,
            "doc_map",
            side_effect=lambda folder: (
                translations if folder.name == "ru" else original_doc_map(folder)
            ),
        ):
            return [chunk for chunk in rag.prepared_corpus() if chunk["source"] == self.workbook]

    def test_current_reviewed_text_is_included_and_edited_text_is_excluded(self):
        translations = copy.deepcopy(self.translations)
        document = translations[self.workbook]
        document["fragments"]["5"] += "\nИзменение после сверки."
        original_ids = {
            piece["id"] for chunk in self.workbook_chunks(self.translations)
            for piece in chunk["fragments"]
        }
        modified_ids = {
            piece["id"] for chunk in self.workbook_chunks(translations)
            for piece in chunk["fragments"]
        }
        self.assertIn(5, original_ids)
        self.assertEqual(modified_ids, original_ids - {5})

    def test_unreviewed_spreadsheet_title_is_excluded_from_search_context(self):
        translations = copy.deepcopy(self.translations)
        document = translations[self.workbook]
        title = document["fragments"]["1"].partition(": ")[2]
        del document["reviewed_sha256"]["1"]
        admitted = self.workbook_chunks(self.translations)
        filtered = self.workbook_chunks(translations)
        self.assertTrue(any(title in chunk["context"] for chunk in admitted))
        self.assertTrue(filtered)
        self.assertFalse(any(title in chunk["context"] for chunk in filtered))


if __name__ == "__main__":
    unittest.main()
