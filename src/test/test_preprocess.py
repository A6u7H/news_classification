import os
import unittest
import sys

sys.path.append(os.path.join(os.getcwd(), "src"))

from preprocessor import BCCPreprocessor


config_path = "./configs/train_config.ini"


class TestDataMaker(unittest.TestCase):
    def setUp(self) -> None:
        self.data_maker = BCCPreprocessor(config_path)

    def test_get_train_data(self):
        train_target_columns = ["ArticleId", "Text", "Category"]
        train_df = self.data_maker.load_train_data()
        self.assertListEqual(
            list(train_df.columns),
            train_target_columns
        )
        self.assertEqual(
            len(train_df),
            1490
        )

    def test_get_test_data(self):
        test_target_columns = ["ArticleId", "Text"]
        test_df = self.data_maker.load_test_data()
        self.assertListEqual(
            list(test_df.columns),
            test_target_columns
        )
        self.assertEqual(
            len(test_df),
            735
        )

    def test_save_splitted_data(self):
        bbc_train_data = self.data_maker.load_train_data()

        bbc_train, bbc_val = self.data_maker.split_data(bbc_train_data)
        self.assertEqual(
            len(bbc_train),
            1192
        )


if __name__ == "__main__":
    unittest.main()
