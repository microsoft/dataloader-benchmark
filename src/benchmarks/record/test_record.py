import unittest
from typing import List

import numpy as np
import numpy.testing as nptest

from record import Record


class Test_Record(unittest.TestCase):
    def test_encode_decode(self):
        """Testing encode and decode of record. No segment involved."""
        record, dataset, features = build_simple_dataset()
        # encode dataset
        for i, item in enumerate(dataset):
            if i % 4 == 0:
                # mock start of a sequence
                record.encode_item(item, True)
            else:
                record.encode_item(item, False)
        record.close_recordfile()
        # decode dataset
        for i, item in enumerate(record.decode_record()):
            for feature in features:
                nptest.assert_equal(item[feature], dataset[i][feature], err_msg="", verbose=True)

    def test_idx4segment(self):
        """Having the record written (and various attributes setup), generate an index mapping for specific segment len."""
        record, dataset, features = build_simple_dataset()
        seq_len = 4
        # encode dataset
        for i, item in enumerate(dataset):
            if i % seq_len == 0:
                # mock start of a sequence
                record.encode_item(item, True)
            else:
                record.encode_item(item, False)
        record.close_recordfile()

        # segment len =2, sequence len =4, full features
        seg_len = 2
        idx4segment = record.idx4segment(segment_len=seg_len)
        self.assertEqual(len(idx4segment), 10)
        for i, (idx4segment, idx4record) in enumerate(zip(idx4segment, record.decode_record())):
            if i in [0, 1, 2, 4, 5, 6, 8]:
                self.assertTrue(idx4segment["is_seg_start"])
            else:
                self.assertFalse(idx4segment["is_seg_start"])
            for key in idx4record:
                if isinstance(idx4record[key], dict):
                    self.assertDictEqual(idx4record[key], idx4segment[key])
                else:
                    self.assertEqual(idx4record[key], idx4segment[key])

        # segment len =2, sequence len =4, full features
        seg_len = 4
        idx4segment = record.idx4segment(segment_len=seg_len)
        self.assertEqual(len(idx4segment), 8)
        for i, idx4record in enumerate(record.decode_record()):
            if i > 7:
                break
            if i in [0, 4]:
                self.assertTrue(idx4segment[i]["is_seg_start"])
            else:
                self.assertFalse(idx4segment[i]["is_seg_start"])
            for key in idx4record:
                if isinstance(idx4record[key], dict):
                    self.assertDictEqual(idx4record[key], idx4segment[i][key])
                else:
                    self.assertEqual(idx4record[key], idx4segment[i][key])

        # segment len =3, sequence len =4, full feature
        seg_len = 3
        idx4segment = record.idx4segment(segment_len=seg_len)
        self.assertEqual(len(idx4segment), 8)
        for i, idx4record in enumerate(record.decode_record()):
            if i > 7:
                break
            if i in [0, 1, 4, 5]:
                self.assertTrue(idx4segment[i]["is_seg_start"])
            else:
                self.assertFalse(idx4segment[i]["is_seg_start"])
            for key in idx4record:
                if isinstance(idx4record[key], dict):
                    self.assertDictEqual(idx4record[key], idx4segment[i][key])
                else:
                    self.assertEqual(idx4record[key], idx4segment[i][key])

    def test_idx4segment_brokenfeatures(self):
        """Test the idx4segment with some features from dataset missing."""
        # segment len = 3, sequence len =4, break two features
        record, dataset, features = build_broken_dataset([3, 4])
        seq_len = 4
        # encode dataset
        for i, item in enumerate(dataset):
            if i % seq_len == 0:
                # mock start of a sequence
                record.encode_item(item, True)
            else:
                record.encode_item(item, False)
        record.close_recordfile()

        seg_len = 3
        idx4segment = record.idx4segment(segment_len=seg_len)
        self.assertEqual(len(idx4segment), 6)
        ids = [item["item_idx"] for item in idx4segment]
        self.assertListEqual(ids, [0, 1, 2, 5, 6, 7])
        for i, item in enumerate(idx4segment):
            if item["item_idx"] in [0, 5]:
                self.assertTrue(item["is_seg_start"])
            else:
                self.assertFalse(item["is_seg_start"])

        seg_len = 3
        idx4segment = record.idx4segment(segment_len=seg_len, sub_features=["ceo"])
        self.assertEqual(len(idx4segment), 6)
        ids = [item["item_idx"] for item in idx4segment]
        self.assertListEqual(ids, [0, 1, 2, 5, 6, 7])
        for i, item in enumerate(idx4segment):
            if item["item_idx"] in [0, 5]:
                self.assertTrue(item["is_seg_start"])
            else:
                self.assertFalse(item["is_seg_start"])

                seg_len = 3

        idx4segment = record.idx4segment(segment_len=seg_len, sub_features=["l5", "l1"])
        self.assertEqual(len(idx4segment), 8)
        ids = [item["item_idx"] for item in idx4segment]
        self.assertListEqual(ids, [0, 1, 2, 3, 4, 5, 6, 7])
        for i, item in enumerate(idx4segment):
            if item["item_idx"] in [0, 1, 4, 5]:
                self.assertTrue(item["is_seg_start"])
            else:
                self.assertFalse(item["is_seg_start"])

        # segment len = 3, sequence len =4, break two features
        record, dataset, features = build_broken_dataset([3, 6])
        seq_len = 4
        # encode dataset
        for i, item in enumerate(dataset):
            if i % seq_len == 0:
                # mock start of a sequence
                record.encode_item(item, True)
            else:
                record.encode_item(item, False)
        record.close_recordfile()

        seg_len = 3
        idx4segment = record.idx4segment(segment_len=seg_len)
        self.assertEqual(len(idx4segment), 3)
        ids = [item["item_idx"] for item in idx4segment]
        self.assertListEqual(ids, [0, 1, 2])
        for i, item in enumerate(idx4segment):
            if item["item_idx"] in [0, 5]:
                self.assertTrue(item["is_seg_start"])
            else:
                self.assertFalse(item["is_seg_start"])

        # segment len = 3, sequence len =4, break two features
        record, dataset, features = build_broken_dataset([2, 6])
        seq_len = 4
        # encode dataset
        for i, item in enumerate(dataset):
            if i % seq_len == 0:
                # mock start of a sequence
                record.encode_item(item, True)
            else:
                record.encode_item(item, False)
        record.close_recordfile()

        seg_len = 3
        idx4segment = record.idx4segment(segment_len=seg_len)
        self.assertEqual(len(idx4segment), 0)


def build_simple_dataset():
    """Generate a fake dataset to test methods of Record

    Returns:
        _type_: _description_
    """
    rootdir = "/home/azureuser/data/data4record"
    seq_len = 10
    features = ["l1", "l5", "60", "63senior", "ceo"]
    dataset = [{} for _ in range(seq_len)]
    for i in range(seq_len):
        for feature in features:
            shape = (np.random.randint(1, 5), np.random.randint(2, 7))
            dtype = np.random.choice(["float32", "int", "bool"])
            dataset[i][feature] = np.random.rand(shape[0], shape[1]) if dtype == "float32" else np.ones(shape=shape)
    record = Record(rootdir, features)
    return record, dataset, features


def build_broken_dataset(feature_is_none_list: List[int]):
    """Generate a fake dataset to test methods of Record where some features does not exist

    Returns:
        _type_: _description_
    """
    rootdir = "/home/azureuser/data/data4record"
    seq_len = 10
    features = ["l1", "l5", "60", "63senior", "ceo"]
    dataset = [{} for _ in range(seq_len)]
    for i in range(seq_len):
        for feature in features:
            shape = (np.random.randint(1, 5), np.random.randint(2, 7))
            dtype = np.random.choice(["float32", "int", "bool"])
            if feature != "ceo" or (i not in feature_is_none_list):
                dataset[i][feature] = np.random.rand(shape[0], shape[1]) if dtype == "float32" else np.ones(shape=shape)
            else:
                dataset[i][feature] = np.array(None)
    record = Record(rootdir, features)
    return record, dataset, features


if __name__ == "__main__":
    np.random.seed(0)
    unittest.main()
