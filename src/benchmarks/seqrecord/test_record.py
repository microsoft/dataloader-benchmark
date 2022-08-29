import unittest
from typing import Dict, List

import numpy as np
import numpy.testing as nptest
from seqrecord import SeqRecord

# todo: test read segment carefully
# todo: test read_all_segment under various conditions


def concate_list(file2segment_item: Dict[str, list]):
    res = []
    for key in sorted(file2segment_item):
        res = res + file2segment_item[key]
    return res


class Test_SeqRecord(unittest.TestCase):
    def test_encode_decode(self):
        """Testing encode and decode of record.

        No segment involved.
        """
        record, dataset, features = build_simple_dataset()
        # encode dataset
        for i, item in enumerate(dataset):
            if i % 4 == 0:
                # mock start of a sequence
                record.write_item(item, True)
            else:
                record.write_item(item, False)
        record.close_recordfile()
        record.dump()
        # decode dataset
        for i, item in enumerate(record.read_record()):
            for feature in features:
                nptest.assert_equal(item[feature], dataset[i][feature], err_msg="", verbose=True)
        loaded_record = SeqRecord.load_record_from_dict("/home/azureuser/data/data4record/records")

    def test_idx4segment(self):
        """Having the record written (and various attributes setup), generate an index mapping for
        specific segment len."""
        record, dataset, features = build_simple_dataset()
        seq_len = 4
        # encode dataset
        for i, item in enumerate(dataset):
            if i % seq_len == 0:
                # mock start of a sequence
                record.write_item(item, True)
            else:
                record.write_item(item, False)
        record.close_recordfile()

        # segment len =2, sequence len =4, full features
        seg_len = 2
        idx4segment = record.get_proto4segment(segment_len=seg_len)
        items = concate_list(idx4segment["file2segment_items"])
        self.assertEqual(len(items), 10)
        ids = [item_idx for _, item_idx in items]
        self.assertListEqual(ids, list(range(10)))
        for i, (is_segment_start, item) in enumerate(items):
            if i in [0, 1, 2, 4, 5, 6, 8]:
                self.assertTrue(is_segment_start)
            else:
                self.assertFalse(is_segment_start)
        heads = idx4segment["head4segment"]
        for i, segment in enumerate(record.read_all_segments(idx4segment)):
            for j in range(seg_len):
                for feature in features:
                    nptest.assert_equal(dataset[heads[i] + j][feature], segment[feature][j], err_msg="", verbose=True)

        # segment len =4, sequence len =4, full features
        seg_len = 4
        idx4segment = record.get_proto4segment(segment_len=seg_len)
        items = concate_list(idx4segment["file2segment_items"])
        ids = [item_idx for _, item_idx in items]
        self.assertEqual(len(ids), 8)
        ids = [item_idx for _, item_idx in items]
        self.assertListEqual(ids, [0, 1, 2, 3, 4, 5, 6, 7])
        for i, (is_segment_start, item) in enumerate(items):
            if i in [0, 4]:
                self.assertTrue(is_segment_start)
            else:
                self.assertFalse(is_segment_start)
        heads = idx4segment["head4segment"]
        for i, segment in enumerate(record.read_all_segments(idx4segment)):
            for j in range(seg_len):
                for feature in features:
                    nptest.assert_equal(dataset[heads[i] + j][feature], segment[feature][j], err_msg="", verbose=True)

        # segment len =3, sequence len =4, full feature
        seg_len = 3
        idx4segment = record.get_proto4segment(segment_len=seg_len)
        self.assertEqual(len(ids), 8)
        items = concate_list(idx4segment["file2segment_items"])
        ids = [item_idx for _, item_idx in items]
        self.assertListEqual(ids, [0, 1, 2, 3, 4, 5, 6, 7])
        for i, (is_segment_start, item) in enumerate(items):
            if i in [0, 1, 4, 5]:
                self.assertTrue(is_segment_start)
            else:
                self.assertFalse(is_segment_start)
        heads = idx4segment["head4segment"]
        for i, segment in enumerate(record.read_all_segments(idx4segment)):
            for j in range(seg_len):
                for feature in features:
                    nptest.assert_equal(dataset[heads[i] + j][feature], segment[feature][j], err_msg="", verbose=True)

    def test_idx4segment_brokenfeatures(self):
        """Test the idx4segment with some features from dataset missing."""
        # segment len = 3, sequence len =4, break two features
        record, dataset, features = build_broken_dataset([3, 4])
        seq_len = 4
        # encode dataset
        for i, item in enumerate(dataset):
            if i % seq_len == 0:
                # mock start of a sequence
                record.write_item(item, True)
            else:
                record.write_item(item, False)
        record.close_recordfile()

        seg_len = 3
        idx4segment = record.get_proto4segment(segment_len=seg_len)
        items = concate_list(idx4segment["file2segment_items"])
        ids = [item_idx for _, item_idx in items]
        self.assertEqual(len(ids), 6)
        self.assertListEqual(ids, [0, 1, 2, 5, 6, 7])
        for i, (is_segment_start, item) in enumerate(items):
            if item in [0, 5]:
                self.assertTrue(is_segment_start)
            else:
                self.assertFalse(is_segment_start)
        heads = idx4segment["head4segment"]
        for i, segment in enumerate(record.read_all_segments(idx4segment)):
            for j in range(seg_len):
                for feature in features:
                    nptest.assert_equal(dataset[heads[i] + j][feature], segment[feature][j], err_msg="", verbose=True)

        seg_len = 3
        idx4segment = record.get_proto4segment(segment_len=seg_len, sub_features=["ceo"])
        items = concate_list(idx4segment["file2segment_items"])
        ids = [item_idx for _, item_idx in items]
        self.assertEqual(len(items), 6)
        self.assertListEqual(ids, [0, 1, 2, 5, 6, 7])
        for i, (is_seg_start, item) in enumerate(items):
            if item in [0, 5]:
                self.assertTrue(is_seg_start)
            else:
                self.assertFalse(is_seg_start)

                seg_len = 3
        heads = idx4segment["head4segment"]
        for i, segment in enumerate(record.read_all_segments(idx4segment)):
            for j in range(seg_len):
                for feature in features:
                    nptest.assert_equal(dataset[heads[i] + j][feature], segment[feature][j], err_msg="", verbose=True)

        idx4segment = record.get_proto4segment(segment_len=seg_len, sub_features=["l5", "l1"])
        items = concate_list(idx4segment["file2segment_items"])
        ids = [item_idx for _, item_idx in items]
        self.assertEqual(len(ids), 8)
        self.assertListEqual(ids, [0, 1, 2, 3, 4, 5, 6, 7])
        for i, (is_seg_start, item) in enumerate(items):
            if item in [0, 1, 4, 5]:
                self.assertTrue(is_seg_start)
            else:
                self.assertFalse(is_seg_start)

        heads = idx4segment["head4segment"]
        for i, segment in enumerate(record.read_all_segments(idx4segment)):
            for j in range(seg_len):
                for feature in features:
                    nptest.assert_equal(dataset[heads[i] + j][feature], segment[feature][j], err_msg="", verbose=True)
        # segment len = 3, sequence len =4, break two features
        record, dataset, features = build_broken_dataset([3, 6])
        seq_len = 4
        # encode dataset
        for i, item in enumerate(dataset):
            if i % seq_len == 0:
                # mock start of a sequence
                record.write_item(item, True)
            else:
                record.write_item(item, False)
        record.close_recordfile()

        seg_len = 3
        idx4segment = record.get_proto4segment(segment_len=seg_len)
        items = concate_list(idx4segment["file2segment_items"])
        ids = [item_idx for _, item_idx in items]
        self.assertEqual(len(ids), 3)
        self.assertListEqual(ids, [0, 1, 2])
        for i, (is_seg_start, item) in enumerate(items):
            if item in [0, 5]:
                self.assertTrue(is_seg_start)
            else:
                self.assertFalse(is_seg_start)
        heads = idx4segment["head4segment"]
        for i, segment in enumerate(record.read_all_segments(idx4segment)):
            for j in range(seg_len):
                for feature in features:
                    nptest.assert_equal(dataset[heads[i] + j][feature], segment[feature][j], err_msg="", verbose=True)

        # segment len = 3, sequence len =4, break two features
        record, dataset, features = build_broken_dataset([2, 6])
        seq_len = 4
        # encode dataset
        for i, item in enumerate(dataset):
            if i % seq_len == 0:
                # mock start of a sequence
                record.write_item(item, True)
            else:
                record.write_item(item, False)
        record.close_recordfile()

        seg_len = 3
        idx4segment = record.get_proto4segment(segment_len=seg_len)
        items = concate_list(idx4segment["file2segment_items"])
        ids = [item_idx for _, item_idx in items]
        self.assertEqual(len(ids), 0)

    def test_segmentreader(self):
        """test segment sharding and stream reading."""
        record, dataset, features = build_seq_dataset()
        seq_len = 4
        # encode dataset
        for i, item in enumerate(dataset):
            if i % seq_len == 0:
                # mock start of a sequence
                record.write_item(item, True)
            else:
                record.write_item(item, False)
        record.close_recordfile()

        # segment len =3, sequence len =4, full features
        seg_len = 3
        idx4segment = record.get_proto4segment(segment_len=seg_len)
        headidx = [0, 1, 4, 5]
        itemidx = [0, 1, 2, 3, 4, 5, 6, 7]
        items = concate_list(idx4segment["file2segment_items"])
        ids = [item_idx for _, item_idx in items]
        self.assertListEqual(idx4segment["head4segment"], headidx)
        self.assertListEqual(ids, itemidx)
        head_idx = 1
        item = record.read_one_segment(seg_len, head_idx)
        for feature in features:
            data = np.stack([dataset[j][feature] for j in range(head_idx, head_idx + seg_len)], axis=0)
            nptest.assert_array_equal(item[feature], data)

        head_idx = 0
        item = record.read_one_segment(seg_len, head_idx)
        for feature in features:
            data = np.stack([dataset[j][feature] for j in range(head_idx, head_idx + seg_len)], axis=0)
            nptest.assert_array_equal(item[feature], data)

        head_idx = 4
        item = record.read_one_segment(seg_len, head_idx)
        for feature in features:
            data = np.stack([dataset[j][feature] for j in range(head_idx, head_idx + seg_len)], axis=0)
            nptest.assert_array_equal(item[feature], data)


def build_simple_dataset():
    """Generate a fake dataset to test methods of Record.

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
    record = SeqRecord(rootdir, features)
    return record, dataset, features


def build_broken_dataset(feature_is_none_list: List[int]):
    """Generate a fake dataset to test methods of Record where some features does not exist.

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
    record = SeqRecord(rootdir, features)
    return record, dataset, features


def build_seq_dataset():
    """Generate a fake dataset to test methods of Record, where data items from the same feature
    shares numpy array shapes.

    Returns:
        _type_: _description_
    """
    rootdir = "/home/azureuser/data/data4record"
    seq_len = 10
    features = ["l1", "l5", "60", "63senior", "ceo"]
    dataset = [{} for _ in range(seq_len)]
    for feature in features:
        shape = (np.random.randint(1, 5), np.random.randint(2, 7))
        dtype = np.random.choice(["float32", "int", "bool"])
        for i in range(seq_len):
            dataset[i][feature] = np.random.rand(shape[0], shape[1]) if dtype == "float32" else np.ones(shape=shape)
    record = SeqRecord(rootdir, features)
    return record, dataset, features


if __name__ == "__main__":
    np.random.seed(0)
    unittest.main()
