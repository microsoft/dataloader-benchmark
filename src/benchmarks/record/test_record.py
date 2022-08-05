import unittest

import numpy as np
import numpy.testing as nptest

from record import Record


class Test_Record(unittest.TestCase):

    def test_encode_decode(self):
        """Testing encode and decode of record. No segment involved.
        """
        record, dataset, features = build_simple_dataset()
        # encode dataset
        for i, item in enumerate(dataset):
            if i % 4 == 0:
                # mock start of a sequence
                record.encode_item(item, True)
            else:
                record.encode_item(item, False)
        record.close()
        # decode dataset
        for i, item in enumerate(record.decode_record()):
            for feature in features:
                nptest.assert_equal(item[feature], dataset[i][feature], err_msg='', verbose=True)

        
    def test_idx4segment(self):
        pass

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
            dtype = np.random.choice(['float32', 'int', 'bool'])
            dataset[i][feature] = np.random.rand(shape[0], shape[1]) if dtype == 'float32' else np.ones(shape=shape)
    record = Record(rootdir, features)
    return record, dataset, features



if __name__ == "__main__":
    np.random.seed(0)
    unittest.main()
