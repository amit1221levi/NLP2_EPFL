
import numpy as numpy
from nli import *
from shortcut import *
from transformers import DistilBertTokenizer
import numpy as np

def hello_A2():
  print('Hello A2!')

def test_compute_metrics(compute):
    predictions = [0, 1, 2, 1, 2, 2, 0, 0]
    gold_labels = [1, 1, 1, 2, 2, 0, 0, 1]
    assert compute(predictions, gold_labels)==(0.375, 0.4, 0.3333333333333333, 0.4), 'compute_metric wrong answer ❌'
    print('compute_metric test correct ✅')

def test_NLIDataset(dataset):
    assert dataset.pad_token=='[PAD]', 'NLIDataset wrong pad token ❌'
    assert dataset.pad_id==0, 'NLIDataset wrong pad token id ❌'
    assert len(dataset)==9815, 'NLIDataset wrong sample number/length ❌'
    sample_0 = \
    {'ids': [101, 1996, 2047, 2916, 2024, 3835, 2438, 102, 3071, 2428, 7777, 1996, 14751, 6666, 102], 'label': 1}
    assert dataset[0]==sample_0, 'NLIDataset wrong sample ids/label ❌'
    collate_5_0 = \
    np.array([[  101,  1996,  2047,  2916,  2024,  3835,  2438,   102,  3071,  2428,
            7777,  1996, 14751,  6666,   102,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0],
            [  101,  2023,  2609,  2950,  1037,  2862,  1997,  2035,  2400,  4791,
            1998,  1037,  3945,  3085,  7809,  1997,  2231,  3237,  4790,  1012,
                102,  1996,  2231,  3237,  4790,  7431,  2006,  1996,  4037,  2024,
            2025,  2583,  2000,  2022,  9022,  1012,   102,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0],
            [  101,  7910,  1045,  2123,  1005,  1056,  2113,  1045,  1045,  2031,
            3816,  6699,  2055,  2032,  7910,  2823,  1045,  2066,  2032,  2021,
            2012,  1996,  2168,  2335,  1045,  2293,  2000,  2156,  8307,  3786,
            2032,   102,  1045,  2066,  2032,  2005,  1996,  2087,  2112,  1010,
            2021,  2052,  2145,  5959,  3773,  2619,  3786,  2032,  1012,   102,
                0],
            [  101,  3398,  1045,  1045,  2228,  2026,  5440,  4825,  2003,  2467,
            2042,  1996,  2028,  7541,  2017,  2113,  1996,  7541,  2004,  2146,
            2004,  2009,  1005,  1055,  2009,  6010,  1996,  6263,  9181,  2017,
            2113,  1997,  2204,  2833,   102,  2026,  5440,  7884,  2024,  2467,
            2012,  2560,  1037,  3634,  2661,  2185,  2013,  2026,  2160,  1012,
                102],
            [  101,  1045,  2123,  1005,  1056,  2113,  8529,  2079,  2017,  2079,
            1037,  2843,  1997, 13215,   102,  1045,  2113,  3599,  1012,   102,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0]])
    collate_5_1 = np.array([1, 2, 0, 2, 2])
    assert (dataset.collate_fn(dataset[:5])[0].numpy()==collate_5_0).all() and \
    (dataset.collate_fn(dataset[:5])[1].numpy()==collate_5_1).all, 'NLIDataset wrong padding or collate_fn ❌'
    collate_5_1_dec = ['neutral', 'contradiction', 'entailment', 'contradiction', 'contradiction']
    assert dataset.decode_class(collate_5_1)==collate_5_1_dec, 'NLIDataset wrong decode_class ❌'
    print('NLIDataset test correct ✅')

def test_get_synonyms(get_synonyms):
    ans = get_synonyms('task')
    print('The synonyms for the word "task" are: ', ans)
