import hashlib
import pickle
import struct
import unittest

import numpy as np

from datasketch.ophr_minhash import MinHashOPHR


class FakeHash(object):
    '''
    Implmenets the hexdigest required by HyperLogLog.
    '''

    def __init__(self, h):
        '''
        Initialize with an integer
        '''
        self.h = h

    def digest(self):
        '''
        Return the bytes representation of the integer
        '''
        return struct.pack('<Q', self.h)


class TestMinhashOPHR(unittest.TestCase):
    def test_init(self):
        m1 = MinHashOPHR(4)
        m2 = MinHashOPHR(4)
        self.assertTrue(np.array_equal(m1._hashvalues, m2._hashvalues))
        self.assertEqual(m1.k_val, m2.k_val)
        self.assertEqual(m1.rot_constant, m2.rot_constant)

    def test_is_empty(self):
        m = MinHashOPHR(4)
        self.assertTrue(m.is_empty())

    def test_update(self):
        m1 = MinHashOPHR(4, hashobj=FakeHash)
        m2 = MinHashOPHR(4, hashobj=FakeHash)
        self.assertFalse(np.any(m1._hashvalues != m2._hashvalues))
        m1.update(12)
        self.assertTrue(np.any(m1._hashvalues != m2._hashvalues))

    def test_dense_hashvalues(self):
        m1 = MinHashOPHR(4, hashobj=FakeHash)
        m2 = MinHashOPHR(4, hashobj=FakeHash)
        self.assertTrue(np.all(m1.hashvalues == 2 ** 32))

        m1.update(12)
        self.assertFalse(np.any(m1.hashvalues == 2 ** 32))

        m2.update(12)
        self.assertTrue(np.all(m1.hashvalues == m2.hashvalues))

    def test_jaccard(self):
        m1 = MinHashOPHR(4, hashobj=FakeHash)
        m2 = MinHashOPHR(4, hashobj=FakeHash)
        self.assertTrue(m1.jaccard(m2) == 1.0)
        m2.update(12)
        self.assertTrue(m1.jaccard(m2) == 0.0)
        m1.update(13)
        self.assertTrue(m1.jaccard(m2) == 0.0)
        m1.update(12)
        self.assertTrue(m1.jaccard(m2) < 1.0)
        m2.update(13)
        self.assertTrue(m1.jaccard(m2) == 1.0)

    def test_eq(self):
        m1 = MinHashOPHR(4, hashobj=FakeHash)
        m2 = MinHashOPHR(4, hashobj=FakeHash)
        m3 = MinHashOPHR(4, hashobj=FakeHash)
        m4 = MinHashOPHR(4, hashobj=FakeHash)
        m5 = MinHashOPHR(4, hashobj=FakeHash)
        m1.update(11)
        m2.update(12)
        m3.update(11)
        m4.update(11)
        m5.update(11)
        self.assertNotEqual(m1, m2)
        self.assertNotEqual(m1, m3)
        self.assertNotEqual(m1, m4)
        self.assertEqual(m1, m5)

        m1.update(12)
        m2.update(11)
        self.assertEqual(m1, m2)

    def test_serialization(self):
        m1 = MinHashOPHR(4, hashobj=hashlib.sha256, hashstr='sha256')
        out = pickle.dumps(m1)
        m1_deserialized = pickle.loads(out)

        self.assertEqual(m1.hashobj, m1_deserialized.hashobj)

    def test_serialization_2(self):
        m1 = MinHashOPHR(4, hashobj=hashlib.sha256, hashstr='sha256')
        buf = bytearray(m1.bytesize())
        m1.serialize(buf)
        new_m1 = MinHashOPHR.deserialize(buf)
        self.assertEqual(m1.hashobj, new_m1.hashobj)

if __name__ == "__main__":
    unittest.main()
