import hashlib
import pickle
import struct
import unittest

import numpy as np

from datasketch.minheap_minhash import MinHashMinHeap


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
        m1 = MinHashMinHeap(4)
        m2 = MinHashMinHeap(4)
        self.assertEqual(m1.hashvalues, m2.hashvalues)
        self.assertEqual(m1.k_val, m2.k_val)

    def test_is_empty(self):
        m = MinHashMinHeap(4)
        self.assertTrue(m.is_empty())

    def test_update(self):
        m1 = MinHashMinHeap(4, hashobj=FakeHash)
        m2 = MinHashMinHeap(4, hashobj=FakeHash)
        self.assertEqual(m1.hashvalues, m2.hashvalues)
        m1.update(12)
        self.assertNotEqual(m1.hashvalues, m2.hashvalues)
        self.assertFalse(m1.is_empty())

    def test_jaccard(self):
        m1 = MinHashMinHeap(4, hashobj=FakeHash)
        m2 = MinHashMinHeap(4, hashobj=FakeHash)
        m2.update(12)
        self.assertTrue(m1.jaccard(m2) == 0.0)
        m1.update(13)
        self.assertTrue(m1.jaccard(m2) == 0.0)
        m1.update(12)
        self.assertTrue(m1.jaccard(m2) == 0.5)
        m2.update(13)
        self.assertTrue(m1.jaccard(m2) == 1.0)

    def test_eq(self):
        m1 = MinHashMinHeap(4, hashobj=FakeHash)
        m2 = MinHashMinHeap(4, hashobj=FakeHash)
        m3 = MinHashMinHeap(4, hashobj=FakeHash)
        m4 = MinHashMinHeap(4, hashobj=FakeHash)
        m5 = MinHashMinHeap(4, hashobj=FakeHash)
        m1.update(11)
        m2.update(12)
        m3.update(11)
        m4.update(11)
        m5.update(11)
        self.assertNotEqual(m1, m2)
        self.assertEqual(m1, m3)
        self.assertEqual(m1, m4)
        self.assertEqual(m1, m5)

        m1.update(12)
        m2.update(11)
        self.assertEqual(m1, m2)

    def test_serialization(self):
        hashes = [(hashlib.sha1, 'sha1'), (hashlib.sha256, 'sha256'), (hashlib.sha512, 'sha512'), (hashlib.md5, 'md5')]
        for hashobj, hashstr in hashes:
            m1 = MinHashMinHeap(4, hashobj=hashobj, hashstr=hashstr)
            out = pickle.dumps(m1)
            m1_deserialized = pickle.loads(out)

            self.assertEqual(m1.hashobj, m1_deserialized.hashobj)
            self.assertEqual(m1, m1_deserialized)

    def test_serialization_2(self):
        hashes = [(hashlib.sha1, 'sha1'), (hashlib.sha256, 'sha256'), (hashlib.sha512, 'sha512'), (hashlib.md5, 'md5')]
        for hashobj, hashstr in hashes:
            m1 = MinHashMinHeap(4, hashobj=hashobj, hashstr=hashstr)
            m1.update('10')
            buf = bytearray(m1.bytesize())
            m1.serialize(buf)
            m1_deserialized = MinHashMinHeap.deserialize(buf)
            self.assertEqual(m1.hashobj, m1_deserialized.hashobj)
            self.assertEqual(m1, m1_deserialized)


if __name__ == "__main__":
    unittest.main()
