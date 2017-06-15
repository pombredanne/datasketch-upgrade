import struct
import unittest

from datasketch.partition_minhash import PartitionMinHash, BetterWeightedPartitionMinHash


class FakeHash(object):
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


class TestPartitionMinhash(unittest.TestCase):
    def test_init(self):
        m1 = PartitionMinHash(4)
        m2 = PartitionMinHash(4)
        self.assertEqual(m1, m2)
        self.assertEqual(m1.k_val, m2.k_val)
        self.assertEqual(m1.partitions, m1.partitions)

    def test_is_empty(self):
        m = PartitionMinHash(4, hashobj=FakeHash)
        self.assertTrue(m.is_empty())
        m.update(1)
        self.assertFalse(m.is_empty())

    def test_update(self):
        m1 = PartitionMinHash(4, hashobj=FakeHash)
        m2 = PartitionMinHash(4, hashobj=FakeHash)
        m1.update(12)
        self.assertTrue(m1 != m2)

    def test_jaccard(self):
        m1 = PartitionMinHash(4, hashobj=FakeHash)
        m2 = PartitionMinHash(4, hashobj=FakeHash)
        self.assertEqual(m1.jaccard(m2), 1.0)
        m2.update(12)
        self.assertEqual(m1.jaccard(m2), 0.0)
        m1.update(13)
        self.assertEqual(m1.jaccard(m2), 0.0)
        m1.update(12)
        self.assertEqual(m1.jaccard(m2), 0.75)
        m2.update(13)
        self.assertEqual(m1.jaccard(m2), 1.0)
        m1.update(14)
        self.assertEqual(m1.jaccard(m2), 2./3)
        m2.update(14)
        self.assertEqual(m1.jaccard(m2), 1.0)

    def test_better_weighting_jaccard(self):
        m1 = BetterWeightedPartitionMinHash(4, hashobj=FakeHash)
        m2 = BetterWeightedPartitionMinHash(4, hashobj=FakeHash)
        self.assertEqual(m1.jaccard(m2), 1.0)
        m2.update(12)
        self.assertEqual(m1.jaccard(m2), 0.0)
        m1.update(13)
        self.assertEqual(m1.jaccard(m2), 0.0)
        m1.update(12)
        self.assertEqual(m1.jaccard(m2), 0.50)
        m2.update(13)
        self.assertEqual(m1.jaccard(m2), 1.0)
        m1.update(14)
        self.assertEqual(m1.jaccard(m2), 2./3)
        m2.update(14)
        self.assertEqual(m1.jaccard(m2), 1.0)

    def test_eq(self):
        m1 = PartitionMinHash(4, hashobj=FakeHash)
        m2 = PartitionMinHash(4, hashobj=FakeHash)
        m3 = PartitionMinHash(4, hashobj=FakeHash)
        m4 = PartitionMinHash(4, hashobj=FakeHash)
        m5 = PartitionMinHash(4, hashobj=FakeHash)
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

if __name__ == "__main__":
    unittest.main()
