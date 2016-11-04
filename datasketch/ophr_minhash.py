import copy
import struct
from hashlib import sha1

import numpy as np

from datasketch.minhash import MinHash

# http://en.wikipedia.org/wiki/Mersenne_prime
_mersenne_prime = (1 << 61) - 1
_max_hash = (1 << 32) - 1
_empty_val = _max_hash + 1
_hash_range = (1 << 32)


class MinHashOPHR(MinHash):
    __slots__ = ('_hashvalues', 'seed', 'hashobj', 'k_val', 'rot_constant')

    def __init__(self, k_val=128, hashobj=sha1):
        self.hashobj = hashobj
        self.k_val = k_val
        self._hashvalues = self._init__hashvalues()
        self.rot_constant = _max_hash / self.k_val + 1

    def _init__hashvalues(self):
        return np.ones(self.k_val, dtype=np.uint64) * _empty_val

    def update(self, b):
        hv = struct.unpack('<I', self.hashobj(b).digest()[:4])[0]
        bucket = hv % self.k_val
        self._hashvalues[bucket] = min(self._hashvalues[bucket], hv)

    @property
    def hashvalues(self):
        dense_hashvals = copy.copy(self._hashvalues)
        for i in range(len(self._hashvalues)):
            if self._hashvalues[i] == _empty_val:
                j = (i + 1) % self.k_val
                distance = 1
                while j != i:
                    if self._hashvalues[j] != _empty_val:
                        dense_hashvals[i] = (self._hashvalues[j] + self.rot_constant * distance) % _max_hash
                        break
                    j = (j + 1) % self.k_val
                    distance += 1
        return dense_hashvals

    def is_empty(self):
        return not np.any(self._hashvalues != _empty_val)

    def jaccard(self, other):
        if len(self) != len(other):
            raise ValueError("Cannot compute Jaccard given MinHash with\
                    different numbers of permutation functions")
        if not isinstance(other, MinHashOPHR):
            raise ValueError("Cannot compute Jaccard of non-MinHashOPHR")

        return np.float(np.count_nonzero(self.hashvalues == other.hashvalues)) / \
               np.float(len(self))

    def __eq__(self, other):
        return self.hashobj == other.hashobj and \
               np.array_equal(self._hashvalues, other._hashvalues)
