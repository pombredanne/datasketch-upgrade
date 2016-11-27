import copy
import hashlib
import struct

import numpy as np

from datasketch.minhash import MinHash

# http://en.wikipedia.org/wiki/Mersenne_prime
_mersenne_prime = (1 << 61) - 1
_max_hash = (1 << 32) - 1
_empty_val = _max_hash + 1
_hash_range = (1 << 32)
_hash_func_dict = {
    'sha1': hashlib.sha1,
    'sha256': hashlib.sha256,
    'sha512': hashlib.sha512,
    'sha384': hashlib.sha384,
    'md5': hashlib.md5
}


class MinHashOPHR(MinHash):
    __slots__ = ('_hashvalues', '_dense_hashvalues', 'hashobj', 'k_val', 'rot_constant', 'hashstr')

    def __init__(self, k_val=128, hashobj=None, hashstr='sha1', _hashvalues=None):
        if hashobj is None:
            self.hashobj = _hash_func_dict[hashstr]
        else:
            self.hashobj = hashobj
        self.hashstr = hashstr
        self.k_val = k_val
        self._hashvalues = _hashvalues or self._init__hashvalues()
        self._dense_hashvalues = None  # set this later to cache dense hashvalues
        self.rot_constant = _max_hash / self.k_val + 1

    def _init__hashvalues(self):
        return np.ones(self.k_val, dtype=np.uint64) * _empty_val

    def update(self, b):
        hv = struct.unpack('<I', self.hashobj(b).digest()[:4])[0]
        bucket = hv % self.k_val
        self._hashvalues[bucket] = min(self._hashvalues[bucket], hv)
        self._dense_hashvalues = None

    def bytesize(self):
        '''
        Returns the size of this MinHash in bytes.
        To be used in serialization.
        '''
        # Use 4 bytes to store the number of hash values
        length_size = struct.calcsize('i')
        # Use 4 bytes to store each hash value as we are using the lower 32 bit
        hashvalue_size = struct.calcsize('L')

        hashstr_size_enc = struct.calcsize('i')
        hashstr_size = struct.calcsize('s') * len(self.hashstr)
        return max(length_size + self.k_val * hashvalue_size + hashstr_size + hashstr_size_enc, 48)

    def serialize(self, buf):
        '''
        Serializes this MinHash into bytes, store in `buf`.
        This is more efficient than using pickle.dumps on the object.
        '''
        if len(buf) < self.bytesize():
            raise ValueError("The buffer does not have enough space\
                    for holding this MinHash.")
        fmt = "ii%ds%dL" % (len(self.hashstr), self.k_val)
        struct.pack_into(fmt, buf, 0, self.k_val, len(self.hashstr), self.hashstr, *self._hashvalues)

    @classmethod
    def deserialize(cls, buf):
        '''
        Reconstruct a MinHash from a byte buffer.
        This is more efficient than using the pickle.loads on the pickled
        bytes.
        '''
        try:
            k_val, = struct.unpack_from('i', buf, 0)
        except TypeError:
            k_val, = struct.unpack_from('i', buffer(buf), 0)
        offset = struct.calcsize('i')
        try:
            hashstr_len, = struct.unpack_from('i', buf, offset)
        except TypeError:
            hashstr_len, = struct.unpack_from('i', buffer(buf), offset)
        offset += struct.calcsize('i')
        try:
            hashstr, = struct.unpack_from('%ds' % hashstr_len, buf, offset)
        except:
            hashstr, = struct.unpack_from('%ds' % hashstr_len, buffer(buf), offset)
        offset += struct.calcsize('s') * hashstr_len
        try:
            _hashvalues = struct.unpack_from('%dL' % k_val, buf, offset)
        except TypeError:
            _hashvalues = struct.unpack_from('%dL' % k_val, buffer(buf), offset)
        return cls(k_val=k_val, _hashvalues=_hashvalues, hashstr=hashstr)

    def __getstate__(self):
        '''
        This function is called when pickling the MinHash.
        Returns a bytearray which will then be pickled.
        Note that the bytes returned by the Python pickle.dumps is not
        the same as the buffer returned by this function.
        '''
        buf = bytearray(self.bytesize())
        fmt = "ii%ds%dL" % (len(self.hashstr), self.k_val)
        struct.pack_into(fmt, buf, 0, self.k_val, len(self.hashstr), self.hashstr, *self._hashvalues)
        return buf

    def __setstate__(self, buf):
        '''
        This function is called when unpickling the MinHash.
        Initialize the object with data in the buffer.
        Note that the input buffer is not the same as the input to the
        Python pickle.loads function.
        '''
        try:
            k_val, = struct.unpack_from('i', buf, 0)
        except TypeError:
            k_val, = struct.unpack_from('i', buffer(buf), 0)
        offset = struct.calcsize('i')
        try:
            hashstr_len, = struct.unpack_from('i', buf, offset)
        except TypeError:
            hashstr_len, = struct.unpack_from('i', buffer(buf), offset)
        offset += struct.calcsize('i')
        try:
            hashstr, = struct.unpack_from('%ds' % hashstr_len, buf, offset)
        except:
            hashstr, = struct.unpack_from('%ds' % hashstr_len, buffer(buf), offset)
        offset += struct.calcsize('s') * hashstr_len
        try:
            _hashvalues = struct.unpack_from('%dL' % k_val, buf, offset)
        except TypeError:
            _hashvalues = struct.unpack_from('%dL' % k_val, buffer(buf), offset)
        self.__init__(k_val=k_val, _hashvalues=_hashvalues, hashstr=hashstr)

    @property
    def hashvalues(self):
        if self._dense_hashvalues is None:
            self._dense_hashvalues = copy.copy(self._hashvalues)
            for i in range(len(self._hashvalues)):
                if self._hashvalues[i] == _empty_val:
                    j = (i + 1) % self.k_val
                    distance = 1
                    while j != i:
                        if self._hashvalues[j] != _empty_val:
                            self._dense_hashvalues[i] = (self._hashvalues[j] + self.rot_constant * distance) % _max_hash
                            break
                        j = (j + 1) % self.k_val
                        distance += 1
        return self._dense_hashvalues

    def is_empty(self):
        return not np.any(self._hashvalues != _empty_val)

    def jaccard(self, other):
        if self.k_val != len(other):
            raise ValueError("Cannot compute Jaccard given MinHash with\
                    different numbers of permutation functions")
        if not isinstance(other, MinHashOPHR):
            raise ValueError("Cannot compute Jaccard of non-MinHashOPHR")

        return np.float(np.count_nonzero(self.hashvalues == other.hashvalues)) / \
               np.float(self.k_val)

    def __eq__(self, other):
        return self.hashobj == other.hashobj and \
               np.array_equal(self._hashvalues, other._hashvalues)
