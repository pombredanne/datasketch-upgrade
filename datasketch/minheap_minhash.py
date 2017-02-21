import copy
import hashlib
import heapq
import struct

from datasketch.minhash import MinHash


# http://en.wikipedia.org/wiki/Mersenne_prime
_mersenne_prime = (1 << 61) - 1
_max_hash = (1 << 32) - 2
_empty_val = _max_hash + 1
_hash_range = (1 << 32)
_hash_func_dict = {
    'sha1': hashlib.sha1,
    'sha256': hashlib.sha256,
    'sha512': hashlib.sha512,
    'sha384': hashlib.sha384,
    'md5': hashlib.md5
}


class MaxHeapObj(object):
    def __init__(self, val):
        self.val = val

    def __lt__(self, other):
        return self.val > other.val

    def __eq__(self, other):
        return self.val == other.val

    def __str__(self):
        return str(self.val)


class UniqueMaxHeap(object):
    def __init__(self, capacity, arr=None):
        self.capacity = capacity
        self.MAX_VAL = _empty_val
        if arr:
            self.heap = map(lambda val: MaxHeapObj(val), arr)
            self.vals = set(arr)
            heapq.heapify(self.heap)

            while len(self.heap) < self.capacity:
                heapq.heappush(self.heap, MaxHeapObj(self.MAX_VAL))

        else:
            self.heap = [MaxHeapObj(self.MAX_VAL) for _ in range(capacity)]
            self.vals = set()

    def push(self, item):
        if item not in self.vals:
            self.vals.add(item)
            value = heapq.heappushpop(self.heap, MaxHeapObj(item)).val
            if value != self.MAX_VAL:
                self.vals.remove(value)

    def pop(self):
        value = heapq.heappop(self.heap).val
        self.vals.remove(value)
        return value

    def sorted_values(self):
        return sorted(map(lambda obj: obj.val, self.heap))


class MinHashMinHeap(MinHash):
    __slots__ = ('heap', 'hashobj', 'k_val', 'hashstr')

    def __init__(self, k_val=128, hashobj=None, hashstr='sha1', minheap_array=None):
        if hashobj is None:
            self.hashobj = _hash_func_dict[hashstr]
        else:
            self.hashobj = hashobj
        self.hashstr = hashstr
        self.k_val = k_val
        if minheap_array is not None:
            self.heap = UniqueMaxHeap(k_val, minheap_array)
        else:
            self.heap = UniqueMaxHeap(k_val)

    def update(self, b):
        hv = struct.unpack('<I', self.hashobj(b).digest()[:4])[0]
        self.heap.push(hv)

    def bytesize(self):
        '''
        Returns the size of this MinHash in bytes.
        To be used in serialization.
        '''
        return struct.calcsize('ii%ds%dI' % (len(self.hashstr), self.k_val))

    def serialize(self, buf):
        '''
        Serializes this MinHash into bytes, store in `buf`.
        This is more efficient than using pickle.dumps on the object.
        '''
        if len(buf) < self.bytesize():
            raise ValueError("The buffer does not have enough space\
                    for holding this MinHash.")
        fmt = "ii%ds%dI" % (len(self.hashstr), self.k_val)
        struct.pack_into(fmt, buf, 0, self.k_val, len(self.hashstr), self.hashstr, *self.hashvalues)

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
            out = struct.unpack_from('%ds%dI' % (hashstr_len, k_val), buf, offset)
        except TypeError:
            out = struct.unpack_from('%ds%dI' % (hashstr_len, k_val), buffer(buf), offset)
        hashstr = out[0]
        minheap_array = out[1:]
        return cls(k_val=k_val, minheap_array=minheap_array, hashstr=hashstr)

    def __getstate__(self):
        '''
        This function is called when pickling the MinHash.
        Returns a bytearray which will then be pickled.
        Note that the bytes returned by the Python pickle.dumps is not
        the same as the buffer returned by this function.
        '''
        buf = bytearray(self.bytesize())
        fmt = "ii%ds%dI" % (len(self.hashstr), self.k_val)
        struct.pack_into(fmt, buf, 0, self.k_val, len(self.hashstr), self.hashstr, *self.hashvalues)
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
            out = struct.unpack_from('%ds%dI' % (hashstr_len, k_val), buf, offset)
        except TypeError:
            out = struct.unpack_from('%ds%dI' % (hashstr_len, k_val), buffer(buf), offset)
        hashstr = out[0]
        minheap_array = out[1:]
        self.__init__(k_val=k_val, minheap_array=minheap_array, hashstr=hashstr)

    @property
    def hashvalues(self):
        return self.heap.sorted_values()

    @property
    def hashvalues_set(self):
        return self.heap.vals

    def is_empty(self):
        return len(self.heap.vals) == 0

    def jaccard(self, other):
        if self.k_val != other.k_val:
            raise ValueError("Cannot compute Jaccard given MinHash with\
                    different k_val.")
        if not isinstance(other, MinHashMinHeap):
            raise ValueError("Cannot compute Jaccard of non-MinHashMinHeap")

        hash_space = self.hashvalues_set.union(other.hashvalues_set)
        minhash_intersection = self.hashvalues_set.intersection(other.hashvalues_set)
        k_val = min(max(len(self.hashvalues_set), len(other.hashvalues_set)), self.k_val)
        return float(len(hash_space.intersection(minhash_intersection))) / float(k_val)

    def __eq__(self, other):
        return self.hashobj == other.hashobj and self.hashvalues == other.hashvalues
