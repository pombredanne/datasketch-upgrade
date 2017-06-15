import hashlib
import struct

from hyperloglog import HyperLogLog

from datasketch.minhash import MinHash
from datasketch.ophr_minhash import MinHashOPHR

_hash_func_dict = {
    'sha1': hashlib.sha1,
    'sha256': hashlib.sha256,
    'sha512': hashlib.sha512,
    'sha384': hashlib.sha384,
    'md5': hashlib.md5
}


class Partition(object):
    def __init__(self, k_val, minhash_cls, hashobj):
        self.minhash = minhash_cls(k_val=k_val, hashobj=hashobj)
        self.minhash_cls = minhash_cls
        self.hll = HyperLogLog(.01)
        self._cardinality = None

    def update(self, value):
        self.minhash._update(value)
        self.hll.add(value)
        self._cardinality = None

    def __len__(self):
        return len(self.minhash)

    def __eq__(self, other):
        return self.minhash == other.minhash

    def is_empty(self):
        return self.minhash.is_empty()

    def jaccard(self, other):
        if not isinstance(other, Partition):
            raise ValueError("Input must be a Partition.")

        if self.minhash_cls != other.minhash_cls:
            raise ValueError("Input minhash class must be the same.")

        return self.minhash.jaccard(other.minhash)

    def cardinality(self):
        if self._cardinality is None:
            self._cardinality = len(self.hll)
        return self._cardinality


class PartitionMinHash(MinHash):
    __slots__ = ('hashobj', 'k_val', 'partitions', 'hashstr')

    def __init__(self, k_val=128, hashobj=None, hashstr='sha1', num_partitions=3, minhash_cls=None):
        if hashobj is None:
            self.hashobj = _hash_func_dict[hashstr]
        else:
            self.hashobj = hashobj

        if minhash_cls is None:
            minhash_cls = MinHashOPHR

        self.partitions = [Partition(k_val, minhash_cls, self.hashobj) for _ in range(num_partitions)]
        self.hashstr = hashstr
        self.k_val = k_val

    def update(self, b):
        hv = struct.unpack('<I', self.hashobj(b).digest()[:4])[0]
        bucket = hv % len(self.partitions)
        self.partitions[bucket].update(hv)

    def jaccard(self, other):
        if self.k_val != len(other):
            raise ValueError("Cannot compute Jaccard given MinHash with\
                    different numbers of permutation functions")
        if len(self.partitions) != len(other.partitions):
            raise ValueError("Cannot compute Jaccard of other PartitionMinhash "
                             "with different number of partitions.")
        if not isinstance(other, PartitionMinHash):
            raise ValueError("Cannot compute Jaccard of non-PartitionMinhash")

        cardinalities = []
        jaccards = []
        cardinality_totals = [0.0, 0.0]
        for i in range(len(self.partitions)):
            partition_jaccard = self.partitions[i].jaccard(other.partitions[i])
            jaccards.append(partition_jaccard)

            my_cardinality = self.partitions[i].cardinality()
            other_cardinality = other.partitions[i].cardinality()

            cardinality_totals[0] += my_cardinality
            cardinality_totals[1] += other_cardinality
            cardinality = (my_cardinality, other_cardinality)
            cardinalities.append(cardinality)

        total_jaccard = 0
        for i in range(len(self.partitions)):
            my_cardinality, other_cardinality = cardinalities[i]
            if cardinality_totals[0] != 0:
                my_cardinality_weight = my_cardinality / cardinality_totals[0]
            else:
                my_cardinality_weight = 0

            if cardinality_totals[1] != 0:
                other_cardinality_weight = other_cardinality / cardinality_totals[1]
            else:
                other_cardinality_weight = 0

            # weight jaccards by average percentage weighting
            total_jaccard += ((my_cardinality_weight + other_cardinality_weight) / 2) * jaccards[i]

        # if totally empty, default to 1
        if total_jaccard == 0 and self.is_empty() and other.is_empty():
            return 1
        return total_jaccard

    def __len__(self):
        '''
        Return the size of the MinHash
        '''
        return len(self.partitions[0])

    def __eq__(self, other):
        return self.partitions == other.partitions

    def is_empty(self):
        for partition in self.partitions:
            if not partition.is_empty():
                return False
        return True


class BetterWeightedPartitionMinHash(PartitionMinHash):
    def _get_alpha(self, min_cardinality, max_cardinality, similarity):
        return (min_cardinality - similarity * max_cardinality) / (similarity + 1)

    def _get_union_size(self, p1, p2, similarity):
        min_cardinality = min(p1.cardinality(), p2.cardinality())
        max_cardinality = max(p1.cardinality(), p2.cardinality())
        return max_cardinality + self._get_alpha(min_cardinality, max_cardinality, similarity)

    def jaccard(self, other):
        if self.k_val != len(other):
            raise ValueError("Cannot compute Jaccard given MinHash with\
                    different numbers of permutation functions")
        if len(self.partitions) != len(other.partitions):
            raise ValueError("Cannot compute Jaccard of other PartitionMinhash "
                             "with different number of partitions.")
        if not isinstance(other, PartitionMinHash):
            raise ValueError("Cannot compute Jaccard of non-PartitionMinhash")

        union_sizes = []
        jaccards = []
        for i in range(len(self.partitions)):
            p1 = self.partitions[i]
            p2 = other.partitions[i]
            partition_jaccard = p1.jaccard(p2)

            jaccards.append(partition_jaccard)
            union_sizes.append(self._get_union_size(p1, p2, partition_jaccard))

        # if empty comparisons, then return 1
        if self.is_empty() and other.is_empty():
            return 1

        total_jaccard = 0
        total_union_size = sum(union_sizes)
        for i in range(len(jaccards)):
            # weight jaccards by weighting of the union_size vs the total
            total_jaccard += (union_sizes[i] / total_union_size) * jaccards[i]

        return total_jaccard
