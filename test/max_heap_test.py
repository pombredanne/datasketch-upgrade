import unittest

from datasketch.minheap_minhash import UniqueMaxHeap


class TestMaxHeap(unittest.TestCase):
    def test_invariant(self):
        max_heap = UniqueMaxHeap(100)
        max_heap.push(3)
        max_heap.push(4)
        max_heap.push(5)
        max_heap.push(6)
        max_heap.push(7)
        max_heap.push(8)
        max_heap.push(1)
        max_heap.push(2)

        self.assertEqual(max_heap.pop(), 8)

    def test_sorted_values(self):
        max_heap = UniqueMaxHeap(100)
        max_heap.push(3)
        max_heap.push(4)
        max_heap.push(5)
        max_heap.push(6)
        max_heap.push(7)
        max_heap.push(8)
        max_heap.push(1)
        max_heap.push(2)

        self.assertEqual(max_heap.sorted_values(), [1, 2, 3, 4, 5, 6, 7, 8])

    def test_capacity(self):
        max_heap = UniqueMaxHeap(5)
        max_heap.push(3)
        max_heap.push(4)
        max_heap.push(5)
        max_heap.push(6)
        max_heap.push(7)
        max_heap.push(8)
        max_heap.push(1)
        max_heap.push(2)

        self.assertEqual(len(max_heap.sorted_values()), 5)
        self.assertEqual(max_heap.sorted_values(), [1, 2, 3, 4, 5])

    def test_input_array(self):
        max_heap = UniqueMaxHeap(5, [2, 3, 1])

        max_heap.push(5)
        max_heap.push(100)
        self.assertEqual(max_heap.sorted_values(), [1, 2, 3, 5, 100])

        self.assertEqual(max_heap.pop(), 100)
        self.assertEqual(max_heap.sorted_values(), [1, 2, 3, 5])

    def test_no_duplicates(self):
        max_heap = UniqueMaxHeap(5)

        max_heap.push(5)
        max_heap.push(5)
        max_heap.push(5)
        max_heap.push(5)
        self.assertEqual(max_heap.sorted_values(), [5])

        max_heap.push(4)
        max_heap.push(4)
        max_heap.push(4)
        max_heap.push(4)

        self.assertEqual(max_heap.sorted_values(), [4, 5])


if __name__ == "__main__":
    unittest.main()
