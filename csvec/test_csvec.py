import tensorflow as tf

from csvec import CSVec


class CSVecTest(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        self.cs = CSVec(d=4, c=3, r=5, seed=42)

    def test_init(self):
        self.assertAllEqual(tf.zeros((5, 3)), self.cs.table)

    def test_randomness(self):
        self.cs2 = CSVec(d=4, c=3, r=5, seed=42)
        self.assertAllClose(self.cs.signs, self.cs2.signs)
        self.assertAllClose(self.cs.buckets, self.cs2.buckets)

    def test_accumulateVec(self):
        self.cs.accumulateVec(tf.ones((4,)))
        self.assertAllEqual([[1, 1, 0],
                             [1, -1, 0],
                             [0, 0, 0],
                             [-2, 1, -1],
                             [0, 1, 1]], self.cs.table)


if __name__ == '__main__':
    tf.test.main()
