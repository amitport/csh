import tensorflow as tf

LARGEPRIME = 2 ** 61 - 1


class CSVec(object):
    """ Count Sketch of a vector

    Treating a vector as a stream of tokens with associated weights,
    this class computes the count sketch of an input vector, and
    supports operations on the resulting sketch.

    """

    def __init__(self, d, c, r, seed=42):
        """ Constductor for CSVec

        Args:
            d: the cardinality of the skteched vector
            c: the number of columns (buckets) in the sketch
            r: the number of rows in the sketch
            seed: random seed for hash generation
        """
        self.d = d
        self.c = c  # num of columns
        self.r = r  # num of rows

        # initialize the sketch to all zeros
        self.table = tf.Variable(tf.zeros((r, c)), trainable=False)

        # initialize hashing functions for each row:
        # 2 random numbers for bucket hashes + 4 random numbers for
        # sign hashes
        # maintain existing random state so we don't mess with
        # the main module trying to set the random seed but still
        # get reproducible hashes for the same value of r

        hashes = tf.random.Generator.from_seed(seed) \
            .uniform(minval=0, maxval=LARGEPRIME, shape=(6, r, 1), dtype=tf.int64)

        h1, h2, h3, h4, h5, h6 = tf.unstack(hashes)

        # tokens are the indices of the vector entries
        tokens = tf.expand_dims(tf.range(self.d, dtype=tf.int64), axis=0)

        # computing bucket hashes (2-wise independence)
        self.buckets = ((h1 * tokens) + h2) % LARGEPRIME % self.c

        # computing sign hashes (4 wise independence)
        self.signs = tf.cast(((((h3 * tokens + h4) * tokens + h5) * tokens + h6) % LARGEPRIME % 2) * 2 - 1,
                             dtype=tf.float32)

    def accumulateVec(self, vec):
        """ Sketches a vector and adds the result to self

        Args:
            vec: the vector to be sketched
        """

        # TODO (!) use this when moving to tensorflow 2.4:
        # return tf.math.bincount(
        #     self.buckets,
        #     weights=self.signs * vec,
        #     minlength=self.c,
        #     axis=-1
        # )

        # the vector is sketched to each row independently
        for r in range(self.r):
            # bincount computes the sum of all values in the vector
            # that correspond to each bucket
            tf.compat.v1.scatter_add(self.table, r, tf.math.bincount(
                tf.cast(self.buckets[r, :], tf.int32),
                weights=self.signs[r, :] * vec,
                minlength=self.c
            ))
