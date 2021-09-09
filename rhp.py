'''
This implementation was inpired by https://github.com/dongguosheng/lsh
'''

import numpy as np
from scipy import spatial


class RHP():
    def __init__(self, dims, bits, cols_per_bucket):
        self._dims = dims
        self._bits = bits
        self._cols_per_bucket = cols_per_bucket
        self.hyperplanes = None

    def init_hyperplane(self):
        '''
        This function creates the hyperplanes that will be used for hashing
        '''
        if self._bits <= 0 or self._dims <= 0:
            raise Exception("number of bit: %d, number of dims: %d\n" % 
                            (self._bits, self._dims))
        self.hyperplanes = np.array([])
        for column in range(self._cols_per_bucket):
            plane = np.random.randn(self._bits, self._dims)
            self.hyperplanes = np.append(self.hyperplanes, plane)
        self.hyperplanes = self.hyperplanes.reshape(self._cols_per_bucket, self._bits, self._dims) # Rehshape planes to share dims with the word vecs.

    def hash(self, word_vector):
        bit_representation = []
        for index, plane in enumerate(self.hyperplanes):
            projection = np.dot(plane, np.array(word_vector))
            bit_representation.append(''.join(['1' if point > 0 else '0' for point in projection]))
        return bit_representation

    def humming_distance(self, word_vector_1, word_vector_2):
        bit_representation_1 = self.hash(word_vector_1)
        bit_representation_2 = self.hash(word_vector_2)
        return sum([self.hamming_distance_string(bit_1, bit_2) for bit_1, bit_2 in zip(bit_representation_1, bit_representation_2)]) / float(self._cols_per_bucket)

    @staticmethod
    def cosine_dist(arr1, arr2):
        return spatial.distance.cosine(arr1, arr2)

    @staticmethod
    def find_nearby_bit_strings(bit_str, dist):
        # Currently the method only works for distace 0 and 1
        assert 0 <= dist <= 1

        # Find all bit strings where at most dist bits are changed
        nearby_str = [bit_str]
        if dist == 0:
            return nearby_str
        for i in range(len(bit_str)):
            nearby = list(bit_str)
            if bit_str[i] == '0':
                nearby[i] = '1'
            else:
                nearby[i] = '0'
            nearby_str.append(''.join(nearby))
        return nearby_str
