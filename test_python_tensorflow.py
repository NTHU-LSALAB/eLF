import unittest

import tensorflow as tf


class TensorFlowTest(unittest.TestCase):
    def setUp(self):
        self.module = tf.load_op_library('./lib_elf_tensorflow.so')

    def test_op_exist(self):
        self.module.zero_out


if __name__ == '__main__':
    unittest.main()
