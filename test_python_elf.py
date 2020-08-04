import sys
import unittest

import elf


class ElfTest(unittest.TestCase):
    def setUp(self):
        self.ctrl = elf.Controller('127.0.0.1:')

    def tearDown(self):
        self.ctrl.stop()

    def test_worker_basic(self):
        worker = elf.Worker('127.0.0.1:{}'.format(self.ctrl.listening_port))

        worker.add_global_variable('gv')
        worker.add_weight_variable('wv')

        worker.commit_and_join()

        sg = worker.shard_generator()
        self.assertIsInstance(next(sg), int)

    def test_worker_basic2(self):
        worker = elf.Worker('127.0.0.1:{}'.format(self.ctrl.listening_port))

        worker.commit_and_join()

        sg = worker.shard_generator(range(20), batch_size=4)
        shard = next(sg)
        self.assertIsInstance(shard, list)
        self.assertEqual(len(shard), 4)

if __name__ == '__main__':
    unittest.main()
