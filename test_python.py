import unittest

import _elf


class BindingTest(unittest.TestCase):
    """just testing the binding works"""

    def test_create_controller(self):
        ctrl = _elf.create_controller()

        ctrl.stop()

    def test_export_controller(self):
        ctrl = _elf.create_controller()
        ectrl = _elf.export_controller(ctrl, '127.0.0.1:')
        cctrl = _elf.connect_controller('127.0.0.1:{}'.format(ectrl.listening_port()))

        ectrl.stop()
        ctrl.stop()

    def test_worker(self):
        ctrl = _elf.create_controller()
        worker = _elf.Worker(ctrl)

        broadcast = worker.add_global_variable('var1')
        allreduce = worker.add_weight_variable('var1')

        worker.commit_and_join()

        self.assertEqual(worker.begin_batch()[:2], (True, False))
        self.assertEqual(broadcast.get_handle()[:2], '0x')
        self.assertEqual(allreduce.get_handle()[:2], '0x')

        worker.leave()

        ctrl.stop()


if __name__ == '__main__':
    unittest.main()
