import unittest

import _elf


class CoreTest(unittest.TestCase):
    def test_create_controller(self):
        ctrl = _elf.create_controller()


if __name__ == '__main__':
    unittest.main()
