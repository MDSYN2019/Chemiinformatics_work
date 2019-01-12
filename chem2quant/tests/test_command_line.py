from unittest import TestCase

from SYN_CI.command_line import main

class TestCmd(TestCase):
    def test_basic(self):
        main()
        
