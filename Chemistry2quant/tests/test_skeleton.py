#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from chemistry2quant.skeleton import fib

__author__ = "Sang Young Noh"
__copyright__ = "Sang Young Noh"
__license__ = "mit"


def test_fib():
    assert fib(1) == 1
    assert fib(2) == 1
    assert fib(7) == 13
    with pytest.raises(AssertionError):
        fib(-10)
