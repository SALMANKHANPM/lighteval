# MIT License
#
# Copyright (c) 2024 The HuggingFace Team
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import pytest

from lighteval.main_vllm import _check_vllm_numpy_compatibility


def test_check_vllm_numpy_compatibility_raises_on_numba_with_new_numpy():
    with pytest.raises(RuntimeError, match='numpy==2.4.2'):
        _check_vllm_numpy_compatibility(numpy_version='2.4.2', numba_installed=True)


def test_check_vllm_numpy_compatibility_accepts_old_numpy_with_numba():
    _check_vllm_numpy_compatibility(numpy_version='2.2.6', numba_installed=True)


def test_check_vllm_numpy_compatibility_accepts_new_numpy_without_numba():
    _check_vllm_numpy_compatibility(numpy_version='2.4.2', numba_installed=False)
