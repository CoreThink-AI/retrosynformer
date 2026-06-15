"""Run all inline doctests via pytest.

Each parametrized case imports one module and asserts that every
``>>>`` example in its docstrings produces the expected output.
Add new module names to DOCTEST_MODULES when you add doctests elsewhere.
"""
import doctest
import importlib

import pytest

DOCTEST_MODULES = [
    "retrosynformer.utils.utils",
    "retrosynformer.utils.reward_functions",
    "retrosynformer.data",
    "retrosynformer.environment",
    "retrosynformer.structured_dropout",
    "retrosynformer.monitor",
    # scripts/ modules (imported via sys.path, not as packages)
]


@pytest.mark.parametrize("modname", DOCTEST_MODULES)
def test_module_doctests(modname):
    mod = importlib.import_module(modname)
    results = doctest.testmod(mod, verbose=False)
    assert results.failed == 0, (
        f"{results.failed} of {results.attempted} doctest(s) failed in {modname}"
    )
