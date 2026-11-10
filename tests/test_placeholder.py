"""Placeholder test to verify test harness and package discovery."""

import raglab


def test_package_import() -> None:
    """Verify raglab package can be imported."""
    assert raglab.__version__ == "0.1.0"
