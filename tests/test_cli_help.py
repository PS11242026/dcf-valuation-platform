"""CLI surface smoke: every product app resolves and the new valuation commands appear.

Uses Typer's in-process runner (no subprocess). COLUMNS is widened so Rich does not wrap
command names / option flags mid-token.
"""
from __future__ import annotations

import os

from typer.testing import CliRunner

from dcf_project.cli import app

runner = CliRunner()


def _invoke(args):
    env = {**os.environ, "COLUMNS": "200"}
    return runner.invoke(app, args, env=env)


def test_root_help_lists_products():
    r = _invoke(["--help"])
    assert r.exit_code == 0
    for cmd in ("valuation", "portfolio", "manage", "research", "export"):
        assert cmd in r.output


def test_valuation_help_lists_dcf_and_compare():
    r = _invoke(["valuation", "--help"])
    assert r.exit_code == 0
    for cmd in ("dcf", "canonical", "compare", "generic", "screen"):
        assert cmd in r.output


def test_dcf_help_is_flagship():
    r = _invoke(["valuation", "dcf", "--help"])
    assert r.exit_code == 0
    assert "Fade" in r.output and "mc-draws" in r.output


def test_compare_help():
    r = _invoke(["valuation", "compare", "--help"])
    assert r.exit_code == 0
    assert "mc-draws" in r.output
