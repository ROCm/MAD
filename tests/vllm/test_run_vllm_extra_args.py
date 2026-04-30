"""Tests for extra_args quoting in run_vllm.py."""
import sys
import os
import shlex

import pytest


def build_extra_args_str_old(extra_args: dict) -> str:
    """Replicates the OLD selective-quoting logic from run_vllm.py (pre-fix)."""
    extra_args_str = ""
    for k, v in extra_args.items():
        if isinstance(v, bool):
            extra_args_str += f" {k}"
        else:
            s = str(v)
            st = s.strip()
            if (
                k == "--limit-mm-per-prompt"
                or (st[:1] in "{[")
                or any(ch.isspace() for ch in s)
            ):
                extra_args_str += f" {k} {shlex.quote(s)}"
            else:
                extra_args_str += f" {k} {s}"
    return extra_args_str.strip()


def build_extra_args_str_new(extra_args: dict) -> str:
    """Replicates the NEW universal-quoting logic (post-fix)."""
    extra_args_str = ""
    for k, v in extra_args.items():
        if isinstance(v, bool):
            extra_args_str += f" {k}"
        else:
            extra_args_str += f" {k} {shlex.quote(str(v))}"
    return extra_args_str.strip()


# --- Tests that FAIL with the old logic, PASS with the new ---

def test_shell_metachar_no_space_is_quoted_by_new():
    """Values with shell metacharacters but no spaces are NOT quoted by old code.

    The old code only quotes when there's whitespace, a JSON-like prefix, or the
    --limit-mm-per-prompt key. A value like 'foo;bar' (no space) slips through
    unquoted, allowing shell injection. The new code always quotes.
    """
    args = {"--some-arg": "foo;bar"}
    old = build_extra_args_str_old(args)
    new = build_extra_args_str_new(args)
    # Old code: no whitespace -> not quoted, semicolon is a live shell metachar
    assert old == "--some-arg foo;bar", f"unexpected old output: {old!r}"
    # New code: shlex.quote wraps the value in single quotes
    assert new == "--some-arg 'foo;bar'", f"unexpected new output: {new!r}"
    assert old != new


def test_plain_string_with_metachar_is_unquoted_by_old():
    """Old code leaves plain strings with $ unquoted (variable expansion risk)."""
    args = {"--trust-remote-code": "yes$HOME"}
    old = build_extra_args_str_old(args)
    new = build_extra_args_str_new(args)
    # Old code: no whitespace, no JSON prefix -> raw string passed to shell
    assert old == "--trust-remote-code yes$HOME", f"unexpected old output: {old!r}"
    # New code: always quoted
    assert new == "--trust-remote-code 'yes$HOME'", f"unexpected new output: {new!r}"


# --- Tests that PASS with BOTH old and new logic ---

def test_json_value_is_quoted():
    args = {"--limit-mm-per-prompt": '{"image":0,"audio":0}'}
    result = build_extra_args_str_new(args)
    assert result == """--limit-mm-per-prompt '{"image":0,"audio":0}'"""


def test_bool_flag_has_no_value():
    args = {"--async-scheduling": True}
    result = build_extra_args_str_new(args)
    assert result == "--async-scheduling"


def test_string_with_space_is_quoted():
    args = {"--served-model-name": "my model"}
    result = build_extra_args_str_new(args)
    assert result == "--served-model-name 'my model'"


def test_plain_safe_scalar_passthrough():
    """shlex.quote does not add quotes to safe alphanumeric values."""
    args = {"--max-model-len": 32768}
    result = build_extra_args_str_new(args)
    # shlex.quote('32768') == '32768' (no shell quoting needed for pure digits)
    assert result == "--max-model-len 32768"
