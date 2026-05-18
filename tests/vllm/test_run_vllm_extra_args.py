import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "vllm"))
from run_vllm import build_extra_args_str


def test_simple_string_value():
    result = build_extra_args_str({"--attention-backend": "TRITON_ATTN"})
    assert result == "--attention-backend TRITON_ATTN"


def test_json_value_is_quoted():
    result = build_extra_args_str({"--limit-mm-per-prompt": '{"image":0,"audio":0}'})
    assert "'" in result or "\\" in result
    assert "--limit-mm-per-prompt" in result


def test_bool_true_emits_flag():
    result = build_extra_args_str({"--enable-prefix-caching": True})
    assert result == "--enable-prefix-caching"


def test_bool_false_skips_flag():
    result = build_extra_args_str({"--enable-prefix-caching": False})
    assert result == ""


def test_numeric_value():
    result = build_extra_args_str({"--max-model-len": 32768})
    assert result == "--max-model-len 32768"


def test_mixed_args():
    args = {
        "--attention-backend": "TRITON_ATTN",
        "--enable-prefix-caching": True,
        "--disable-log-stats": False,
        "--max-model-len": 32768,
        "--limit-mm-per-prompt": '{"image":0,"audio":0}',
    }
    result = build_extra_args_str(args)
    assert "--attention-backend TRITON_ATTN" in result
    assert "--enable-prefix-caching" in result
    assert "--disable-log-stats" not in result
    assert "--max-model-len 32768" in result
    assert "--limit-mm-per-prompt" in result


def test_empty_args():
    assert build_extra_args_str({}) == ""


def test_value_with_spaces_is_quoted():
    result = build_extra_args_str({"--chat-template": "path with spaces/template.jinja"})
    assert "'" in result or "\\" in result


def test_shell_metacharacters_are_quoted():
    result = build_extra_args_str({"--some-arg": "value;rm -rf /"})
    assert "'" in result or "\\" in result
