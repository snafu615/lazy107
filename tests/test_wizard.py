"""Tests for `lazy107 everything` (wizard orchestration; stages faked)."""

from pathlib import Path

import pytest

from lazy107.cli import main


def _make_interactive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lazy107.cli._stdin_interactive", lambda: True)


def _never_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(_prompt: str) -> str:
        raise AssertionError("everything should not prompt here")

    monkeypatch.setattr("builtins.input", boom)


def _stub_submit(monkeypatch: pytest.MonkeyPatch, calls: list) -> None:
    monkeypatch.setattr("lazy107.cli.cmd_submit", lambda root, args: (calls.append(args), 0)[1])


def _stub_env(monkeypatch: pytest.MonkeyPatch, calls: list) -> None:
    monkeypatch.setattr("lazy107.cli.cmd_env", lambda root, args: (calls.append(args), 0)[1])


def _train_only(tmp_path: Path) -> None:
    (tmp_path / "train.py").write_text("print('x')\n", encoding="utf-8")


def test_everything_full_flow_picks_entry_reuses_env_and_submits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    _train_only(tmp_path)
    (tmp_path / "other.py").write_text("print('y')\n", encoding="utf-8")
    _make_interactive(monkeypatch)
    monkeypatch.setattr("lazy107.cli._prompt_reuse_env", lambda root, plan: 0)
    submits: list = []
    _stub_submit(monkeypatch, submits)
    prompts: list[str] = []

    def fake_input(prompt: str) -> str:
        prompts.append(prompt)
        if prompt.startswith("choose entry"):
            return "1"
        if "pick a preset" in prompt:
            return ""  # keep the current values
        if "number of GPUs" in prompt or "CPUs per task" in prompt:
            return ""  # Enter keeps the current value
        if "memory" in prompt or "time limit" in prompt:
            return ""  # Enter keeps the current value
        if "preview" in prompt:
            return "n"
        raise AssertionError(f"unexpected prompt: {prompt}")

    monkeypatch.setattr("builtins.input", fake_input)

    assert main(["everything"]) == 0
    captured = capsys.readouterr()
    assert "pinned entry = 'train.py'" in captured.out
    assert any(p.startswith("choose entry") for p in prompts)
    assert submits and submits[0].entry == "train.py" and submits[0].yes is False
    assert 'entry = "train.py"' in (tmp_path / "107.toml").read_text(encoding="utf-8")


def test_everything_installs_fresh_env_when_reuse_declined(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _train_only(tmp_path)
    _make_interactive(monkeypatch)
    monkeypatch.setattr("lazy107.cli._prompt_reuse_env", lambda root, plan: None)
    env_calls: list = []
    _stub_env(monkeypatch, env_calls)
    submits: list = []
    _stub_submit(monkeypatch, submits)
    prompts: list[str] = []

    def fake_input(prompt: str) -> str:
        prompts.append(prompt)
        if "install a fresh" in prompt:
            return "y"
        if "pick a preset" in prompt:
            return ""
        if "number of GPUs" in prompt or "CPUs per task" in prompt:
            return ""
        if "memory" in prompt or "time limit" in prompt:
            return ""
        if "preview" in prompt:
            return "n"
        raise AssertionError(f"unexpected prompt: {prompt}")

    monkeypatch.setattr("builtins.input", fake_input)

    assert main(["everything"]) == 0
    assert env_calls and env_calls[0].yes is True and env_calls[0].name is None
    assert submits
    assert any(p.startswith("no reusable env picked") for p in prompts)


def test_everything_custom_editor_overrides_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    _train_only(tmp_path)
    _make_interactive(monkeypatch)
    monkeypatch.setattr("lazy107.cli._prompt_reuse_env", lambda root, plan: 0)
    submits: list = []
    _stub_submit(monkeypatch, submits)

    def fake_input(prompt: str) -> str:
        if "pick a preset" in prompt:
            return "c"
        if "number of GPUs" in prompt:
            return "4"
        if "CPUs per task" in prompt:
            return ""
        if "memory" in prompt or "time limit" in prompt:
            return ""
        if "preview" in prompt:
            return "n"
        raise AssertionError(f"unexpected prompt: {prompt}")

    monkeypatch.setattr("builtins.input", fake_input)

    assert main(["everything"]) == 0
    assert "slurm resources" in capsys.readouterr().err
    assert "gpu = 4" in (tmp_path / "107.toml").read_text(encoding="utf-8")
    assert submits


def test_everything_custom_editor_reprompts_on_non_int_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    _train_only(tmp_path)
    _make_interactive(monkeypatch)
    monkeypatch.setattr("lazy107.cli._prompt_reuse_env", lambda root, plan: 0)
    submits: list = []
    _stub_submit(monkeypatch, submits)
    gpu_answers = iter(["abc", "2"])

    def fake_input(prompt: str) -> str:
        if "pick a preset" in prompt:
            return "c"
        if "number of GPUs" in prompt:
            return next(gpu_answers)
        if "CPUs per task" in prompt:
            return ""
        if "memory" in prompt or "time limit" in prompt:
            return ""
        if "preview" in prompt:
            return "n"
        raise AssertionError(f"unexpected prompt: {prompt}")

    monkeypatch.setattr("builtins.input", fake_input)

    assert main(["everything"]) == 0
    captured = capsys.readouterr()
    assert "gpu must be a whole number (got 'abc')" in captured.err
    assert "gpu = 2" in (tmp_path / "107.toml").read_text(encoding="utf-8")
    assert submits


def test_everything_preset_wires_all_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _train_only(tmp_path)
    _make_interactive(monkeypatch)
    monkeypatch.setattr("lazy107.cli._prompt_reuse_env", lambda root, plan: 0)
    submits: list = []
    _stub_submit(monkeypatch, submits)

    def fake_input(prompt: str) -> str:
        if "pick a preset" in prompt:
            return "4"  # gpu-heavy
        if "preview" in prompt:
            return "n"
        raise AssertionError(f"unexpected prompt: {prompt}")

    monkeypatch.setattr("builtins.input", fake_input)

    assert main(["everything"]) == 0
    text = (tmp_path / "107.toml").read_text(encoding="utf-8")
    assert "gpu = 1" in text
    assert "cpus = 8" in text
    assert 'mem = "64G"' in text
    assert 'time = "12:00:00"' in text
    assert submits


def test_everything_preset_reprompts_on_invalid_pick(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    _train_only(tmp_path)
    _make_interactive(monkeypatch)
    monkeypatch.setattr("lazy107.cli._prompt_reuse_env", lambda root, plan: 0)
    submits: list = []
    _stub_submit(monkeypatch, submits)
    picks = iter(["9", "2"])  # invalid, then cpu-heavy

    def fake_input(prompt: str) -> str:
        if "pick a preset" in prompt:
            return next(picks)
        if "preview" in prompt:
            return "n"
        raise AssertionError(f"unexpected prompt: {prompt}")

    monkeypatch.setattr("builtins.input", fake_input)

    assert main(["everything"]) == 0
    captured = capsys.readouterr()
    assert "pick a number 1-5 or c" in captured.err
    text = (tmp_path / "107.toml").read_text(encoding="utf-8")
    assert "cpus = 8" in text
    assert 'mem = "32G"' in text
    assert 'time = "8:00:00"' in text
    assert "gpu =" not in text  # preset gpu matches the plan's current value
    assert submits


def test_everything_preset_enter_keeps_current_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _train_only(tmp_path)
    _make_interactive(monkeypatch)
    monkeypatch.setattr("lazy107.cli._prompt_reuse_env", lambda root, plan: 0)
    submits: list = []
    _stub_submit(monkeypatch, submits)

    def fake_input(prompt: str) -> str:
        if "pick a preset" in prompt:
            return ""
        if "number of GPUs" in prompt or "CPUs per task" in prompt:
            return ""
        if "memory" in prompt or "time limit" in prompt:
            return ""
        if "preview" in prompt:
            return "n"
        raise AssertionError(f"unexpected prompt: {prompt}")

    monkeypatch.setattr("builtins.input", fake_input)

    assert main(["everything"]) == 0
    assert not (tmp_path / "107.toml").exists()
    assert submits


def test_everything_preset_equal_to_current_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _train_only(tmp_path)
    _make_interactive(monkeypatch)
    monkeypatch.setattr("lazy107.cli._prompt_reuse_env", lambda root, plan: 0)
    submits: list = []
    _stub_submit(monkeypatch, submits)

    def fake_input(prompt: str) -> str:
        if "pick a preset" in prompt:
            return "1"  # cpu-light == the plan's CPU defaults
        if "preview" in prompt:
            return "n"
        raise AssertionError(f"unexpected prompt: {prompt}")

    monkeypatch.setattr("builtins.input", fake_input)

    assert main(["everything"]) == 0
    assert not (tmp_path / "107.toml").exists()
    assert submits


def test_everything_yes_runs_noninteractively(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _train_only(tmp_path)
    _make_interactive(monkeypatch)  # tty present, but --yes must still skip prompts
    _never_prompt(monkeypatch)
    env_calls: list = []
    _stub_env(monkeypatch, env_calls)
    submits: list = []
    _stub_submit(monkeypatch, submits)

    assert main(["everything", "--yes"]) == 0
    assert env_calls and env_calls[0].yes is True
    assert submits and submits[0].yes is True


def test_everything_aborts_on_invalid_edits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    _train_only(tmp_path)
    _make_interactive(monkeypatch)
    monkeypatch.setattr("lazy107.cli._prompt_reuse_env", lambda root, plan: 0)
    submits: list = []
    _stub_submit(monkeypatch, submits)

    def fake_input(prompt: str) -> str:
        if "pick a preset" in prompt:
            return "c"
        if "number of GPUs" in prompt:
            return ""
        if "CPUs per task" in prompt:
            return "0"
        if "memory" in prompt or "time limit" in prompt:
            return ""
        raise AssertionError(f"unexpected prompt: {prompt}")

    monkeypatch.setattr("builtins.input", fake_input)

    assert main(["everything"]) == 1
    assert not submits
    assert "cpus: must be >= 1" in capsys.readouterr().err


def test_everything_declines_preview(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _train_only(tmp_path)
    _make_interactive(monkeypatch)
    monkeypatch.setattr("lazy107.cli._prompt_reuse_env", lambda root, plan: 0)
    renders: list = []
    monkeypatch.setattr(
        "lazy107.cli.cmd_render", lambda root, args: (renders.append(args), 0)[1]
    )
    submits: list = []
    _stub_submit(monkeypatch, submits)

    def fake_input(prompt: str) -> str:
        if "pick a preset" in prompt:
            return ""
        if "number of GPUs" in prompt or "CPUs per task" in prompt:
            return ""
        if "memory" in prompt or "time limit" in prompt:
            return ""
        if "preview" in prompt:
            return "n"
        raise AssertionError(f"unexpected prompt: {prompt}")

    monkeypatch.setattr("builtins.input", fake_input)

    assert main(["everything"]) == 0
    assert not renders
    assert submits


def test_everything_skips_env_when_conda_env_pinned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    _train_only(tmp_path)
    (tmp_path / "107.toml").write_text(
        'entry = "train.py"\nconda_env = "demo"\n', encoding="utf-8"
    )
    _make_interactive(monkeypatch)
    reuse: list = []
    monkeypatch.setattr(
        "lazy107.cli._prompt_reuse_env", lambda root, plan: (reuse.append(1), 0)[1]
    )
    env_calls: list = []
    _stub_env(monkeypatch, env_calls)
    submits: list = []
    _stub_submit(monkeypatch, submits)

    def fake_input(prompt: str) -> str:
        if "pick a preset" in prompt:
            return ""
        if "number of GPUs" in prompt or "CPUs per task" in prompt:
            return ""
        if "memory" in prompt or "time limit" in prompt:
            return ""
        if "preview" in prompt:
            return "n"
        raise AssertionError(f"unexpected prompt: {prompt}")

    monkeypatch.setattr("builtins.input", fake_input)

    assert main(["everything"]) == 0
    assert not reuse and not env_calls
    assert "skipping env setup" in capsys.readouterr().out
    assert submits


def test_cli_help_lists_everything(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["--help"])
    assert "everything" in capsys.readouterr().out
