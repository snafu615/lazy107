"""Tests for the lazy107 CLI (all subcommands; I/O faked)."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from lazy107 import __version__
from lazy107.cli import main


def test_cli_version_flag_prints_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["--version"])
    assert __version__ in capsys.readouterr().out


def test_cli_help_lists_all_commands(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    for sub in ("init", "plan", "render", "env", "submit", "watch", "status", "logs", "debug", "transfer", "discover", "config", "check"):
        assert sub in out


def test_cli_every_subcommand_help_renders(capsys: pytest.CaptureFixture[str]) -> None:
    # Regression: help strings containing literal % (e.g. "1-5%2") crash
    # argparse's `help % params` expansion with a ValueError, so every
    # subcommand's --help must print cleanly.
    for sub in ("init", "plan", "render", "env", "submit", "watch", "status", "logs", "debug", "transfer", "discover", "config", "check"):
        with pytest.raises(SystemExit) as exc_info:
            main([sub, "--help"])
        assert exc_info.value.code == 0, f"{sub} --help crashed"
        assert "usage:" in capsys.readouterr().out


def test_cli_init_scaffolds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["init", "foo"]) == 0
    assert (tmp_path / "foo" / "src" / "train.py").exists()
    assert "name: foo" in (tmp_path / "foo" / "environment.yml").read_text()


def test_cli_init_twice_fails_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["init", "foo"]) == 0
    assert main(["init", "foo"]) == 1
    assert "non-empty" in capsys.readouterr().err


def test_cli_init_records_history_in_new_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["init", "foo"]) == 0
    history = (tmp_path / "foo" / "notes" / "history.md").read_text(encoding="utf-8")
    assert "lazy107 init foo (exit 0)" in history


def test_cli_records_history_inside_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["init", "foo"]) == 0
    monkeypatch.chdir(tmp_path / "foo")
    assert main(["plan"]) == 0
    history = (tmp_path / "foo" / "notes" / "history.md").read_text(encoding="utf-8")
    assert "lazy107 plan (exit 0)" in history


def test_cli_history_gated_outside_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "train.py").write_text("print('x')\n", encoding="utf-8")
    assert main(["plan"]) == 0
    assert not (tmp_path / "notes").exists()


def test_cli_records_history_with_implicit_argv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Console-script path: main() called without argv must still record
    # history (it reads sys.argv[1:]).
    monkeypatch.chdir(tmp_path)
    assert main(["init", "foo"]) == 0
    monkeypatch.chdir(tmp_path / "foo")
    monkeypatch.setattr("sys.argv", ["lazy107", "plan"])
    assert main() == 0
    history = (tmp_path / "foo" / "notes" / "history.md").read_text(encoding="utf-8")
    assert "lazy107 plan (exit 0)" in history


def test_cli_check_shows_command_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["init", "foo"]) == 0
    monkeypatch.chdir(tmp_path / "foo")
    assert main(["plan"]) == 0
    monkeypatch.setattr("lazy107.cli.env_mod.find_conda", lambda: None)
    assert main(["check"]) == 0
    out = capsys.readouterr().out
    assert "commands: 2 recorded" in out
    assert "latest: " in out


def test_cli_plan_prints_resolved_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "train.py").write_text("print('x')\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "sandbox"\ndependencies = ["torch"]\n',
        encoding="utf-8",
    )
    assert main(["plan"]) == 0
    out = capsys.readouterr().out
    assert "entry=train.py" in out
    assert "gpu=1" in out
    assert "partition=Students" in out


def test_cli_plan_no_entry_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["plan"]) == 1


def _make_interactive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lazy107.cli._stdin_interactive", lambda: True)


def _never_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(_prompt: str) -> str:
        raise AssertionError("plan should not prompt here")

    monkeypatch.setattr("builtins.input", boom)


def test_cli_plan_prompts_to_choose_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "train.py").write_text("print('x')\n", encoding="utf-8")
    (tmp_path / "other.py").write_text("print('y')\n", encoding="utf-8")
    _make_interactive(monkeypatch)
    monkeypatch.setattr("builtins.input", lambda prompt: "2")
    assert main(["plan"]) == 0
    captured = capsys.readouterr()
    assert "detected entry files:" in captured.err
    assert "1) train.py" in captured.err
    assert "2) other.py" in captured.err
    assert "entry=other.py" in captured.out
    assert "pinned entry" in captured.out
    assert 'entry = "other.py"' in (tmp_path / "107.toml").read_text(encoding="utf-8")


def test_cli_plan_prompt_default_accepts_recommendation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "train.py").write_text("print('x')\n", encoding="utf-8")
    (tmp_path / "other.py").write_text("print('y')\n", encoding="utf-8")
    _make_interactive(monkeypatch)
    monkeypatch.setattr("builtins.input", lambda prompt: "")
    assert main(["plan"]) == 0
    captured = capsys.readouterr()
    assert "<- default" in captured.err
    assert "entry=train.py" in captured.out
    assert 'entry = "train.py"' in (tmp_path / "107.toml").read_text(encoding="utf-8")


def test_cli_plan_prompt_accepts_typed_filename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "train.py").write_text("print('x')\n", encoding="utf-8")
    (tmp_path / "run.py").write_text("print('y')\n", encoding="utf-8")
    _make_interactive(monkeypatch)
    monkeypatch.setattr("builtins.input", lambda prompt: "run.py")
    assert main(["plan"]) == 0
    assert "entry=run.py" in capsys.readouterr().out
    assert 'entry = "run.py"' in (tmp_path / "107.toml").read_text(encoding="utf-8")


def test_cli_plan_prompt_reprompts_after_invalid_number(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "train.py").write_text("print('x')\n", encoding="utf-8")
    (tmp_path / "other.py").write_text("print('y')\n", encoding="utf-8")
    _make_interactive(monkeypatch)
    answers = iter(["99", "1"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    assert main(["plan"]) == 0
    captured = capsys.readouterr()
    assert "no entry numbered 99" in captured.err
    assert "entry=train.py" in captured.out


def test_cli_plan_no_prompt_when_single_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "train.py").write_text("print('x')\n", encoding="utf-8")
    _make_interactive(monkeypatch)
    _never_prompt(monkeypatch)
    assert main(["plan"]) == 0
    assert "entry=train.py" in capsys.readouterr().out
    assert not (tmp_path / "107.toml").exists()


def test_cli_plan_no_prompt_when_entry_pinned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "train.py").write_text("print('x')\n", encoding="utf-8")
    (tmp_path / "main.py").write_text("print('y')\n", encoding="utf-8")
    (tmp_path / "107.toml").write_text('entry = "main.py"\n', encoding="utf-8")
    _make_interactive(monkeypatch)
    _never_prompt(monkeypatch)
    assert main(["plan"]) == 0
    captured = capsys.readouterr()
    assert "entry=main.py" in captured.out  # the pin wins over detection
    assert (tmp_path / "107.toml").read_text(encoding="utf-8") == 'entry = "main.py"\n'


def test_cli_plan_no_prompt_when_stdin_not_a_tty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "train.py").write_text("print('x')\n", encoding="utf-8")
    (tmp_path / "other.py").write_text("print('y')\n", encoding="utf-8")
    monkeypatch.setattr("lazy107.cli._stdin_interactive", lambda: False)
    _never_prompt(monkeypatch)
    assert main(["plan"]) == 0
    assert "entry=train.py" in capsys.readouterr().out
    assert not (tmp_path / "107.toml").exists()


def test_cli_plan_prints_edit_hint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "train.py").write_text("print('x')\n", encoding="utf-8")
    assert main(["plan"]) == 0
    out = capsys.readouterr().out
    assert "107.toml" in out
    assert "Web Shell GUI" in out


def test_cli_render_dry_run_prints_without_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "train.py").write_text("print('x')\n", encoding="utf-8")
    assert main(["render", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "#SBATCH --job-name=train" in out
    assert "python train.py" in out
    assert not (tmp_path / "scripts" / "train.sbatch").exists()


def test_cli_render_writes_script(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "train.py").write_text("print('x')\n", encoding="utf-8")
    assert main(["render"]) == 0
    assert (tmp_path / "scripts" / "train.sbatch").exists()


def test_cli_render_missing_entry_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["render", "--entry", "nope.py"]) == 1


def test_cli_render_ddp_project_uses_torchrun(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "train.py").write_text("import torch.distributed as dist\n", encoding="utf-8")
    (tmp_path / "107.toml").write_text('gpu = 2\n', encoding="utf-8")
    assert main(["render", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "torchrun --standalone --nproc_per_node=$SLURM_GPUS_ON_NODE train.py" in out
    assert "python train.py" not in out


def test_cli_render_command_override_from_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "train.py").write_text("print('x')\n", encoding="utf-8")
    (tmp_path / "107.toml").write_text('command = "python -m trainer.main --config cfg.yaml"\n', encoding="utf-8")
    assert main(["render", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "python -m trainer.main --config cfg.yaml" in out
    assert "python train.py" not in out


def test_cli_render_array_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "train.py").write_text("print('x')\n", encoding="utf-8")
    assert main(["render", "--dry-run", "--array", "1-5%2"]) == 0
    out = capsys.readouterr().out
    assert "#SBATCH --array=1-5%2" in out
    assert "#SBATCH --output=logs/%x_%A_%a.out" in out


def test_cli_render_rejects_bad_array(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "train.py").write_text("print('x')\n", encoding="utf-8")
    assert main(["render", "--dry-run", "--array", "1-5; evil"]) == 1


def test_cli_submit_multi_node_ddp_requires_ntasks_match(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "train.py").write_text("import torch.distributed as dist\n", encoding="utf-8")
    (tmp_path / "107.toml").write_text('gpu = 4\nnodes = 2\n', encoding="utf-8")
    monkeypatch.setattr("lazy107.cli.env_mod.find_conda", lambda: None)
    assert main(["submit", "--dry-run", "--skip-check", "--yes"]) == 1


def test_cli_submit_warns_when_env_exists_but_unwired(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "train.py").write_text("print('x')\n", encoding="utf-8")
    monkeypatch.setattr(
        "lazy107.cli.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    monkeypatch.setattr("lazy107.cli.env_mod.find_conda", lambda: "/usr/bin/conda")
    monkeypatch.setattr("lazy107.cli.env_mod.env_exists", lambda conda, name: True)

    assert main(["submit", "--dry-run", "--skip-check", "--yes"]) == 0
    captured = capsys.readouterr()
    assert "exists but conda_env is unset" in captured.err
    assert "#SBATCH --job-name=train" in captured.out


def test_cli_submit_dry_run_never_submits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "train.py").write_text("print('x')\n", encoding="utf-8")
    monkeypatch.setattr("lazy107.cli.env_mod.find_conda", lambda: None)  # keep the unwired-env probe inert

    calls: list = []

    def fake_run(*args, **kwargs):
        calls.append(args)
        return SimpleNamespace(stdout="Submitted batch job 42\n", stderr="", returncode=0)

    monkeypatch.setattr("lazy107.cli.subprocess.run", fake_run)
    assert main(["submit", "--dry-run", "--skip-check", "--yes"]) == 0
    assert "#SBATCH --job-name=train" in capsys.readouterr().out
    assert calls == []  # sinfo and sbatch both skipped in dry-run
    assert not (tmp_path / "scripts" / "train.sbatch").exists()


def test_cli_submit_full_flow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "train.py").write_text("print('x')\n", encoding="utf-8")

    def fake_run(*args, **kwargs):
        return SimpleNamespace(stdout="Submitted batch job 42\n", stderr="", returncode=0)

    monkeypatch.setattr("lazy107.cli.subprocess.run", fake_run)
    monkeypatch.setattr("builtins.input", lambda prompt: "y")

    assert main(["submit", "--skip-check"]) == 0
    out = capsys.readouterr().out
    assert "submitted job 42" in out
    assert "lazy107 watch 42" in out
    runs = (tmp_path / "notes" / "runs.md").read_text()
    assert "## Job 42" in runs
    assert "train.py" in runs


def test_cli_submit_declines_without_yes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "train.py").write_text("print('x')\n", encoding="utf-8")

    monkeypatch.setattr("lazy107.cli.subprocess.run", lambda *a, **k: SimpleNamespace(stdout="", stderr="", returncode=0))
    monkeypatch.setattr("builtins.input", lambda prompt: "n")

    assert main(["submit", "--skip-check"]) == 1  # aborted
    assert not (tmp_path / "scripts" / "train.sbatch").exists()
    assert not (tmp_path / "notes" / "runs.md").exists()


def test_cli_submit_aborts_on_stdin_eof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "train.py").write_text("print('x')\n", encoding="utf-8")

    monkeypatch.setattr("lazy107.cli.subprocess.run", lambda *a, **k: SimpleNamespace(stdout="", stderr="", returncode=0))

    def eof(_prompt: str) -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", eof)

    assert main(["submit", "--skip-check"]) == 1  # aborted, no traceback
    assert "--yes" in capsys.readouterr().err
    assert not (tmp_path / "scripts" / "train.sbatch").exists()


def test_cli_env_dry_run_prints_commands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("lazy107.cli.env_mod.find_conda", lambda: "/usr/bin/conda")
    monkeypatch.setattr("lazy107.cli.env_mod.env_exists", lambda conda, name: False)

    assert main(["env", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "module load miniconda/py312" in out
    assert "conda create -y" in out


def test_cli_env_requires_yes_and_conda(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["env"]) == 1  # executing needs --yes
    monkeypatch.setattr("lazy107.cli.env_mod.find_conda", lambda: None)
    assert main(["env", "--yes"]) == 1  # executing needs conda
    assert main(["env", "--dry-run"]) == 0  # printing does not


def test_cli_env_autowrites_requirements(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "train.py").write_text("import torch\nimport numpy as np\nimport os\n", encoding="utf-8")
    monkeypatch.setattr("lazy107.cli.env_mod.find_conda", lambda: "/usr/bin/conda")
    monkeypatch.setattr("lazy107.cli.env_mod.env_exists", lambda conda, name: False)
    monkeypatch.setattr(
        "lazy107.cli.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    monkeypatch.setattr("lazy107.cli.env_mod.gpu_build_check", lambda conda, name: "12.6")

    assert main(["env", "--yes"]) == 0
    req = (tmp_path / "requirements.txt").read_text(encoding="utf-8")
    assert "torch" in req
    assert "numpy" in req
    out = capsys.readouterr().out
    assert "wrote" in out
    assert "pip install -r requirements.txt" in out
    assert "wired conda_env" in out
    manifest_text = (tmp_path / "107.toml").read_text(encoding="utf-8")
    assert 'conda_env = "' in manifest_text


def test_cli_env_sanitizes_explicit_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["env", "--dry-run", "--name", "my env!"]) == 0
    out = capsys.readouterr().out
    assert "conda create -y -n my_env_ python=3.12" in out
    assert "my env!" not in out


def test_cli_env_fails_when_torch_missing_after_install(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "train.py").write_text("import torch\n", encoding="utf-8")
    monkeypatch.setattr("lazy107.cli.env_mod.find_conda", lambda: "/usr/bin/conda")
    monkeypatch.setattr("lazy107.cli.env_mod.env_exists", lambda conda, name: False)
    monkeypatch.setattr(
        "lazy107.cli.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    monkeypatch.setattr("lazy107.cli.env_mod.gpu_build_check", lambda conda, name: None)

    assert main(["env", "--yes"]) == 1
    assert "not importable" in capsys.readouterr().err


def test_cli_env_dry_run_previews_wiring_without_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "train.py").write_text("import torch\n", encoding="utf-8")
    assert main(["env", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "would set conda_env" in out
    assert not (tmp_path / "107.toml").exists()


def test_cli_env_mismatched_conda_env_notes_not_wires(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "train.py").write_text("import torch\n", encoding="utf-8")
    (tmp_path / "107.toml").write_text('conda_env = "other"\n', encoding="utf-8")
    monkeypatch.setattr("lazy107.cli.env_mod.find_conda", lambda: "/usr/bin/conda")
    monkeypatch.setattr("lazy107.cli.env_mod.env_exists", lambda conda, name: False)
    monkeypatch.setattr(
        "lazy107.cli.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    monkeypatch.setattr("lazy107.cli.env_mod.gpu_build_check", lambda conda, name: "12.6")

    assert main(["env", "--yes"]) == 0
    out = capsys.readouterr().out
    assert "conda_env is already set to 'other'" in out
    assert (tmp_path / "107.toml").read_text(encoding="utf-8") == 'conda_env = "other"\n'


def test_cli_env_reuse_prompts_and_wires_validated_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "train.py").write_text("import torch\n", encoding="utf-8")
    _make_interactive(monkeypatch)
    monkeypatch.setattr("lazy107.cli.env_mod.find_conda", lambda: "/usr/bin/conda")
    monkeypatch.setattr("lazy107.cli.env_mod.list_envs", lambda conda: ["demo_env", "torch2"])
    monkeypatch.setattr("lazy107.cli.env_mod.missing_deps", lambda conda, name, deps: [])
    monkeypatch.setattr("lazy107.cli.env_mod.gpu_build_check", lambda conda, name: "12.6")
    prompts: list[str] = []

    def fake_input(prompt: str) -> str:
        prompts.append(prompt)
        return "2"

    monkeypatch.setattr("builtins.input", fake_input)

    assert main(["env"]) == 0
    captured = capsys.readouterr()
    assert "existing conda environments:" in captured.err
    assert "1) demo_env" in captured.err
    assert "2) torch2" in captured.err
    assert any(p.startswith("reuse an env") for p in prompts)
    assert "validated env 'torch2'" in captured.out
    assert "torch CUDA build 12.6" in captured.out
    assert 'conda_env = "torch2"' in (tmp_path / "107.toml").read_text(encoding="utf-8")


def test_cli_env_reuse_accepts_typed_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "train.py").write_text("print('x')\n", encoding="utf-8")
    _make_interactive(monkeypatch)
    monkeypatch.setattr("lazy107.cli.env_mod.find_conda", lambda: "/usr/bin/conda")
    monkeypatch.setattr("lazy107.cli.env_mod.list_envs", lambda conda: ["demo_env", "torch2"])
    monkeypatch.setattr("builtins.input", lambda prompt: "torch2")

    assert main(["env"]) == 0
    assert "validated env 'torch2'" in capsys.readouterr().out
    assert 'conda_env = "torch2"' in (tmp_path / "107.toml").read_text(encoding="utf-8")


def test_cli_env_reuse_reports_missing_then_declines(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "train.py").write_text("import torch\n", encoding="utf-8")
    _make_interactive(monkeypatch)
    monkeypatch.setattr("lazy107.cli.env_mod.find_conda", lambda: "/usr/bin/conda")
    monkeypatch.setattr("lazy107.cli.env_mod.list_envs", lambda conda: ["demo_env"])
    monkeypatch.setattr("lazy107.cli.env_mod.missing_deps", lambda conda, name, deps: ["torch", "numpy"])
    answers = iter(["1", ""])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))

    assert main(["env"]) == 1
    captured = capsys.readouterr()
    assert "missing 2 needed package(s): numpy, torch" in captured.err
    assert "press Enter to install a fresh env" in captured.err
    assert "use --yes to execute" in captured.err
    assert not (tmp_path / "107.toml").exists()


def test_cli_env_reuse_rejects_cpu_torch_for_gpu_project(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "train.py").write_text("import torch\n", encoding="utf-8")
    _make_interactive(monkeypatch)
    monkeypatch.setattr("lazy107.cli.env_mod.find_conda", lambda: "/usr/bin/conda")
    monkeypatch.setattr("lazy107.cli.env_mod.list_envs", lambda conda: ["demo_env"])
    monkeypatch.setattr("lazy107.cli.env_mod.missing_deps", lambda conda, name, deps: [])
    monkeypatch.setattr("lazy107.cli.env_mod.gpu_build_check", lambda conda, name: "")
    answers = iter(["1", ""])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))

    assert main(["env"]) == 1
    captured = capsys.readouterr()
    assert "cannot run this GPU job" in captured.err
    assert "use --yes to execute" in captured.err
    assert not (tmp_path / "107.toml").exists()


def test_cli_env_reuse_reprompts_after_invalid_number(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "train.py").write_text("print('x')\n", encoding="utf-8")
    _make_interactive(monkeypatch)
    monkeypatch.setattr("lazy107.cli.env_mod.find_conda", lambda: "/usr/bin/conda")
    monkeypatch.setattr("lazy107.cli.env_mod.list_envs", lambda conda: ["demo_env"])
    answers = iter(["9", ""])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))

    assert main(["env"]) == 1
    captured = capsys.readouterr()
    assert "no conda env numbered or named '9'" in captured.err
    assert "use --yes to execute" in captured.err


def test_cli_env_reuse_switches_an_existing_pin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "train.py").write_text("print('x')\n", encoding="utf-8")
    (tmp_path / "107.toml").write_text('conda_env = "old"\n', encoding="utf-8")
    _make_interactive(monkeypatch)
    monkeypatch.setattr("lazy107.cli.env_mod.find_conda", lambda: "/usr/bin/conda")
    monkeypatch.setattr("lazy107.cli.env_mod.list_envs", lambda conda: ["old", "torch2"])
    monkeypatch.setattr("builtins.input", lambda prompt: "2")

    assert main(["env"]) == 0
    assert "validated env 'torch2'" in capsys.readouterr().out
    assert (tmp_path / "107.toml").read_text(encoding="utf-8") == 'conda_env = "torch2"\n'


def test_cli_env_reuse_skips_with_explicit_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "train.py").write_text("print('x')\n", encoding="utf-8")
    _make_interactive(monkeypatch)
    _never_prompt(monkeypatch)

    assert main(["env", "--name", "custom"]) == 1
    assert "use --yes to execute" in capsys.readouterr().err


def test_cli_env_reuse_skips_when_no_envs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "train.py").write_text("print('x')\n", encoding="utf-8")
    _make_interactive(monkeypatch)
    monkeypatch.setattr("lazy107.cli.env_mod.find_conda", lambda: "/usr/bin/conda")
    monkeypatch.setattr("lazy107.cli.env_mod.list_envs", lambda conda: [])
    _never_prompt(monkeypatch)

    assert main(["env"]) == 1
    captured = capsys.readouterr()
    assert "no existing conda environments to reuse" in captured.err
    assert "use --yes to execute" in captured.err


def test_cli_env_dry_run_previews_autowrite_without_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "train.py").write_text("import torch\n", encoding="utf-8")
    assert main(["env", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "would write requirements.txt" in out
    assert not (tmp_path / "requirements.txt").exists()


def test_cli_config_init_and_show(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["config", "--init"]) == 0
    manifest = tmp_path / "107.toml"
    assert manifest.exists()
    text = manifest.read_text(encoding="utf-8")
    assert "partition" in text
    assert 'entry = ""' in text  # the editable entry slot ships in the template

    assert main(["config"]) == 0
    out = capsys.readouterr().out
    assert "manifest:" in out
    assert "present" in out
    assert "partition=Students" in out


def test_cli_status_logs_watch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["status"]) == 0
    out = capsys.readouterr().out
    assert "squeue --me" in out

    assert main(["status", "12345"]) == 0
    assert "scontrol show job 12345" in capsys.readouterr().out

    assert main(["logs", "12345"]) == 0
    out = capsys.readouterr().out
    assert "tail -f" in out
    assert "12345" in out

    assert main(["watch", "12345"]) == 0
    out = capsys.readouterr().out
    assert "squeue --me" in out
    assert "scontrol show job 12345" in out
    assert "tail -f" in out


def test_cli_watch_uses_recorded_array_from_ledger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    notes = tmp_path / "notes" / "runs.md"
    notes.parent.mkdir(parents=True)
    notes.write_text("## Job 12345\n\n- **Task array**: 1-5\n\n", encoding="utf-8")
    assert main(["watch", "12345"]) == 0
    out = capsys.readouterr().out
    assert "tail -f logs/*_12345_*.out" in out





def test_cli_transfer_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["transfer"]) == 0





def test_cli_submit_blocks_when_wired_env_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "train.py").write_text("print('x')\n", encoding="utf-8")
    (tmp_path / "107.toml").write_text('conda_env = "train"\n', encoding="utf-8")
    monkeypatch.setattr("lazy107.cli.env_mod.find_conda", lambda: "/usr/bin/conda")
    monkeypatch.setattr("lazy107.cli.env_mod.env_exists", lambda conda, name: False)

    assert main(["submit", "--yes"]) == 1
    assert "not created" in capsys.readouterr().err
    assert not (tmp_path / "scripts" / "train.sbatch").exists()


def test_cli_submit_skip_check_bypasses_env_block(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "train.py").write_text("print('x')\n", encoding="utf-8")
    (tmp_path / "107.toml").write_text('conda_env = "train"\n', encoding="utf-8")
    monkeypatch.setattr("lazy107.cli.env_mod.find_conda", lambda: "/usr/bin/conda")
    monkeypatch.setattr("lazy107.cli.env_mod.env_exists", lambda conda, name: False)
    monkeypatch.setattr(
        "lazy107.cli.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(stdout="Submitted batch job 42\n", stderr="", returncode=0),
    )

    assert main(["submit", "--skip-check", "--yes"]) == 0
    assert "submitted job 42" in capsys.readouterr().out


def test_cli_watch_and_logs_warn_without_ledger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["watch", "12345"]) == 0
    assert "no recorded runs" in capsys.readouterr().err
    assert main(["logs", "12345"]) == 0
    assert "no recorded runs" in capsys.readouterr().err


def test_cli_watch_warns_for_unknown_job_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    notes = tmp_path / "notes" / "runs.md"
    notes.parent.mkdir(parents=True)
    notes.write_text("## Job 999\n\n", encoding="utf-8")
    assert main(["watch", "12345"]) == 0
    assert "not found in notes/runs.md" in capsys.readouterr().err


def test_cli_check_dashboard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "train.py").write_text("print('x')\n", encoding="utf-8")
    monkeypatch.setattr("lazy107.cli.env_mod.find_conda", lambda: None)
    assert main(["check"]) == 0
    out = capsys.readouterr().out
    assert "entry: train.py" in out
    assert "conda not on PATH" in out
    assert "next: lazy107 env --yes" in out


def test_cli_check_suggests_watch_after_recorded_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "train.py").write_text("print('x')\n", encoding="utf-8")
    (tmp_path / "107.toml").write_text('conda_env = "train"\n', encoding="utf-8")
    monkeypatch.setattr("lazy107.cli.env_mod.find_conda", lambda: "/usr/bin/conda")
    monkeypatch.setattr("lazy107.cli.env_mod.env_exists", lambda conda, name: True)
    notes = tmp_path / "notes" / "runs.md"
    notes.parent.mkdir(parents=True)
    notes.write_text("## Job 42\n\n", encoding="utf-8")
    assert main(["check"]) == 0
    out = capsys.readouterr().out
    assert "env: 'train' ready" in out
    assert "next: lazy107 watch 42" in out


def test_cli_bare_invocation_prints_hint(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 1
    captured = capsys.readouterr()
    assert "typical flow" in captured.err
    assert "submit" in captured.out


def _fake_slurm(argv: list[str]) -> str:
    if argv[0] == "sacctmgr" and "assoc" in argv:
        return "competition||qos_p107-a100,qos_p107-rtx5090,qos_stu_default\n"
    if argv[0] == "sacctmgr" and "user" in argv:
        return "pb24061316|competition\n"
    if argv[0] == "scontrol":
        return (
            "PartitionName=P107-RTX5090\n"
            "   AllowAccounts=competition AllowQos=qos_p107-rtx5090\n"
            "   Default=YES\n\n"
            "PartitionName=Students\n"
            "   AllowAccounts=stu AllowQos=qos_stu_default\n"
            "   Default=NO\n"
        )
    return ""


def test_cli_discover_dry_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("lazy107.cluster.discover._slurm_run", _fake_slurm)
    assert main(["discover", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "partition=P107-RTX5090" in out
    assert "qos=qos_p107-rtx5090" in out
    assert "not writing" in out


def test_cli_discover_writes_global_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    from lazy107 import manifest

    monkeypatch.setattr("lazy107.cluster.discover._slurm_run", _fake_slurm)
    target = manifest.global_config_path()
    assert main(["discover"]) == 0
    assert "wrote" in capsys.readouterr().out
    assert target.exists()
    assert 'partition = "P107-RTX5090"' in target.read_text(encoding="utf-8")


def test_cli_discover_fails_cleanly_without_slurm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("lazy107.cluster.discover._slurm_run", lambda argv: "")
    assert main(["discover"]) == 1


def test_cli_debug_reports_cause_and_fix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "train_53073.err").write_text(
        "Traceback (most recent call last):\n"
        "  File \"train.py\", line 7, in <module>\n"
        "ModuleNotFoundError: No module named 'numpy'\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("lazy107.cli.sacct_row", lambda job_id: ("FAILED", "1:0"))
    assert main(["debug", "53073"]) == 0
    out = capsys.readouterr().out
    assert "state=FAILED" in out
    assert "exit=1:0" in out
    assert "cause: missing dependency 'numpy'" in out
    assert "lazy107 env --yes" in out


def test_cli_debug_falls_back_to_exit_code_without_logs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("lazy107.cli.sacct_row", lambda job_id: ("TIMEOUT", "0:15"))
    assert main(["debug", "53074"]) == 0
    out = capsys.readouterr().out
    assert "exit=0:15" in out
    assert "cause: terminated by SIGTERM" in out
    assert "raise `time`" in out


def test_cli_debug_reports_no_logs_when_unknown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("lazy107.cli.sacct_row", lambda job_id: None)
    assert main(["debug", "53073"]) == 0
    out = capsys.readouterr().out
    assert "state=UNKNOWN" in out
    assert "exit=n/a" in out
    assert "no log output found" in out
