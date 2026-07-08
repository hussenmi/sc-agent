"""Precedence of _load_dotenv: the shared $SCAGENT_HOME/.env must win.

When scagent is installed as a shared Lmod module, $SCAGENT_HOME points at the
install dir and its .env is the centrally-managed lab config. Editing that one
file has to control every user's setup, even if they run from a directory that
happens to contain its own ./.env. See scagent/agent/agent.py:_load_dotenv.
"""

import os


def _write_env(path, **values):
    path.write_text("".join(f"{k}={v}\n" for k, v in values.items()))


def test_scagent_home_env_overrides_cwd(tmp_path, monkeypatch):
    from scagent.agent.agent import _load_dotenv

    cwd = tmp_path / "work"
    home = tmp_path / "install"
    cwd.mkdir()
    home.mkdir()
    _write_env(cwd / ".env", SCAGENT_PROVIDER="local_cwd", DP_CWD_ONLY="yes")
    _write_env(home / ".env", SCAGENT_PROVIDER="shared_home")

    monkeypatch.chdir(cwd)
    monkeypatch.setenv("SCAGENT_HOME", str(home))
    monkeypatch.delenv("SCAGENT_PROVIDER", raising=False)
    monkeypatch.delenv("DP_CWD_ONLY", raising=False)

    assert _load_dotenv() is True
    # The shared $SCAGENT_HOME/.env is loaded last and wins the conflict.
    assert os.environ["SCAGENT_PROVIDER"] == "shared_home"
    # A var only in the local .env is still merged in (not first-wins-and-stop).
    assert os.environ["DP_CWD_ONLY"] == "yes"


def test_cwd_env_used_when_no_scagent_home(tmp_path, monkeypatch):
    from scagent.agent.agent import _load_dotenv

    cwd = tmp_path / "work2"
    cwd.mkdir()
    _write_env(cwd / ".env", DP_LOCAL="local_only")

    monkeypatch.chdir(cwd)
    monkeypatch.delenv("SCAGENT_HOME", raising=False)
    monkeypatch.delenv("DP_LOCAL", raising=False)

    assert _load_dotenv() is True
    assert os.environ["DP_LOCAL"] == "local_only"
