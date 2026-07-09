"""Smoke tests for the dashboard app and launcher (no visual rendering)."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

HAS_STREAMLIT = importlib.util.find_spec("streamlit") is not None


@pytest.mark.skipif(not HAS_STREAMLIT, reason="streamlit not installed")
def test_app_module_imports():
    import src.dashboard.app as app

    assert callable(app.main)
    assert callable(app.get_service)


@pytest.mark.skipif(not HAS_STREAMLIT, reason="streamlit not installed")
def test_app_renders_offline(monkeypatch, tmp_path):
    """Execute the full app render path headlessly in offline fixture mode."""
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("EV_DASHBOARD_MODE", "offline")
    monkeypatch.setenv("EV_DASHBOARD_EXPORT_DIR", str(tmp_path))
    at = AppTest.from_file(
        os.path.join(REPO_ROOT, "src", "dashboard", "app.py"),
        default_timeout=120,
    )
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert at.title and "Positive-EV" in at.title[0].value
    assert len(at.dataframe) >= 1


def test_launcher_once_offline_runs_headless(tmp_path):
    """`run_ev_dashboard.py --mode offline --once` works without Streamlit."""
    env = dict(os.environ)
    env["EV_DASHBOARD_EXPORT_DIR"] = str(tmp_path)
    proc = subprocess.run(
        [sys.executable, os.path.join("scripts", "run_ev_dashboard.py"),
         "--mode", "offline", "--once"],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=180,
    )
    assert proc.returncode == 0, proc.stderr
    assert "EV dashboard snapshot (offline)" in proc.stdout
    assert os.path.isfile(os.path.join(tmp_path, "latest_opportunities.json"))


def test_launcher_rejects_live_mode():
    proc = subprocess.run(
        [sys.executable, os.path.join("scripts", "run_ev_dashboard.py"),
         "--mode", "live", "--once"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode != 0
