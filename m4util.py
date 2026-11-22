"""m4util.py -- tiny helpers shared by the Module 4 scripts (start / wait for / stop server processes).
Adapted from the m4util.py written in the Lecture 10 serving notebooks (REST / MLflow / Ray), DA3408 repo.
Also provides repo_path() so every part can find data/ no matter which folder it is run from."""
import os, signal, subprocess, time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent          # the submission root (holds eta_data.py and data/)
_procs = []


def repo_path(p):
    """Resolve a path given relative to the submission root (e.g. 'data/eta_v1.joblib')."""
    p = Path(p)
    return str(p if p.is_absolute() or p.exists() else ROOT / p)


def start(cmd, log="server.log", env=None):
    """Start a server as a child process; stdout/stderr go to a log file (read it when things break)."""
    e = {**os.environ, **(env or {})}
    f = open(log, "w")
    p = subprocess.Popen(cmd, shell=isinstance(cmd, str), stdout=f, stderr=subprocess.STDOUT, env=e,
                         preexec_fn=os.setsid, cwd=str(ROOT))
    _procs.append(p)
    return p


def wait_ready(url, timeout=120):
    """Poll a URL until it answers 200 (readiness probe, exactly what Kubernetes does)."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if requests.get(url, timeout=2).status_code == 200:
                return round(time.time() - t0, 1)
        except Exception:
            pass
        time.sleep(0.5)
    raise TimeoutError(f"{url} not ready after {timeout}s -- check the log file")


def stop(p, timeout=10):
    """SIGTERM the whole process group (so uvicorn workers die too), then wait until it is really gone."""
    try:
        pgid = os.getpgid(p.pid)
    except Exception:
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                os.killpg(pgid, 0)                 # signal 0 = "does the group still exist?"
            except ProcessLookupError:
                break
            time.sleep(0.2)
        else:
            os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        p.wait(timeout=2)
    except Exception:
        pass


def stop_all():
    for p in _procs:
        stop(p)
    _procs.clear()
