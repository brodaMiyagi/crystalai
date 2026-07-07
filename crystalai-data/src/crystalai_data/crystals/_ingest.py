"""Shared plumbing for the ICSD / MP-20 ingest scripts.

Two concerns live here so both converters share one implementation:

* ``.env`` reading + repo-root resolution (no python-dotenv dependency), and
* the **build-local-then-publish** flow. The repo tree is on NFS, where live
  SQLite writes corrupt the image; so every ingest builds the DB on local disk
  and copies the finished file to its final path in one shot. See the
  ``nfs-sqlite-build-local`` project note.
"""

from __future__ import annotations

import shutil
import sqlite3
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from . import database as db


def load_dotenv(path: Path) -> dict[str, str]:
    """Minimal ``KEY='value'`` reader."""
    env: dict[str, str] = {}
    if not path.is_file():
        return env
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        env[key.strip()] = val.strip().strip("'\"")
    return env


def repo_root() -> Path:
    # src/crystalai_data/crystals/_ingest.py -> repo root is 4 levels up.
    return Path(__file__).resolve().parents[4]


@contextmanager
def local_build(
    db_path: Path, work_dir: Path | None = None, *, bulk: bool = True
) -> Iterator[sqlite3.Connection]:
    """Yield a schema-ready connection to a local-disk DB, publishing on success.

    On enter: resolve a local working path (``work_dir`` or system temp), seed it
    from an existing ``db_path`` so resume/append works, open + create schema.
    On clean exit: copy the finished local DB back to ``db_path``. On exception:
    leave the local DB in place (inspectable) and do not publish.
    """
    db_path = db_path.resolve()
    work_dir = Path(work_dir) if work_dir else Path(tempfile.gettempdir())
    work_dir.mkdir(parents=True, exist_ok=True)
    work_db = work_dir / db_path.name
    publish = work_db != db_path

    print(f"[ingest] db            = {db_path}")
    print(f"[ingest] build (local) = {work_db}")
    if publish and db_path.exists() and not work_db.exists():
        print("[ingest] seeding local working copy from existing DB")
        shutil.copy2(db_path, work_db)

    conn = db.connect(work_db, bulk=bulk)
    db.create_schema(conn)
    try:
        yield conn
    finally:
        conn.close()

    if publish:
        print(f"[ingest] publishing local DB -> {db_path}")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(work_db, db_path)
