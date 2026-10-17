"""Atomic writes and resumable checkpoints.

Every long computation in the experiments saves what it has computed as soon
as it has computed it, so that a job killed by the scheduler (e.g. Triton's
wall-time limit) loses at most the unit of work that was running:

* :func:`atomic_write_bytes`, :func:`atomic_json`, :func:`atomic_npz`,
  :func:`atomic_pickle` write to ``<path>.tmp.<host>.<pid>`` and then
  ``os.replace`` it onto ``<path>``. A reader therefore sees either the old
  file or the complete new one, never a truncated file, also on Lustre/NFS.
* :class:`ChunkStore` keeps a directory of numbered chunk files for a
  computation over a long index range (e.g. exact curvature on 4x10^5 edges);
  on restart only the missing chunks are recomputed.
* :func:`fingerprint` hashes the inputs (graph, configuration) into the
  checkpoint names, so stale partial results from a different configuration
  are never mixed in.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import pickle
import socket
from typing import Any, Callable, Optional

import numpy as np

__all__ = ["tmp_path", "atomic_write_bytes", "atomic_text", "atomic_json", "read_json", "atomic_npz", "read_npz",
           "atomic_pickle", "read_pickle", "fingerprint", "ChunkStore", "cleanup_tmp"]


def tmp_path(path: str) -> str:
    """Process-unique temporary name next to ``path``."""
    return f"{path}.tmp.{socket.gethostname()}.{os.getpid()}"


def _replace(tmp: str, path: str) -> None:
    os.replace(tmp, path)
    try:  # make the rename durable where the filesystem allows it
        fd = os.open(os.path.dirname(os.path.abspath(path)), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def _prepare(path: str) -> str:
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    return tmp_path(path)


def atomic_write_bytes(data: bytes, path: str) -> None:
    tmp = _prepare(path)
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    _replace(tmp, path)


def atomic_text(text: str, path: str) -> None:
    atomic_write_bytes(text.encode("utf-8"), path)


def atomic_json(obj: Any, path: str, indent: int = 2) -> None:
    atomic_write_bytes(json.dumps(obj, indent=indent, default=_json_default).encode("utf-8"), path)


def _json_default(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.bool_,)):
        return bool(x)
    raise TypeError(f"not JSON serializable: {type(x)}")


def read_json(path: str, default: Any = None) -> Any:
    """Read JSON, returning ``default`` if the file is missing or unreadable."""
    try:
        with open(path) as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def atomic_npz(path: str, **arrays) -> None:
    tmp = _prepare(path)
    with open(tmp, "wb") as fh:  # file object: numpy must not append ".npz" to the tmp name
        np.savez(fh, **arrays)
        fh.flush()
        os.fsync(fh.fileno())
    _replace(tmp, path)


def read_npz(path: str) -> Optional[dict]:
    """Load an .npz into a dict, or None if missing/corrupt."""
    try:
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    except (FileNotFoundError, OSError, ValueError, EOFError):
        return None


def atomic_pickle(obj: Any, path: str) -> None:
    atomic_write_bytes(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL), path)


def read_pickle(path: str, default: Any = None) -> Any:
    try:
        with open(path, "rb") as fh:
            return pickle.load(fh)
    except (FileNotFoundError, OSError, EOFError, pickle.UnpicklingError, AttributeError, ImportError):
        return default


def fingerprint(*parts: Any, length: int = 16) -> str:
    """Stable short hash of arrays / JSON-able objects."""
    h = hashlib.sha256()
    for p in parts:
        if isinstance(p, np.ndarray):
            a = np.ascontiguousarray(p)
            h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
        else:
            h.update(json.dumps(p, sort_keys=True, default=_json_default).encode())
        h.update(b"|")
    return h.hexdigest()[:length]


def cleanup_tmp(directory: str, older_than: float = 6 * 3600) -> int:
    """Remove leftover ``*.tmp.*`` files of killed writers under ``directory``.
    Only files not modified for ``older_than`` seconds are removed, so that a
    concurrent job's in-progress write is never touched."""
    import time
    n, now = 0, time.time()
    for f in glob.glob(os.path.join(directory, "**", "*.tmp.*"), recursive=True):
        try:
            if now - os.path.getmtime(f) > older_than:
                os.remove(f); n += 1
        except OSError:
            pass
    return n


class ChunkStore:
    """Resumable computation of ``f(ids) -> array`` over ``ids`` in chunks.

    Each chunk is written atomically to ``<dir>/chunk<k>.npy`` as soon as it is
    computed. ``run`` returns the concatenation over all chunks, computing only
    the chunks that are missing. ``dir`` should contain a fingerprint of the
    inputs, so that a different graph or configuration never reuses chunks.
    """

    def __init__(self, directory: str, chunk_size: int):
        self.dir = directory
        self.chunk_size = max(1, int(chunk_size))

    def _path(self, k: int) -> str:
        return os.path.join(self.dir, f"chunk{k:06d}.npy")

    def run(self, ids: np.ndarray, f: Callable[[np.ndarray], np.ndarray],
            log: Optional[Callable[[str], None]] = None) -> np.ndarray:
        ids = np.asarray(ids)
        n = len(ids)
        nch = max(1, -(-n // self.chunk_size))
        os.makedirs(self.dir, exist_ok=True)
        parts = []
        done_before = sum(os.path.exists(self._path(k)) for k in range(nch))
        if log and done_before:
            log(f"resuming: {done_before}/{nch} chunks already on disk ({self.dir})")
        for k in range(nch):
            sl = ids[k * self.chunk_size:(k + 1) * self.chunk_size]
            p = self._path(k)
            arr = None
            if os.path.exists(p):
                try:
                    arr = np.load(p, allow_pickle=False)
                    if arr.shape[0] != len(sl):
                        arr = None
                except (OSError, ValueError, EOFError):
                    arr = None
            if arr is None:
                arr = np.asarray(f(sl))
                tmp = _prepare(p)
                with open(tmp, "wb") as fh:
                    np.save(fh, arr)
                    fh.flush()
                    os.fsync(fh.fileno())
                _replace(tmp, p)
                if log:
                    log(f"chunk {k + 1}/{nch} saved ({len(sl)} items)")
            parts.append(arr)
        return np.concatenate(parts, axis=0) if parts else np.empty((0,))

    def clear(self) -> None:
        for f in glob.glob(os.path.join(self.dir, "chunk*.npy")):
            try:
                os.remove(f)
            except OSError:
                pass
        try:
            os.rmdir(self.dir)
        except OSError:
            pass
