"""Bounded LRU of open MRC file descriptors and their parsed headers, keyed
by (resolved_path, size, mtime_ns) so a replaced source file misses the
cache and is reopened -- never serving stale data from an old fd.

Handles are refcounted and must be acquired through the `open()` context
manager. A bare fd int is never handed out: an evicted fd whose number has
been recycled by a later os.open() would make an in-flight pread return
*another file's* voxels with a 200 OK. Eviction therefore only marks an entry
dead; the last holder to release it does the close.
"""
from __future__ import annotations

import os
import threading
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path

from mrcng.mrcheader import parse_header


class _Entry:
    __slots__ = ("fd", "hdr", "refs", "evicted")

    def __init__(self, fd: int, hdr):
        self.fd = fd
        self.hdr = hdr
        self.refs = 0
        self.evicted = False


class FdCache:
    def __init__(self, max_size: int = 256):
        self._max_size = max_size
        self._entries: OrderedDict[tuple, _Entry] = OrderedDict()
        self._lock = threading.Lock()

    def _key_for(self, path: Path) -> tuple:
        st = os.stat(path)
        return (str(path), st.st_size, st.st_mtime_ns)

    @contextmanager
    def open(self, path: Path):
        """Yield (fd, header). The fd stays open for the whole body even if the
        entry is evicted meanwhile, so it is safe to hand to a worker thread."""
        entry = self._acquire(path)
        try:
            yield entry.fd, entry.hdr
        finally:
            self._release(entry)

    def _acquire(self, path: Path) -> _Entry:
        key = self._key_for(path)
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                self._entries.move_to_end(key)
                entry.refs += 1
                return entry

        fd = os.open(str(path), os.O_RDONLY)
        try:
            st = os.stat(fd)
            hdr = parse_header(fd, st.st_size, st.st_mtime_ns)
        except BaseException:
            os.close(fd)
            raise

        with self._lock:
            existing = self._entries.get(key)
            if existing is not None:  # lost the race; keep the entry already in
                os.close(fd)
                self._entries.move_to_end(key)
                existing.refs += 1
                return existing

            entry = _Entry(fd, hdr)
            entry.refs = 1
            self._entries[key] = entry
            self._evict_if_needed()
            return entry

    def _release(self, entry: _Entry) -> None:
        with self._lock:
            entry.refs -= 1
            if entry.refs == 0 and entry.evicted:
                os.close(entry.fd)

    def _evict_if_needed(self):
        """Caller holds the lock."""
        while len(self._entries) > self._max_size:
            _, entry = self._entries.popitem(last=False)
            entry.evicted = True
            if entry.refs == 0:
                os.close(entry.fd)

    def close_all(self):
        with self._lock:
            for entry in self._entries.values():
                entry.evicted = True
                if entry.refs == 0:
                    os.close(entry.fd)
            self._entries.clear()
