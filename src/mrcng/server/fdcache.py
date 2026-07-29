"""Bounded LRU of open MRC file descriptors and their parsed headers, keyed
by (resolved_path, size, mtime_ns) so a replaced source file misses the
cache and is reopened -- never serving stale data from an old fd."""
from __future__ import annotations

import os
import threading
from collections import OrderedDict
from pathlib import Path

from mrcng.mrcheader import parse_header


class FdCache:
    def __init__(self, max_size: int = 256):
        self._max_size = max_size
        self._entries: OrderedDict[tuple, tuple[int, object]] = OrderedDict()
        self._lock = threading.Lock()

    def _key_for(self, path: Path) -> tuple:
        st = os.stat(path)
        return (str(path), st.st_size, st.st_mtime_ns)

    def get(self, path: Path):
        key = self._key_for(path)
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                self._entries.move_to_end(key)
                return entry[1]

        fd = os.open(str(path), os.O_RDONLY)
        st = os.stat(fd)
        hdr = parse_header(fd, st.st_size, st.st_mtime_ns)

        with self._lock:
            existing = self._entries.get(key)
            if existing is not None:
                os.close(fd)
                self._entries.move_to_end(key)
                return existing[1]

            self._entries[key] = (fd, hdr)
            self._evict_if_needed()
            return hdr

    def fd_for(self, path: Path) -> int:
        key = self._key_for(path)
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                raise KeyError(f"no cached fd for {path}; call get() first")
            return entry[0]

    def _evict_if_needed(self):
        while len(self._entries) > self._max_size:
            _, (fd, _) = self._entries.popitem(last=False)
            os.close(fd)

    def close_all(self):
        with self._lock:
            for fd, _ in self._entries.values():
                os.close(fd)
            self._entries.clear()
