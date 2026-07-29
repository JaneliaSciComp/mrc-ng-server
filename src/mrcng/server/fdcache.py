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

import logging
import os
import threading
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path

from mrcng.fingerprint import read_fingerprint, validate
from mrcng.mrcheader import parse_header

_logger = logging.getLogger("mrcng.server")


class _Entry:
    __slots__ = ("fd", "hdr", "refs", "evicted", "validity_key", "validity", "validity_fp")

    def __init__(self, fd: int, hdr):
        self.fd = fd
        self.hdr = hdr
        self.refs = 0
        self.evicted = False
        # Fingerprint-validity memo. Keyed by fingerprint.json's own
        # (size, mtime_ns), NOT by this entry's source key -- a build that
        # completes while the source file is untouched must be picked up on
        # the next request, not wait for this entry to be evicted.
        self.validity_key = "__unset__"
        self.validity = None
        self.validity_fp = None


class Handle:
    """What open() yields. Unpacks as (fd, hdr) for the common case; also
    exposes validity_for() for the fingerprint-validity memo."""
    __slots__ = ("_cache", "_entry")

    def __init__(self, cache: "FdCache", entry: _Entry):
        self._cache = cache
        self._entry = entry

    @property
    def fd(self) -> int:
        return self._entry.fd

    @property
    def hdr(self):
        return self._entry.hdr

    def __iter__(self):
        return iter((self._entry.fd, self._entry.hdr))

    def validity_for(self, cache_dir: Path, params):
        """(Validity, fingerprint dict) for this source's cache entry, or
        (None, None) if there is no fingerprint.json."""
        return self._cache._validity_for(self._entry, cache_dir, params)


class FdCache:
    def __init__(self, max_size: int = 256, assume_mode0: str | None = None):
        self._max_size = max_size
        self._assume_mode0 = assume_mode0
        self._entries: OrderedDict[tuple, _Entry] = OrderedDict()
        self._lock = threading.Lock()
        self._eviction_count = 0

    def _key_for(self, path: Path) -> tuple:
        st = os.stat(path)
        return (str(path), st.st_size, st.st_mtime_ns)

    @contextmanager
    def open(self, path: Path):
        """Yield a Handle (unpacks as (fd, hdr)). The fd stays open for the
        whole body even if the entry is evicted meanwhile, so it is safe to
        hand to a worker thread."""
        entry = self._acquire(path)
        try:
            yield Handle(self, entry)
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
            hdr = parse_header(fd, st.st_size, st.st_mtime_ns, assume_mode0=self._assume_mode0)
        except BaseException:
            os.close(fd)
            raise

        if hdr.mode0_signedness_is_ambiguous:
            # Parsed once per fd-cache entry, i.e. once per (path, size,
            # mtime_ns) -- this already rate-limits to one warning per file
            # version rather than one per request.
            _logger.warning(
                "%s: mode-0 signedness is ambiguous (no IMOD stamp), defaulting to "
                "int8; set MRCNG_ASSUME_MODE0 to override", path,
            )

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

    def _validity_for(self, entry: _Entry, cache_dir: Path, params):
        # A stat call on fingerprint.json here, versus a full read + JSON
        # parse + sha256-over-the-header on every request when nothing about
        # the cache has changed.
        try:
            st = os.stat(Path(cache_dir) / "fingerprint.json")
            fp_key = (st.st_size, st.st_mtime_ns)
        except FileNotFoundError:
            fp_key = None

        with self._lock:
            if entry.validity_key == fp_key:
                return entry.validity, entry.validity_fp

        if fp_key is None:
            computed_validity, fp = None, None
        else:
            fp = read_fingerprint(cache_dir)
            computed_validity = validate(fp, entry.hdr, entry.fd, params) if fp is not None else None

        with self._lock:
            entry.validity_key = fp_key
            entry.validity = computed_validity
            entry.validity_fp = fp
        return computed_validity, fp

    def _evict_if_needed(self):
        """Caller holds the lock."""
        while len(self._entries) > self._max_size:
            _, entry = self._entries.popitem(last=False)
            entry.evicted = True
            if entry.refs == 0:
                os.close(entry.fd)
            self._eviction_count += 1
            # One warning per "cache size worth" of evictions, not one per
            # eviction: scales the frequency to how bad the situation is
            # without flooding logs. A high rate means the working set of
            # files exceeds fd_cache_size, so every request pays an open().
            if self._eviction_count % self._max_size == 0:
                _logger.warning(
                    "fd cache has evicted %d entries (cache size %d) -- the "
                    "working set of files may exceed fd_cache_size",
                    self._eviction_count, self._max_size,
                )

    def close_all(self):
        with self._lock:
            for entry in self._entries.values():
                entry.evicted = True
                if entry.refs == 0:
                    os.close(entry.fd)
            self._entries.clear()
