"""Detect complete new or updated JPEG files in a camera directory."""

import glob
import os
import shutil
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


Signature = Tuple[int, int]


@dataclass
class _FileState:
    published: Optional[Signature]
    candidate: Optional[Signature] = None


def _jpeg_signatures(folder: str) -> Dict[str, Signature]:
    """Return readable ``.jpg`` paths and their size/mtime signatures."""
    signatures = {}
    for path in glob.glob(os.path.join(folder, "*.jpg")):
        try:
            stat = os.stat(path)
        except FileNotFoundError:
            # A camera may replace or remove a frame between glob and stat.
            continue
        signatures[path] = (stat.st_size, stat.st_mtime_ns)
    return signatures


class StableJpegWatcher:
    """Report JPEG versions whose signature is unchanged for two scans.

    Files present when the watcher starts are treated as the baseline and are
    not reported. A new path, or a new signature at an existing path, becomes
    reportable after the same non-empty signature is observed twice. State for
    deleted paths is discarded, so recreating a path is handled as a new file
    and the state size stays proportional to the current directory contents.
    """

    def __init__(self, folder: str):
        self.folder = os.fspath(folder)
        self._states = {
            path: _FileState(published=signature)
            for path, signature in _jpeg_signatures(self.folder).items()
        }

    @property
    def tracked_file_count(self) -> int:
        """Number of paths retained between scans (useful for diagnostics)."""
        return len(self._states)

    def scan_versions(self) -> List[Tuple[str, Signature]]:
        """Return stable paths together with the exact observed signatures."""
        observed = _jpeg_signatures(self.folder)
        ready = []

        for path in sorted(observed):
            signature = observed[path]
            state = self._states.get(path)

            if state is None:
                self._states[path] = _FileState(
                    published=None,
                    candidate=signature,
                )
            elif signature == state.published:
                # A partial rewrite can return to the last published version.
                state.candidate = None
            elif signature != state.candidate:
                state.candidate = signature
            elif signature[0] > 0:
                state.published = signature
                state.candidate = None
                ready.append((path, signature))

        # Do not retain every filename a long-running camera has ever created.
        for path in self._states.keys() - observed.keys():
            del self._states[path]

        return ready

    def scan(self) -> List[str]:
        """Return paths with a new version that has stabilized since last scan."""
        return [path for path, _signature in self.scan_versions()]


def snapshot_jpeg_version(path, expected_signature, destination):
    """Copy exactly one observed file version to an immutable destination.

    Atomic camera replacement can happen after a watcher scan. The source file
    descriptor is checked before and after copying so a superseded or in-place
    changing version is skipped rather than analyzed twice under one filename.
    """
    destination = os.fspath(destination)
    parent = os.path.dirname(destination)
    if parent:
        os.makedirs(parent, exist_ok=True)
    try:
        with open(path, "rb") as source:
            before = os.fstat(source.fileno())
            signature = (before.st_size, before.st_mtime_ns)
            if signature != expected_signature or before.st_size <= 0:
                return False
            with open(destination, "wb") as target:
                shutil.copyfileobj(source, target)
            after = os.fstat(source.fileno())
            if (after.st_size, after.st_mtime_ns) != expected_signature:
                os.unlink(destination)
                return False
    except FileNotFoundError:
        try:
            os.unlink(destination)
        except FileNotFoundError:
            pass
        return False
    return True


def safe_reopen_mask(paths):
    """Mark deferred frame paths whose original contents can still be reopened.

    Unique camera filenames are immutable and all remain safe. If one filename
    appears more than once, its older entries may already have been replaced;
    only the most recent occurrence can be trusted for deferred image inference.
    """
    normalized = [os.fspath(path) for path in paths]
    seen = set()
    result = []
    for path in reversed(normalized):
        result.append(path not in seen)
        seen.add(path)
    return list(reversed(result))
