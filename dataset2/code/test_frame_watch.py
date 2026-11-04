import os
import tempfile
import unittest
from pathlib import Path

from frame_watch import (
    StableJpegWatcher,
    safe_reopen_mask,
    snapshot_jpeg_version,
)


class StableJpegWatcherTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp_dir.name)
        self.next_mtime = 2_000_000_000_000_000_000

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_version(self, path, contents, *, atomic=False, mtime=None):
        """Write a version with a deterministic, distinct nanosecond mtime."""
        if mtime is None:
            mtime = self.next_mtime
            self.next_mtime += 1
        target = Path(path)
        write_path = target.with_suffix(".tmp") if atomic else target
        write_path.write_bytes(contents)
        os.utime(write_path, ns=(mtime, mtime))
        if atomic:
            os.replace(write_path, target)
        return mtime

    def test_ignores_files_present_at_startup(self):
        existing = self.folder / "existing.jpg"
        self.write_version(existing, b"initial")

        watcher = StableJpegWatcher(self.folder)

        self.assertEqual(watcher.scan(), [])
        self.assertEqual(watcher.scan(), [])

    def test_yields_a_new_unique_file_after_two_equal_scans(self):
        watcher = StableJpegWatcher(self.folder)
        frame = self.folder / "frame-001.jpg"
        self.write_version(frame, b"complete frame")

        self.assertEqual(watcher.scan(), [])
        self.assertEqual(watcher.scan(), [str(frame)])
        self.assertEqual(watcher.scan(), [])

    def test_scan_versions_returns_the_stabilized_signature(self):
        watcher = StableJpegWatcher(self.folder)
        frame = self.folder / "frame-version.jpg"
        mtime = self.write_version(frame, b"complete frame")

        self.assertEqual(watcher.scan_versions(), [])
        self.assertEqual(
            watcher.scan_versions(),
            [(str(frame), (len(b"complete frame"), mtime))],
        )

    def test_yields_each_stable_atomic_replacement_of_the_same_path(self):
        latest = self.folder / "latest_frame.jpg"
        self.write_version(latest, b"startup frame")
        watcher = StableJpegWatcher(self.folder)

        self.write_version(latest, b"second frame", atomic=True)
        self.assertEqual(watcher.scan(), [])
        self.assertEqual(watcher.scan(), [str(latest)])
        self.assertEqual(watcher.scan(), [])

        self.write_version(latest, b"third frame", atomic=True)
        self.assertEqual(watcher.scan(), [])
        self.assertEqual(watcher.scan(), [str(latest)])
        self.assertEqual(watcher.scan(), [])

    def test_waits_until_a_changing_partial_file_is_stable(self):
        watcher = StableJpegWatcher(self.folder)
        frame = self.folder / "partial.jpg"

        self.write_version(frame, b"part one")
        self.assertEqual(watcher.scan(), [])
        self.write_version(frame, b"part one plus two")
        self.assertEqual(watcher.scan(), [])
        self.write_version(frame, b"complete frame data")
        self.assertEqual(watcher.scan(), [])

        self.assertEqual(watcher.scan(), [str(frame)])
        self.assertEqual(watcher.scan(), [])

    def test_deleted_path_is_forgotten_and_recreation_can_be_yielded(self):
        watcher = StableJpegWatcher(self.folder)
        frame = self.folder / "reused.jpg"
        old_mtime = self.write_version(frame, b"same bytes")
        self.assertEqual(watcher.scan(), [])
        self.assertEqual(watcher.scan(), [str(frame)])

        frame.unlink()
        self.assertEqual(watcher.scan(), [])
        self.assertEqual(watcher.tracked_file_count, 0)

        # Reuse the exact old signature to prove deletion cleared its history.
        self.write_version(frame, b"same bytes", mtime=old_mtime)
        self.assertEqual(watcher.scan(), [])
        self.assertEqual(watcher.scan(), [str(frame)])


class DeferredFramePathTest(unittest.TestCase):
    def test_all_unique_paths_remain_safe_to_reopen(self):
        self.assertEqual(
            safe_reopen_mask(["frame-1.jpg", "frame-2.jpg", "frame-3.jpg"]),
            [True, True, True],
        )

    def test_only_latest_occurrence_of_mutable_path_is_reopened(self):
        self.assertEqual(
            safe_reopen_mask(["latest.jpg", "latest.jpg", "latest.jpg"]),
            [False, False, True],
        )
        self.assertEqual(
            safe_reopen_mask(["latest.jpg", "unique.jpg", "latest.jpg"]),
            [False, True, True],
        )


class ImmutableSnapshotTest(unittest.TestCase):
    def test_copies_only_the_expected_source_version(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "latest.jpg"
            destination = root / "snapshots" / "latest.jpg"
            source.write_bytes(b"first version")
            stat = source.stat()
            signature = (stat.st_size, stat.st_mtime_ns)

            self.assertTrue(
                snapshot_jpeg_version(source, signature, destination),
            )
            self.assertEqual(destination.read_bytes(), b"first version")

            replacement = root / "replacement.tmp"
            replacement.write_bytes(b"new version with a different signature")
            os.replace(replacement, source)
            stale_destination = root / "stale" / "latest.jpg"
            self.assertFalse(
                snapshot_jpeg_version(source, signature, stale_destination),
            )
            self.assertFalse(stale_destination.exists())


if __name__ == "__main__":
    unittest.main()
