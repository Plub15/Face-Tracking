from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

import collect_reference as collector
import face_tracker as ft


FRAME = np.full((240, 320, 3), 100, dtype=np.uint8)
FACE = np.array([100, 60, 80, 100, 120, 85, 160, 85, 140, 110,
                 120, 135, 160, 135, 0.99], dtype=np.float32)
FEATURE = np.eye(1, 128, dtype=np.float32)[0]


class EnrollmentTests(unittest.TestCase):
    def test_multiple_references_are_fixed_and_scored_individually(self):
        second = np.roll(FEATURE, 1)
        with patch.object(ft, "read_image", return_value=FRAME), \
                patch.object(ft, "detect_faces", return_value=np.array([FACE])), \
                patch.object(ft, "face_embedding", side_effect=[FEATURE, second]):
            gallery, variants, skipped = ft.enroll_references(None, None, ["a.jpg", "b.jpg"])
        self.assertFalse(gallery.flags.writeable)
        np.testing.assert_allclose(ft.reference_scores(np.array([FEATURE, second]), gallery), [1, 1])
        self.assertEqual([item["variant"] for item in variants], ["original", "original"])
        self.assertEqual(skipped, 0)

    def test_invalid_reference_names_the_file(self):
        for count in (0, 2):
            with self.subTest(count=count), patch.object(ft, "read_image", return_value=FRAME), \
                    patch.object(ft, "detect_faces", return_value=np.repeat([FACE], count, axis=0)):
                with self.assertRaisesRegex(ValueError, "bad.jpg.*exactly one"):
                    ft.enroll_references(None, None, ["bad.jpg"])

    def test_degraded_and_failed_variants_are_skipped(self):
        variants = [(label, FRAME, FACE) for label in ("good", "bad", "failed")]
        with patch.object(ft, "read_image", return_value=FRAME), \
                patch.object(ft, "detect_faces", return_value=np.array([FACE])), \
                patch.object(ft, "reference_variants", return_value=iter(variants)), \
                patch.object(ft, "face_embedding", side_effect=[FEATURE, FEATURE, -FEATURE, ValueError("bad")]):
            gallery, metadata, skipped = ft.enroll_references(None, None, ["a.jpg"])
        self.assertEqual(len(gallery), 2)
        self.assertEqual([item["variant"] for item in metadata], ["original", "good"])
        self.assertEqual(skipped, 2)

    def test_cache_roundtrip_startup_and_source_changes(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stderr(StringIO()):
            photo = Path(directory) / "a.jpg"
            photo.write_bytes(b"fixture")
            config = ft.ReferenceAugmentation(enabled=False)
            with patch.object(ft, "create_face_models", return_value=(None, None)), \
                    patch.object(ft, "read_image", return_value=FRAME), \
                    patch.object(ft, "detect_faces", return_value=np.array([FACE])), \
                    patch.object(ft, "face_embedding", return_value=FEATURE):
                cache = ft.prepare_gallery_cache([photo], augmentation=config)
            with patch.object(ft, "create_face_models", return_value=(None, None)), \
                    patch.object(ft, "enroll_references", side_effect=AssertionError("startup enrollment")):
                tracker = ft.FaceTracker([photo])
            self.assertTrue(tracker.cache_loaded)
            np.testing.assert_array_equal(tracker.reference_embeddings, [FEATURE])
            photo.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "photos.*changed"):
                ft.load_gallery_cache([photo], cache)

    def test_cache_write_failure_preserves_previous_cache(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stderr(StringIO()):
            photo = Path(directory) / "a.jpg"
            photo.write_bytes(b"fixture")
            cache = photo.parent / ft.GALLERY_CACHE_NAME
            cache.write_bytes(b"previous")
            with patch.object(ft, "create_face_models", return_value=(None, None)), \
                    patch.object(ft, "enroll_references", return_value=(np.array([FEATURE]),
                        [{"file": str(photo), "variant": "original"}], 0)), \
                    patch.object(ft.os, "replace", side_effect=OSError("write failed")):
                with self.assertRaisesRegex(OSError, "write failed"):
                    ft.prepare_gallery_cache([photo])
            self.assertEqual(cache.read_bytes(), b"previous")
            self.assertFalse(list(photo.parent.glob("*.part")))


class TrackingTests(unittest.TestCase):
    def test_confirmation_continuation_ambiguity_loss_and_reacquisition(self):
        with patch.object(ft, "create_face_models", return_value=(None, None)), \
                patch.object(ft, "load_gallery_cache", return_value=(np.array([FEATURE]), [], 0,
                    ft.ReferenceAugmentation())):
            tracker = ft.FaceTracker(["a.jpg"])
        score = 1.0

        def feature(*args):
            result = np.zeros(128, dtype=np.float32)
            result[:2] = [score, np.sqrt(1 - score * score)]
            return result

        with patch.object(ft, "detect_faces", return_value=np.array([FACE])) as detect, \
                patch.object(ft, "face_embedding", side_effect=feature), \
                patch.object(ft.time, "monotonic", side_effect=(n / 30 for n in range(100))):
            self.assertIsNone(tracker.process(FRAME).bbox)
            self.assertIsNone(tracker.process(FRAME).bbox)
            result = tracker.process(FRAME)
            self.assertEqual(result.bbox, (100, 60, 80, 100))
            self.assertEqual(result.center, (140, 110))
            score = 0.4
            self.assertEqual(tracker.process(FRAME).status, "tracking")
            detect.return_value = np.array([FACE, FACE])
            self.assertIsNone(tracker.process(FRAME).bbox)
            self.assertEqual(tracker.last_diagnostics["reason"], "ambiguous_faces")
            detect.return_value = np.array([FACE])
            self.assertIsNone(tracker.process(FRAME).bbox)  # .4 cannot acquire again
            score = 1.0
            self.assertIsNone(tracker.process(FRAME).bbox)
            self.assertIsNone(tracker.process(FRAME).bbox)
            self.assertEqual(tracker.process(FRAME).status, "tracking")
            detect.return_value = np.empty((0, 15))
            result = tracker.process(FRAME)
            self.assertEqual(result.status, "lost")
            self.assertIsNone(result.center_error)

    def test_letterbox_mapping_and_clipping(self):
        frame = np.zeros((200, 400, 3), dtype=np.uint8)
        image, transform = ft.letterbox(frame)
        self.assertEqual(image.shape, (240, 320, 3))
        face = FACE.copy()
        face[[0, 2, 4, 6, 8, 10, 12]] *= transform.scale_x
        face[0] += transform.left
        face[4:14:2] += transform.left
        face[[1, 3, 5, 7, 9, 11, 13]] *= transform.scale_y
        face[1] += transform.top
        face[5:14:2] += transform.top
        np.testing.assert_allclose(transform.restore_faces(np.array([face]))[0], FACE, atol=1e-5)
        self.assertEqual(ft.clip_bbox(np.array([-5, -5, 20, 20]), (10, 10)), (0, 0, 10, 10))


class CollectionTests(unittest.TestCase):
    def test_folder_modes_paths_and_empty_reference_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = collector.resolve_destination("alice", False, root)
            self.assertFalse(folder.exists())
            folder.mkdir()
            self.assertEqual(collector.resolve_destination("alice", True, root), folder)
            for name, update in (("alice", False), ("missing", True), ("../escape", False), ("CON", False)):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    collector.resolve_destination(name, update, root)
            self.assertEqual(ft.collect_references(None, folder, allow_empty=True), [])
            with self.assertRaises(ValueError):
                ft.collect_references(None, folder)
            for name in ("b.PNG", "a.jpg", "session.json"):
                (folder / name).touch()
            self.assertEqual([p.name for p in ft.collect_references(None, folder)], ["a.jpg", "b.PNG"])

    def test_parser_modes_and_configured_defaults(self):
        parser = collector.build_parser()
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["-n", "alice", "-u", "alice"])
        args = parser.parse_args(["-n", "alice", "--photos-per-direction", "3"])
        self.assertEqual(args.samples_per_stage, 3)
        defaults = parser.parse_args(["-n", "alice"])
        self.assertEqual(defaults.samples_per_stage, 20)
        self.assertEqual(defaults.similarity_threshold, 0.3)

    def test_collector_reuses_enrollment_without_augmentation(self):
        with patch.object(ft, "create_face_models", return_value=(None, None)), \
                patch.object(ft, "enroll_references", return_value=(np.array([FEATURE]), [], 0)) as enroll:
            analyzer = collector.FaceAnalyzer(Path("models"), collector.CollectionConfig(), [Path("a.jpg")])
        enroll.assert_called_once_with(None, None, [Path("a.jpg")])
        analyzer.set_anchor(-FEATURE)
        np.testing.assert_array_equal(analyzer.gallery, [FEATURE])
        analyzer.close()
        self.assertIsNone(analyzer.detector)

    def test_hold_reset_sharpest_capture_and_pose_boundaries(self):
        guide = collector.CaptureGuide(collector.CollectionConfig(samples_per_stage=1))
        observation = collector.Observation((100, 60, 80, 100), np.eye(3),
                                            {"sharpness": 60, "brightness": 100}, FEATURE, None)
        guide.tick(observation, FRAME, 0)
        self.assertIsNone(guide.tick(None, FRAME, 0.3)[2])
        guide.tick(observation, FRAME, 0.5)
        sharper = collector.Observation(observation.bbox, observation.rotation,
                                        {"sharpness": 100, "brightness": 100}, FEATURE, None)
        guide.tick(sharper, FRAME, 1.0)
        candidate = guide.tick(observation, FRAME, 1.5)[2]
        self.assertIs(candidate.observation, sharper)
        guide.commit(candidate)
        for stage, valid, invalid in (
            ("left", collector.Pose(20, 0, 0), collector.Pose(19, 0, 0)),
            ("right", collector.Pose(-35, 0, 0), collector.Pose(-36, 0, 0)),
            ("up", collector.Pose(0, -8, 0), collector.Pose(0, -7, 0)),
            ("down", collector.Pose(0, 12, 0), collector.Pose(0, 11, 0)),
        ):
            with self.subTest(stage=stage):
                self.assertEqual(guide.stage, stage)
                self.assertEqual(guide.check_stage(valid), "")
                self.assertTrue(guide.check_stage(invalid))
                guide.commit(candidate)
        self.assertTrue(guide.complete)

    def test_session_append_and_cancellation_preserve_photos(self):
        config = collector.CollectionConfig(samples_per_stage=1)
        guide = collector.CaptureGuide(config)
        observation = collector.Observation((100, 60, 80, 100), np.eye(3),
                                            {"sharpness": 60, "brightness": 100}, FEATURE, None)
        candidate = collector.Candidate(FRAME, observation, collector.Pose(0, 0, 0), 1)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory).resolve() / "alice"
            session = collector.ReferenceSession(folder, False, config)
            session.finish(guide, "cancelled")
            self.assertFalse(folder.exists())
            first = session.save("front", candidate, guide.remaining())
            original = first.read_bytes()
            session.finish(guide, "cancelled")
            update = collector.ReferenceSession(folder, True, config)
            second = update.save("front", candidate, guide.remaining())
            self.assertNotEqual(first, second)
            self.assertEqual(first.read_bytes(), original)
            self.assertEqual(len(list(folder.glob("*.jpg"))), 2)

    def test_video_collection_captures_front_and_cleans_up_at_eof(self):
        config = collector.CollectionConfig(samples_per_stage=1)
        observation = collector.Observation((100, 60, 80, 100), np.eye(3),
                                            {"sharpness": 60, "brightness": 100}, FEATURE, None)
        capture = Mock()
        capture.isOpened.return_value = True
        capture.get.return_value = 10
        capture.read.side_effect = [(True, FRAME)] * 12 + [(False, None)]
        with tempfile.TemporaryDirectory() as directory, redirect_stderr(StringIO()), \
                patch.object(collector, "FaceAnalyzer", autospec=True) as analyzer_class, \
                patch.object(collector.cv2, "VideoCapture", return_value=capture):
            analyzer = analyzer_class.return_value
            analyzer.analyze.return_value = (observation, "")
            folder = Path(directory).resolve() / "alice"
            complete = collector.run_collection(folder, False, config, Path("models"), "video.avi", False)
            self.assertFalse(complete)
            self.assertEqual(len(list(folder.glob("*.jpg"))), 1)
            analyzer.analyze.assert_called_with(FRAME)
            analyzer.set_anchor.assert_called_once_with(FEATURE)
            analyzer.close.assert_called_once()
            capture.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()
