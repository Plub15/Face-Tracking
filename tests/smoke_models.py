"""Offline integration check using an existing person's reference folder."""

import argparse
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import cv2
import numpy as np

import collect_reference as collector
import face_tracker as ft


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    args = parser.parse_args()
    references = ft.collect_references(None, args.reference_dir)
    frame = ft.read_image(references[0])
    # Verify compatibility with the existing gallery without rebuilding it.
    with patch.object(ft, "enroll_references", side_effect=AssertionError("startup enrollment")), \
            patch.object(ft, "read_image", side_effect=AssertionError("startup reference decoding")):
        existing = ft.FaceTracker(references)
    assert existing.cache_loaded

    # Keep the temporary cache on the same drive as its source photos.
    with tempfile.TemporaryDirectory(dir=args.reference_dir.resolve().parent) as directory:
        cache = Path(directory) / ft.GALLERY_CACHE_NAME
        photos = references[:2]
        ft.prepare_gallery_cache(photos, cache_path=cache)
        with patch.object(ft, "enroll_references", side_effect=AssertionError("startup enrollment")):
            tracker = ft.FaceTracker(photos, cache_path=cache)
        assert not tracker.reference_embeddings.flags.writeable
        for _ in range(3):
            result = tracker.process(frame)
        assert result.status == "tracking" and result.bbox is not None
        result = tracker.process(np.zeros_like(frame))
        assert result.status == "lost" and result.bbox is None
        for _ in range(3):
            result = tracker.process(frame)
        assert result.status == "tracking"

        analyzer = collector.FaceAnalyzer(ft.DEFAULT_MODEL_DIR, collector.CollectionConfig(), photos)
        try:
            assert len(analyzer.gallery) == len(photos)
            score = ft.reference_scores(tracker.reference_embeddings[:1], analyzer.gallery)[0]
            assert score > 0.99
        finally:
            analyzer.close()

        video = Path(directory) / "tracking.avi"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10,
                                 (frame.shape[1], frame.shape[0]))
        try:
            assert writer.isOpened()
            for _ in range(6):
                writer.write(frame)
        finally:
            writer.release()
        output = StringIO()
        with redirect_stdout(output):
            code = ft.main(["--references", *map(str, photos), "--cache-path", str(cache),
                            "--source", str(video), "--no-display", "--json"])
        rows = [json.loads(line) for line in output.getvalue().splitlines()]
        assert code == 0 and len(rows) == 6
        assert rows[0]["bbox"] is None and rows[2]["bbox"] is not None
    print("Real-model smoke passed: existing cache, preparation, identity, loss/reacquisition, collector enrollment, video JSON.")


if __name__ == "__main__":
    main()
