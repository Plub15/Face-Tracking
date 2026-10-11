"""Automatically collect five head-pose references for one person."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Sequence
import uuid

import cv2
import numpy as np

import face_tracker as ft


REFERENCE_ROOT = Path(__file__).resolve().parent / "references"
STAGES = ("front", "left", "right", "up", "down")
PROMPTS = {
    "front": "Face the camera",
    "left": "Turn to YOUR LEFT (moderately)",
    "right": "Turn to YOUR RIGHT (moderately)",
    "up": "Look slightly UP",
    "down": "Look slightly DOWN",
}
# Generic face geometry in YuNet landmark order: subject's right eye, left eye,
# nose, right mouth corner, left mouth corner. Camera axes: x right, y down,
# z forward; the nose is closer to the camera than the eyes/mouth.
FACE_TEMPLATE = np.array([
    [-30, -30, 30], [30, -30, 30], [0, 0, 0],
    [-25, 30, 20], [25, 30, 20],
], dtype=np.float64)


@dataclass(frozen=True)
class CollectionConfig:
    samples_per_stage: int = 20
    hold_seconds: float = 1.0
    min_face_size: int = 64
    min_brightness: float = 40.0
    max_brightness: float = 215.0
    min_sharpness: float = 50.0
    detection_threshold: float = 0.8
    similarity_threshold: float = 0.5
    max_frame_gap: float = 0.5

    def validate(self) -> None:
        if self.samples_per_stage < 1 or self.min_face_size < 1:
            raise ValueError("Sample count and minimum face size must be positive.")
        values = list(asdict(self).values())
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Collection settings must be finite numbers.")
        if self.hold_seconds <= 0 or self.max_frame_gap <= 0 or self.min_sharpness < 0:
            raise ValueError("Hold/gap durations must be positive; sharpness must be nonnegative.")
        if not 0 <= self.min_brightness < self.max_brightness <= 255:
            raise ValueError("Brightness limits must satisfy 0 <= min < max <= 255.")
        if not 0 < self.detection_threshold <= 1 or not -1 <= self.similarity_threshold <= 1:
            raise ValueError("Invalid detection or similarity threshold.")


def resolve_destination(name: str, update: bool, root: Path | None = None) -> Path:
    """Validate folder mode before loading models or opening a camera."""
    reserved = {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"}
    reserved.update(f"{prefix}{n}" for prefix in ("COM", "LPT") for n in range(1, 10))
    if (not name or name in {".", ".."} or name[-1:] in {" ", "."}
            or any(ord(char) < 32 or char in '/\\:<>"|?*' for char in name)
            or name.split(".")[0].upper() in reserved):
        raise ValueError("Use a single valid folder name, without paths or traversal.")
    root = Path(REFERENCE_ROOT if root is None else root).resolve()
    candidate = root / name
    destination = candidate.resolve()
    if not destination.is_relative_to(root) or destination == root:
        raise ValueError("Reference folder must stay inside the references directory.")
    if update:
        if not candidate.is_dir():
            raise ValueError(f"Cannot update: reference folder does not exist: {candidate}")
    elif os.path.lexists(candidate):
        raise ValueError(f"Reference folder already exists: {candidate}. Use --update {name} to append.")
    return destination


@dataclass(frozen=True)
class Pose:
    # Positive yaw turns toward the subject's LEFT; positive pitch looks DOWN.
    yaw: float
    pitch: float
    roll: float


def estimate_head_rotation(face: np.ndarray, frame_size: tuple[int, int]) -> np.ndarray:
    """Fit approximate pose from YuNet's five points and guessed camera intrinsics."""
    points = np.asarray(face[4:14], dtype=np.float64).reshape(5, 2)
    width, height = frame_size
    x, y, w, h = face[:4]
    eye_span = np.linalg.norm(points[1] - points[0])
    if (not np.all(np.isfinite(points)) or eye_span < 5
            or np.any(points[:, 0] < x) or np.any(points[:, 0] > x + w)
            or np.any(points[:, 1] < y) or np.any(points[:, 1] > y + h)):
        raise ValueError("Invalid face landmarks.")
    focal = float(max(width, height))
    camera = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]])
    ok, rvec, tvec = cv2.solvePnP(FACE_TEMPLATE, points, camera, None, flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        raise ValueError("Could not estimate head direction.")
    rvec, tvec = cv2.solvePnPRefineLM(FACE_TEMPLATE, points, camera, None, rvec, tvec)
    rotation = cv2.Rodrigues(rvec)[0]
    projected = cv2.projectPoints(FACE_TEMPLATE, rvec, tvec, camera, None)[0].reshape(5, 2)
    error = float(np.sqrt(np.mean(np.sum((projected - points) ** 2, axis=1))))
    depths = (rotation @ FACE_TEMPLATE.T + tvec.reshape(3, 1))[2]
    if (not np.all(np.isfinite(rotation)) or not math.isfinite(error)
            or np.any(depths <= 0) or error > eye_span * 0.15):
        raise ValueError("Face landmarks do not give a clear head direction.")
    return rotation


def pose_from_rotation(rotation: np.ndarray) -> Pose:
    """Extract angles; positive yaw is the person's left, positive pitch is down."""
    yaw = math.atan2(-rotation[2, 0], math.hypot(rotation[0, 0], rotation[1, 0]))
    pitch = math.atan2(rotation[2, 1], rotation[2, 2])
    roll = math.atan2(rotation[1, 0], rotation[0, 0])
    return Pose(*map(math.degrees, (-yaw, pitch, roll)))


def measure_quality(frame: np.ndarray, face: np.ndarray, config: CollectionConfig):
    height, width = frame.shape[:2]
    x, y, w, h = map(float, face[:4])
    if not all(math.isfinite(v) for v in (x, y, w, h)):
        return {}, "Invalid face box"
    if x < 0 or y < 0 or x + w > width or y + h > height:
        return {}, "Keep your entire face inside the frame"
    if min(w, h) < config.min_face_size:
        return {}, "Move closer: face is too small"
    box = ft.clip_bbox(face, (width, height))
    if box is None:
        return {}, "Invalid face box"
    left, top, bw, bh = box
    gray = cv2.cvtColor(frame[top:top + bh, left:left + bw], cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    gray = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    metrics = {"brightness": brightness, "sharpness": sharpness}
    if not config.min_brightness <= brightness <= config.max_brightness:
        return metrics, "Adjust lighting: face is too dark or bright"
    if sharpness < config.min_sharpness:
        return metrics, "Hold still / focus camera: face is blurry"
    return metrics, ""


@dataclass
class Observation:
    bbox: tuple[int, int, int, int]
    rotation: np.ndarray
    quality: dict[str, float]
    feature: np.ndarray
    similarity: float | None


class FaceAnalyzer:
    """YuNet quality/pose and a fixed SFace identity gallery."""

    def __init__(self, model_dir: Path, config: CollectionConfig, references: Sequence[Path] = ()):
        config.validate()
        self.config = config
        self.gallery = None
        self.last_faces = np.empty((0, 15), dtype=np.float32)
        self.detector, self.recognizer = ft.create_face_models(model_dir, config.detection_threshold)
        if references:
            self.gallery, _, _ = ft.enroll_references(self.detector, self.recognizer, references)

    def close(self):
        self.detector = self.recognizer = None

    def set_anchor(self, feature: np.ndarray) -> None:
        if self.gallery is None:
            self.gallery = np.array([feature], dtype=np.float32, copy=True)
            self.gallery.setflags(write=False)

    def analyze(self, frame: np.ndarray) -> tuple[Observation | None, str]:
        faces = ft.detect_faces(self.detector, frame)
        self.last_faces = faces
        if len(faces) != 1:
            return None, "Show exactly one face" if len(faces) else "No face detected"
        face = faces[0]
        quality, reason = measure_quality(frame, face, self.config)
        if reason:
            return None, reason
        feature = ft.face_embedding(self.recognizer, frame, face)
        similarity = None
        if self.gallery is not None:
            similarity = float(ft.reference_scores(np.array([feature]), self.gallery)[0])
            if similarity < self.config.similarity_threshold:
                return None, f"Identity mismatch ({similarity:.2f}); return the enrolled person"
        height, width = frame.shape[:2]
        try:
            rotation = estimate_head_rotation(face, (width, height))
        except (ValueError, cv2.error):
            return None, "Head direction is unclear"
        return Observation(ft.clip_bbox(face, (width, height)), rotation, quality, feature, similarity), ""


@dataclass
class Candidate:
    frame: np.ndarray
    observation: Observation
    pose: Pose
    timestamp: float


class CaptureGuide:
    def __init__(self, config: CollectionConfig):
        config.validate()
        self.config = config
        self.counts = {stage: 0 for stage in STAGES}
        self.baseline = None
        self.reset_hold()

    @property
    def stage(self) -> str | None:
        return next((stage for stage in STAGES if self.counts[stage] < self.config.samples_per_stage), None)

    @property
    def complete(self) -> bool:
        return self.stage is None

    def reset_hold(self) -> None:
        self.hold_start = self.last_valid_time = None
        self.hold_pose = self.best = None

    def relative_pose(self, observation: Observation) -> Pose:
        rotation = observation.rotation
        if self.baseline is not None:
            rotation = rotation @ self.baseline.T
        return pose_from_rotation(rotation)

    def check_stage(self, pose: Pose) -> str:
        stage = self.stage
        if abs(pose.roll) > 15:
            return "Keep your head upright (avoid side tilt)"
        limit = 15 if self.baseline is None else 10
        yaw, pitch = pose.yaw, pose.pitch
        if stage == "front":
            pose_ok = abs(yaw) <= limit and abs(pitch) <= limit
        elif stage == "left":
            pose_ok = 20 <= yaw <= 45 and abs(pitch) <= 15
        elif stage == "right":
            pose_ok = -45 <= yaw <= -20 and abs(pitch) <= 15
        elif stage == "up":
            pose_ok = -25 <= pitch <= -5 and abs(yaw) <= 15
        else:  # down
            pose_ok = 5 <= pitch <= 25 and abs(yaw) <= 15
        if not pose_ok:
            return "Adjust head direction to the requested pose"
        return ""

    def tick(self, observation: Observation | None, frame: np.ndarray, now: float,
             rejection: str = "") -> tuple[str, float, Candidate | None]:
        if self.complete:
            return "Complete", 1.0, None
        if observation is None:
            self.reset_hold()
            return rejection or "Waiting for a valid face", 0.0, None
        pose = self.relative_pose(observation)
        reason = self.check_stage(pose)
        if reason:
            self.reset_hold()
            return reason, 0.0, None
        if self.last_valid_time is not None:
            unstable = max(abs(a - b) for a, b in zip(asdict(pose).values(), asdict(self.hold_pose).values())) > 5
            if now < self.last_valid_time or now - self.last_valid_time > self.config.max_frame_gap or unstable:
                self.reset_hold()
        if self.hold_start is None:
            self.hold_start, self.hold_pose = now, pose
        self.last_valid_time = now
        if self.best is None or observation.quality["sharpness"] > self.best.observation.quality["sharpness"]:
            self.best = Candidate(frame.copy(), observation, pose, now)
        progress = min(1.0, (now - self.hold_start) / self.config.hold_seconds)
        return "Hold still ...", progress, self.best if progress >= 1 else None

    def commit(self, candidate: Candidate) -> None:
        stage = self.stage
        if stage is None:
            raise ValueError("All collection stages are already complete.")
        if self.baseline is None:
            self.baseline = candidate.observation.rotation.copy()
        self.counts[stage] += 1
        self.reset_hold()

    def remaining(self) -> dict[str, int]:
        return {stage: self.config.samples_per_stage - count for stage, count in self.counts.items()
                if count < self.config.samples_per_stage}


class ReferenceSession:
    """Append unique photos; retain successful captures on errors or cancellation."""

    def __init__(self, folder: Path, update: bool, config: CollectionConfig):
        self.folder = Path(folder)
        self.update = update
        self.config = config
        self.id = uuid.uuid4().hex
        self.created_folder = False
        self.manifest_path = self.folder / f"session_{self.id}.json"
        self.manifest = {
            "session_id": self.id, "created_at": datetime.now(timezone.utc).isoformat(),
            "mode": "update" if update else "new", "settings": asdict(config),
            "complete": False, "outcome": "collecting", "captures": [],
            "remaining": {stage: config.samples_per_stage for stage in STAGES},
        }

    def _ensure_folder(self):
        if self.folder.resolve() != self.folder:
            raise ValueError("Reference folder changed during collection.")
        if self.update or self.created_folder:
            if not self.folder.is_dir():
                raise ValueError("Reference folder disappeared during collection.")
        else:
            self.folder.parent.mkdir(parents=True, exist_ok=True)
            self.folder.mkdir()  # Deliberately fail if a competing session created it.
            self.created_folder = True

    def _write_manifest(self):
        if not self.folder.is_dir():
            return
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.folder, suffix=".part", delete=False) as stream:
                temporary = Path(stream.name)
                stream.write((json.dumps(self.manifest, indent=2, allow_nan=False) + "\n").encode("utf-8"))
            os.replace(temporary, self.manifest_path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def save(self, stage: str, candidate: Candidate, remaining: dict[str, int]) -> Path:
        ok, image = cv2.imencode(".jpg", candidate.frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise OSError("Could not encode the reference photo.")
        self._ensure_folder()
        filename = f"{self.id}_{stage}_{len(self.manifest['captures']) + 1:03d}.jpg"
        path = self.folder / filename
        opened = False
        try:
            with path.open("xb") as stream:  # Never overwrite an existing photo.
                opened = True
                stream.write(image.tobytes())
        except BaseException:
            if opened:
                path.unlink(missing_ok=True)
            raise
        observation = candidate.observation
        self.manifest["captures"].append({
            "file": filename, "category": stage, "timestamp": candidate.timestamp,
            "bbox": observation.bbox, "pose": asdict(candidate.pose),
            "quality": observation.quality,
            "similarity": observation.similarity,
        })
        self.manifest["remaining"] = remaining
        self._write_manifest()
        return path

    def finish(self, guide: CaptureGuide, outcome: str):
        self.manifest.update(complete=guide.complete, outcome=outcome, remaining=guide.remaining())
        # Do not write into a folder claimed by a competing new-mode session.
        if self.update or self.created_folder:
            self._write_manifest()


def draw_collection_preview(frame, observation, guide, reason, progress, faces=()):
    # A separate dashboard keeps readable prompts without increasing camera resolution.
    image, transform = ft.letterbox(frame)
    dashboard = np.full((470, 640, 3), 25, dtype=np.uint8)
    dashboard[:240, 160:480] = image
    boxes = [face[:4] for face in faces]
    if not boxes and observation is not None:
        boxes = [observation.bbox]
    for x, y, w, h in boxes:
        left = round(x * transform.scale_x) + transform.left + 160
        top = round(y * transform.scale_y) + transform.top
        color = (0, 255, 0) if observation is not None else (0, 0, 255)
        cv2.rectangle(dashboard, (left, top), (left + round(w * transform.scale_x), top + round(h * transform.scale_y)), color, 1)
    stage = guide.stage
    lines = [f"Next: {PROMPTS[stage]}" if stage else "Collection complete",
             "Unmirrored view: LEFT / RIGHT mean YOUR left / right", reason]
    if stage is not None:
        lines.append(f"{stage.capitalize()} photos: {guide.counts[stage]}/{guide.config.samples_per_stage}")
    if observation is not None:
        pose = guide.relative_pose(observation)
        lines.extend([
            f"Approx pose: yaw {pose.yaw:+.1f}  pitch {pose.pitch:+.1f}  roll {pose.roll:+.1f}",
            f"Sharpness: {observation.quality['sharpness']:.0f}  Brightness: {observation.quality['brightness']:.0f}",
        ])
    count = sum(guide.counts.values())
    total = len(STAGES) * guide.config.samples_per_stage
    lines.append(f"Saved {count}/{total}  |  Hold: {progress * 100:.0f}%  |  q: quit")
    lines.append("Done: " + (", ".join(stage for stage in STAGES if guide.counts[stage] >= guide.config.samples_per_stage) or "none"))
    for index, text in enumerate(lines):
        cv2.putText(dashboard, text, (12, 265 + index * 23), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (235, 235, 235), 1, cv2.LINE_AA)
    return dashboard


def run_collection(folder: Path, update: bool, config: CollectionConfig, model_dir: Path,
                   source: str = "0", display: bool = True) -> bool:
    references = ft.collect_references(None, folder, allow_empty=True) if update else []
    analyzer = FaceAnalyzer(model_dir, config, references)
    guide = CaptureGuide(config)
    session = ReferenceSession(folder, update, config)
    capture = None
    window_created = False
    outcome = "cancelled"
    error_in_progress = False
    window = "Reference collection (q to quit)"
    try:
        is_camera = source.isdecimal()
        capture = cv2.VideoCapture(int(source) if is_camera else source)
        if not capture.isOpened():
            raise ValueError(f"Cannot open source: {source}")
        if is_camera:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not is_camera and (not math.isfinite(fps) or fps <= 0):
            raise ValueError("Recorded video must have a valid FPS for timed capture.")
        index = 0
        previous_reason = None
        print(f"Collecting {config.samples_per_stage} photos per direction into {folder}. "
              "Start facing the camera; left/right mean YOUR directions.", file=sys.stderr)
        while not guide.complete:
            ok, frame = capture.read()
            if not ok:
                if is_camera or index == 0:
                    raise ValueError(f"Cannot read a frame from source: {source}")
                outcome = "video ended before all categories were captured"
                break
            now = time.monotonic() if is_camera else index / fps
            observation, rejection = analyzer.analyze(frame)
            reason, progress, candidate = guide.tick(observation, frame, now, rejection)
            if reason != previous_reason:
                print(f"{guide.stage}: {reason}", file=sys.stderr)
                previous_reason = reason
            if candidate is not None:
                stage = guide.stage
                remaining = guide.remaining()
                remaining[stage] -= 1
                remaining = {key: value for key, value in remaining.items() if value > 0}
                path = session.save(stage, candidate, remaining)
                analyzer.set_anchor(candidate.observation.feature)
                guide.commit(candidate)
                print(f"Saved {path.name} ({stage}: {guide.counts[stage]}/{config.samples_per_stage}); "
                      f"next: {guide.stage or 'complete'}", file=sys.stderr)
                previous_reason = None
                progress = 0.0
                reason = "Saved. Follow the next instruction."
            if display:
                cv2.imshow(window, draw_collection_preview(frame, observation, guide, reason, progress,
                                                          getattr(analyzer, "last_faces", ())))
                window_created = True
                if cv2.waitKey(1) & 0xFF == ord("q") or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    if guide.complete:
                        outcome = "complete"
                    break
            index += 1
        if guide.complete:
            outcome = "complete"
        return guide.complete
    except BaseException as error:
        error_in_progress = True
        outcome = "interrupted" if isinstance(error, KeyboardInterrupt) else f"error: {error}"
        raise
    finally:
        cleanup_errors = []
        for label, cleanup in (("camera", capture.release if capture is not None else lambda: None),
                               ("face models", analyzer.close)):
            try:
                cleanup()
            except Exception as error:
                cleanup_errors.append(f"{label}: {error}")
                print(f"Could not close {label}: {error}", file=sys.stderr)
        if window_created:
            try:
                cv2.destroyWindow(window)
            except cv2.error:
                pass  # The user may have already closed the preview window.
        try:
            session.finish(guide, outcome)
        except OSError as error:
            print(f"Could not finalize manifest: {error}. Saved photos remain in {folder}.", file=sys.stderr)
            if not error_in_progress:
                raise
        remaining = guide.remaining()
        print(f"Collection {outcome}. Saved {len(session.manifest['captures'])} photos.", file=sys.stderr)
        if remaining:
            print("Remaining: " + ", ".join(f"{stage} ({count})" for stage, count in remaining.items()), file=sys.stderr)
        if cleanup_errors and not error_in_progress:
            raise RuntimeError("Resource cleanup failed: " + "; ".join(cleanup_errors))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("-n", "--name", metavar="NAME", help="collect into a new references/NAME folder")
    mode.add_argument("-u", "--update", metavar="NAME", help="append photos to an existing references/NAME folder")
    parser.add_argument("--download-models", action="store_true", help="download/verify collection models and exit")
    parser.add_argument("--prepare-only", action="store_true", help="prepare an existing --update folder's gallery without opening a camera")
    parser.add_argument("--model-dir", type=Path, default=ft.DEFAULT_MODEL_DIR)
    parser.add_argument("--source", default="0", help="webcam index or recorded video")
    parser.add_argument("--no-display", action="store_true", help="no preview; useful for recorded-video tests")
    ft.add_augmentation_arguments(parser)
    parser.add_argument("--photos-per-direction", "--samples-per-stage", dest="samples_per_stage",
                        type=int, default=CollectionConfig().samples_per_stage, metavar="N",
                        help="photos for each of front/left/right/up/down (positive integer; default: %(default)s)")
    for name, kind, default in (
        ("hold-seconds", float, 1.0), ("min-face-size", int, 64),
        ("min-brightness", float, 40), ("max-brightness", float, 215), ("min-sharpness", float, 50),
        ("detection-threshold", float, 0.8), ("similarity-threshold", float, 0.3),
        ("max-frame-gap", float, 0.5),
    ):
        parser.add_argument(f"--{name}", type=kind, default=default)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.download_models:
        if args.name is not None or args.update is not None or args.prepare_only:
            parser.error("--download-models is a separate setup action; omit --name/--update.")
    elif args.name is None and args.update is None:
        parser.error("Choose exactly one of -n/--name NAME or -u/--update NAME.")
    if args.prepare_only and args.update is None:
        parser.error("--prepare-only requires -u/--update NAME.")
    try:
        if args.download_models:
            ft.download_models(args.model_dir)
            return 0
        config = CollectionConfig(**{key: getattr(args, key) for key in CollectionConfig.__dataclass_fields__})
        config.validate()
        augmentation = ft.augmentation_from_args(args)
        update = args.update is not None
        folder = resolve_destination(args.update if update else args.name, update)
        if args.prepare_only:
            paths = ft.collect_references(None, folder)
            ft.prepare_gallery_cache(paths, args.model_dir, augmentation, config.detection_threshold)
            return 0
        complete = run_collection(folder, update, config, args.model_dir, args.source, not args.no_display)
        paths = ft.collect_references(None, folder, allow_empty=True) if folder.is_dir() else []
        if paths:
            print("Camera released. Preparing the reference gallery for fast tracking startup ...", file=sys.stderr)
            ft.prepare_gallery_cache(paths, args.model_dir, augmentation, config.detection_threshold)
        return 0 if complete else 2
    except KeyboardInterrupt:
        print("Stopped by Ctrl+C. Saved photos remain; use --update NAME --prepare-only to prepare their gallery.", file=sys.stderr)
        return 130
    except (ValueError, OSError, RuntimeError, cv2.error) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
