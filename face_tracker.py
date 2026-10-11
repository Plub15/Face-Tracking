"""YuNet + SFace tracking of one person enrolled from multiple photographs."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
import zipfile
from typing import Sequence
from urllib.request import Request, urlopen

import cv2
import numpy as np


INPUT_SIZE = (320, 240)  # width, height
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "models"
GALLERY_CACHE_NAME = "face_gallery.npz"
GALLERY_CACHE_VERSION = 1
ZOO_MEDIA = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models"
ZOO_RAW = "https://raw.githubusercontent.com/opencv/opencv_zoo/main/models"


@dataclass(frozen=True)
class ModelFile:
    name: str
    directory: str
    sha256: str
    size: int


MODEL_FILES = (
    ModelFile(
        "face_detection_yunet_2023mar.onnx", "face_detection_yunet",
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4", 232589,
    ),
    ModelFile(
        "face_recognition_sface_2021dec.onnx", "face_recognition_sface",
        "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79", 38696353,
    ),
)


def verify_model(path: Path, model: ModelFile) -> None:
    if not path.is_file():
        raise ValueError(f"Missing model: {path}. Run with --download-models first.")
    if path.stat().st_size != model.size:
        raise ValueError(f"Invalid model size: {path}. Run --download-models to repair it.")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != model.sha256:
        raise ValueError(f"Model checksum mismatch: {path}. Run --download-models to repair it.")


def _download(url: str, destination: Path, model: ModelFile | None = None) -> None:
    """Download atomically; never replace a working model with an incomplete file."""
    temporary = None
    try:
        request = Request(url, headers={"User-Agent": "python-face-tracker/1.0"})
        with urlopen(request, timeout=30) as response:
            with tempfile.NamedTemporaryFile(
                dir=destination.parent, suffix=".part", delete=False
            ) as stream:
                temporary = Path(stream.name)
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    stream.write(block)
        if model is not None:
            verify_model(temporary, model)
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def download_models(model_dir: Path = DEFAULT_MODEL_DIR) -> None:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    for model in MODEL_FILES:
        # Keep each upstream license next to the corresponding ONNX model.
        license_path = model_dir / f"{model.directory}.LICENSE"
        if not license_path.is_file():
            _download(f"{ZOO_RAW}/{model.directory}/LICENSE", license_path)
        path = model_dir / model.name
        try:
            verify_model(path, model)
        except ValueError:
            print(f"Downloading {model.name} ...", file=sys.stderr)
            _download(f"{ZOO_MEDIA}/{model.directory}/{model.name}", path, model)
        print(f"Verified {path}", file=sys.stderr)


@dataclass(frozen=True)
class LetterboxTransform:
    scale_x: float
    scale_y: float
    left: int
    top: int

    def restore_faces(self, faces: np.ndarray) -> np.ndarray:
        result = np.array(faces, dtype=np.float32, copy=True)
        result[:, 0] = (result[:, 0] - self.left) / self.scale_x
        result[:, 1] = (result[:, 1] - self.top) / self.scale_y
        result[:, 2] /= self.scale_x
        result[:, 3] /= self.scale_y
        result[:, 4:14:2] = (result[:, 4:14:2] - self.left) / self.scale_x
        result[:, 5:14:2] = (result[:, 5:14:2] - self.top) / self.scale_y
        return result


def letterbox(frame: np.ndarray) -> tuple[np.ndarray, LetterboxTransform]:
    height, width = frame.shape[:2]
    target_w, target_h = INPUT_SIZE
    scale = min(target_w / width, target_h / height)
    new_w = max(1, min(target_w, round(width * scale)))
    new_h = max(1, min(target_h, round(height * scale)))
    left, top = (target_w - new_w) // 2, (target_h - new_h) // 2
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    image = cv2.copyMakeBorder(
        resized, top, target_h - new_h - top, left, target_w - new_w - left,
        cv2.BORDER_CONSTANT, value=(0, 0, 0),
    )
    # Account for rounding independently on each axis when restoring coordinates.
    return image, LetterboxTransform(new_w / width, new_h / height, left, top)


def clip_bbox(face: np.ndarray, frame_size: tuple[int, int]) -> tuple[int, int, int, int] | None:
    width, height = frame_size
    x, y, w, h = map(float, face[:4])
    if not all(math.isfinite(value) for value in (x, y, w, h)) or w <= 0 or h <= 0:
        return None
    left = max(0, min(width, math.floor(x)))
    top = max(0, min(height, math.floor(y)))
    right = max(0, min(width, math.ceil(x + w)))
    bottom = max(0, min(height, math.ceil(y + h)))
    if right <= left or bottom <= top:
        return None
    return left, top, right - left, bottom - top


def normalized_embedding(feature: np.ndarray) -> np.ndarray:
    vector = np.asarray(feature, dtype=np.float32).reshape(-1).copy()
    norm = float(np.linalg.norm(vector))
    if not np.all(np.isfinite(vector)) or not math.isfinite(norm) or norm <= 0:
        raise ValueError("Face recognition produced an invalid embedding.")
    return vector / norm


def reference_scores(features: np.ndarray, references: np.ndarray) -> np.ndarray:
    """Each candidate's best cosine match against the fixed reference gallery."""
    if len(features) == 0:
        return np.empty(0, dtype=np.float32)
    return np.clip(np.max(features @ references.T, axis=1), -1.0, 1.0)


def bbox_iou(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    intersection = max(0, min(ax + aw, bx + bw) - max(ax, bx)) * max(0, min(ay + ah, by + bh) - max(ay, by))
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


class IdentityLock:
    """Withhold positions until confirmed; discard the lock immediately on doubt."""

    def __init__(self, threshold: float, margin: float, confirmation_frames: int):
        self.threshold = threshold
        self.margin = margin
        self.confirmation_frames = confirmation_frames
        self.matches = 0
        self.ever_tracked = False
        self.reason = "no_faces"

    def update(self, scores: np.ndarray, threshold: float | None = None) -> tuple[int | None, str, float | None]:
        threshold = self.threshold if threshold is None else threshold
        best_score = None
        accepted = False
        best = 0
        self.reason = "no_faces" if len(scores) == 0 else "invalid_scores"
        if len(scores) and np.all(np.isfinite(scores)):
            ranking = np.argsort(scores)[::-1]
            best = int(ranking[0])
            best_score = float(scores[best])
            accepted = best_score >= threshold
            self.reason = "confirming" if accepted else "identity_below_threshold"
            if len(ranking) > 1:
                second = float(scores[ranking[1]])
                accepted = accepted and best_score > second and best_score - second >= self.margin
                if best_score >= threshold and not accepted:
                    self.reason = "ambiguous_faces"
        if accepted:
            self.matches = min(self.matches + 1, self.confirmation_frames)
            if self.matches >= self.confirmation_frames:
                self.ever_tracked = True
                self.reason = "verified"
                return best, "tracking", best_score
        else:
            self.matches = 0
        return None, "lost" if self.ever_tracked else "searching", best_score


@dataclass(frozen=True)
class TrackingResult:
    status: str
    bbox: tuple[int, int, int, int] | None
    center: tuple[float, float] | None
    center_error: tuple[float, float] | None
    similarity: float | None
    frame_size: tuple[int, int]
    timestamp: float

    def to_dict(self) -> dict:
        return asdict(self)


def read_image(path: Path) -> np.ndarray:
    """Support Unicode Windows paths, which cv2.imread may fail to open."""
    path = Path(path)
    try:
        data = np.fromfile(path, dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    except (OSError, cv2.error) as error:
        raise ValueError(f"Cannot read reference image: {path}") from error
    if image is None:
        raise ValueError(f"Cannot read reference image: {path}")
    return image


def collect_references(paths: Sequence[str] | None, directory: str | None) -> list[Path]:
    if directory is not None:
        folder = Path(directory)
        if not folder.is_dir():
            raise ValueError(f"Reference directory does not exist: {folder}")
        result = sorted(
            (path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}),
            key=lambda path: path.name,
        )
    else:
        result = [Path(path) for path in paths or []]
    if not result:
        raise ValueError("Provide at least one reference using --references or --reference-dir.")
    return result


def create_face_models(model_dir: str | Path, detection_threshold: float = 0.8):
    """Create verified CPU YuNet/SFace models shared by tracking and enrollment."""
    if not 0 < detection_threshold <= 1:
        raise ValueError("detection_threshold must be in (0, 1].")
    model_dir = Path(model_dir)
    for model in MODEL_FILES:
        verify_model(model_dir / model.name, model)
    detector = cv2.FaceDetectorYN.create(
        str(model_dir / MODEL_FILES[0].name), "", INPUT_SIZE,
        detection_threshold, 0.3, 5000, cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU,
    )
    recognizer = cv2.FaceRecognizerSF.create(
        str(model_dir / MODEL_FILES[1].name), "",
        cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU,
    )
    return detector, recognizer


def detect_faces(detector, frame: np.ndarray) -> np.ndarray:
    """Detect at 320x240 and return boxes/landmarks in original-frame coordinates."""
    image, transform = letterbox(frame)
    _, faces = detector.detect(image)
    if faces is None or len(faces) == 0:
        return np.empty((0, 15), dtype=np.float32)
    restored = transform.restore_faces(faces)
    frame_size = (frame.shape[1], frame.shape[0])
    valid = [face for face in restored
             if np.all(np.isfinite(face)) and clip_bbox(face, frame_size) is not None]
    return np.stack(valid) if valid else np.empty((0, 15), dtype=np.float32)


def face_embedding(recognizer, frame: np.ndarray, face: np.ndarray) -> np.ndarray:
    aligned = recognizer.alignCrop(frame, face)
    return normalized_embedding(recognizer.feature(aligned))


@dataclass(frozen=True)
class ReferenceAugmentation:
    """Extra enrollment views; model weights and source files remain unchanged."""

    enabled: bool = True
    angles: tuple[float, ...] = (-30.0, -15.0, 15.0, 30.0)
    lighting: tuple[float, ...] = (0.65, 1.35)
    gamma: tuple[float, ...] = (0.7, 1.4)

    def validate(self) -> None:
        if any(not math.isfinite(value) or abs(value) > 90 for value in self.angles):
            raise ValueError("Augmentation angles must be finite and between -90 and 90 degrees.")
        if any(not math.isfinite(value) or not 0 < value <= 3 for value in (*self.lighting, *self.gamma)):
            raise ValueError("Augmentation lighting factors and gamma values must be in (0, 3].")


def rotate_reference(image: np.ndarray, face: np.ndarray, angle: float) -> tuple[np.ndarray, np.ndarray]:
    """Rotate without clipping; transform all five landmarks with the same affine map."""
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    cosine, sine = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_width = math.ceil(width * cosine + height * sine)
    new_height = math.ceil(height * cosine + width * sine)
    matrix[:, 2] += [(new_width - width) / 2, (new_height - height) / 2]
    rotated = cv2.warpAffine(image, matrix, (new_width, new_height), flags=cv2.INTER_LINEAR)
    restored = np.array(face, dtype=np.float32, copy=True)
    x, y, w, h = face[:4]
    corners = np.array([[x, y], [x + w, y], [x, y + h], [x + w, y + h]])
    corners = corners @ matrix[:, :2].T + matrix[:, 2]
    low, high = corners.min(axis=0), corners.max(axis=0)
    restored[:4] = [*low, *(high - low)]
    landmarks = face[4:14].reshape(5, 2) @ matrix[:, :2].T + matrix[:, 2]
    restored[4:14] = landmarks.reshape(-1)
    return rotated, restored


def reference_variants(image: np.ndarray, face: np.ndarray, config: ReferenceAugmentation, detector=None):
    """Yield independent lighting/roll variants, preserving landmark correspondence.

    A tilted view can change detector alignment as well as image appearance.
    Retain transformed landmarks as a fallback if YuNet loses the augmented face.
    """
    if not config.enabled:
        return
    for factor in dict.fromkeys(config.lighting):
        if factor != 1:
            lit = np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)
            yield f"lighting:{factor:g}", lit, face
    levels = np.arange(256, dtype=np.float64) / 255
    for gamma in dict.fromkeys(config.gamma):
        if gamma != 1:
            table = np.round(255 * levels ** gamma).astype(np.uint8)
            yield f"gamma:{gamma:g}", cv2.LUT(image, table), face
    for angle in dict.fromkeys(config.angles):
        if angle != 0:
            rotated, points = rotate_reference(image, face, angle)
            if detector is not None:
                detected = detect_faces(detector, rotated)
                if len(detected) == 1:
                    size = (rotated.shape[1], rotated.shape[0])
                    expected, actual = clip_bbox(points, size), clip_bbox(detected[0], size)
                    if expected is not None and actual is not None and bbox_iou(expected, actual) >= 0.5:
                        points = detected[0]
            yield f"roll:{angle:g}", rotated, points


def add_augmentation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--augment-references", action=argparse.BooleanOptionalAction, default=True,
                        help="include lighting and roll variants when preparing the gallery (default: enabled)")
    parser.add_argument("--augmentation-angles", nargs="*", type=float, default=ReferenceAugmentation().angles,
                        metavar="DEGREES", help="prepared reference roll angles (default: -30 -15 15 30)")
    parser.add_argument("--augmentation-lighting", nargs="*", type=float, default=ReferenceAugmentation().lighting,
                        metavar="FACTOR", help="prepared brightness multipliers (default: 0.65 1.35)")
    parser.add_argument("--augmentation-gamma", nargs="*", type=float, default=ReferenceAugmentation().gamma,
                        metavar="GAMMA", help="prepared gamma values (default: 0.7 1.4)")


def augmentation_from_args(args) -> ReferenceAugmentation:
    config = ReferenceAugmentation(args.augment_references, tuple(args.augmentation_angles),
                                   tuple(args.augmentation_lighting), tuple(args.augmentation_gamma))
    config.validate()
    return config


def gallery_cache_path(references: Sequence[str | Path], cache_path: str | Path | None = None) -> Path:
    return Path(cache_path) if cache_path is not None else Path(references[0]).resolve().parent / GALLERY_CACHE_NAME


def reference_fingerprints(references: Sequence[str | Path], cache_path: Path) -> list[dict]:
    result = []
    for reference in references:
        path = Path(reference).resolve()
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        result.append({"file": Path(os.path.relpath(path, cache_path.resolve().parent)).as_posix(),
                       "sha256": digest.hexdigest()})
    return sorted(result, key=lambda item: item["file"])


def load_gallery_cache(references: Sequence[str | Path], cache_path: Path,
                       augmentation: ReferenceAugmentation | None = None):
    """Load numeric arrays only; validate source contents and model/schema identity."""
    try:
        with np.load(cache_path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            embeddings = np.array(archive["embeddings"], dtype=np.float32, copy=True)
        if metadata["version"] != GALLERY_CACHE_VERSION or metadata["models"] != [asdict(model) for model in MODEL_FILES]:
            raise ValueError("cache model or format version changed")
        if metadata["sources"] != reference_fingerprints(references, cache_path):
            raise ValueError("reference photos were added, removed, or changed")
        saved_config = ReferenceAugmentation(**metadata["augmentation"])
        saved_config.validate()
        if augmentation is not None:
            requested = json.loads(json.dumps(asdict(augmentation)))
            if augmentation.enabled and metadata["augmentation"] != requested:
                raise ValueError("augmentation settings changed")
        variants = metadata["variants"]
        source_names = {item["file"] for item in metadata["sources"]}
        if (embeddings.ndim != 2 or embeddings.shape[1] != 128 or len(embeddings) == 0
                or len(embeddings) != len(variants) or not np.all(np.isfinite(embeddings))
                or not np.allclose(np.linalg.norm(embeddings, axis=1), 1, atol=0.002)
                or any(item["file"] not in source_names for item in variants)
                or {item["file"] for item in variants if item["variant"] == "original"} != source_names):
            raise ValueError("invalid embedding data")
        if augmentation is not None and not augmentation.enabled:
            indices = [i for i, item in enumerate(variants) if item["variant"] == "original"]
            embeddings = embeddings[indices]
            variants = [variants[i] for i in indices]
        embeddings.setflags(write=False)
        return embeddings, variants, int(metadata["skipped_augmentations"]), saved_config
    except (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError, EOFError, zipfile.BadZipFile) as error:
        raise ValueError(f"Cannot use prepared gallery {cache_path}: {error}. "
                         "Run face_tracker.py --prepare-references with the same reference options.") from error


def prepare_gallery_cache(references: Sequence[str | Path], model_dir: str | Path = DEFAULT_MODEL_DIR,
                          augmentation: ReferenceAugmentation | None = None,
                          detection_threshold: float = 0.8, cache_path: str | Path | None = None) -> Path:
    """Perform enrollment once and atomically save it for future tracking runs."""
    if not references:
        raise ValueError("No reference photos to prepare.")
    destination = gallery_cache_path(references, cache_path)
    sources = reference_fingerprints(references, destination)
    print(f"Preparing {len(references)} reference photos ...", file=sys.stderr)
    tracker = FaceTracker(references, model_dir, detection_threshold, augmentation=augmentation, use_cache=False)
    if sources != reference_fingerprints(references, destination):
        raise ValueError("Reference photos changed during preparation; please retry.")
    metadata = {
        "version": GALLERY_CACHE_VERSION, "models": [asdict(model) for model in MODEL_FILES],
        "sources": sources, "augmentation": asdict(tracker.augmentation),
        "detection_threshold": detection_threshold, "opencv": cv2.__version__,
        "skipped_augmentations": tracker.skipped_augmentations,
        "variants": [{"file": Path(os.path.relpath(Path(item["file"]).resolve(), destination.resolve().parent)).as_posix(),
                      "variant": item["variant"]} for item in tracker.reference_variants],
    }
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".part", delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, embeddings=tracker.reference_embeddings,
                                metadata=np.array(json.dumps(metadata, allow_nan=False)))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f"Prepared {destination}: {len(tracker.reference_embeddings)} embeddings "
          f"({tracker.skipped_augmentations} degraded variants skipped).", file=sys.stderr)
    return destination


class FaceTracker:
    """Process BGR uint8 frames. One instance belongs to one sequential camera stream."""

    def __init__(
        self, references: Sequence[str | Path], model_dir: str | Path = DEFAULT_MODEL_DIR,
        detection_threshold: float = 0.8, similarity_threshold: float = 0.5,
        similarity_margin: float = 0.05, confirmation_frames: int = 3,
        tracking_threshold: float | None = None, tracking_iou: float = 0.2,
        tracking_timeout: float = 0.5,
        augmentation: ReferenceAugmentation | None = None,
        cache_path: str | Path | None = None, use_cache: bool = True,
    ):
        if not 0 < detection_threshold <= 1:
            raise ValueError("detection_threshold must be in (0, 1].")
        if not -1 <= similarity_threshold <= 1:
            raise ValueError("similarity_threshold must be in [-1, 1].")
        if not 0 <= similarity_margin <= 2:
            raise ValueError("similarity_margin must be in [0, 2].")
        if not isinstance(confirmation_frames, int) or confirmation_frames < 1:
            raise ValueError("confirmation_frames must be a positive integer.")
        tracking_threshold = min(0.363, similarity_threshold) if tracking_threshold is None else tracking_threshold
        if not -1 <= tracking_threshold <= similarity_threshold:
            raise ValueError("tracking_threshold must be in [-1, similarity_threshold].")
        if not 0 < tracking_iou <= 1 or not math.isfinite(tracking_timeout) or tracking_timeout <= 0:
            raise ValueError("tracking_iou must be in (0, 1]; tracking_timeout must be positive and finite.")
        if not references:
            raise ValueError("Provide at least one reference image.")
        self.augmentation = ReferenceAugmentation() if augmentation is None else augmentation
        self.augmentation.validate()
        self.detector, self.recognizer = create_face_models(model_dir, detection_threshold)
        self.identity_lock = IdentityLock(similarity_threshold, similarity_margin, confirmation_frames)
        self.tracking_threshold = tracking_threshold
        self.tracking_iou = tracking_iou
        self.tracking_timeout = tracking_timeout
        self.last_bbox = self.last_frame_size = self.last_verified_time = None
        self.last_diagnostics = {}
        self.reference_count = len(references)
        self.reference_variants = []
        self.skipped_augmentations = 0
        self.cache_loaded = False
        if use_cache:
            self.reference_embeddings, self.reference_variants, self.skipped_augmentations, self.augmentation = load_gallery_cache(
                references, gallery_cache_path(references, cache_path), augmentation)
            self.cache_loaded = True
            return
        gallery = []
        for reference in references:
            path = Path(reference)
            image = read_image(path)
            faces = self._detect(image)
            if len(faces) != 1:
                raise ValueError(f"Reference {path}: expected exactly one detectable face, found {len(faces)}.")
            try:
                original = self._feature(image, faces[0])
                gallery.append(original)
                self.reference_variants.append({"file": str(path), "variant": "original"})
                for label, augmented, augmented_face in reference_variants(image, faces[0], self.augmentation, self.detector):
                    try:
                        feature = self._feature(augmented, augmented_face)
                    except (ValueError, cv2.error):
                        self.skipped_augmentations += 1
                        continue
                    # Do not enroll heavily degraded variants that no longer
                    # resemble their own original, even though the transform is known.
                    if float(feature @ original) < 0.363:
                        self.skipped_augmentations += 1
                        continue
                    gallery.append(feature)
                    self.reference_variants.append({"file": str(path), "variant": label})
            except (ValueError, cv2.error) as error:
                raise ValueError(f"Cannot enroll reference {path}: {error}") from error
        self.reference_embeddings = np.stack(gallery)
        self.reference_embeddings.setflags(write=False)

    def _detect(self, frame: np.ndarray) -> np.ndarray:
        return detect_faces(self.detector, frame)

    def _feature(self, frame: np.ndarray, face: np.ndarray) -> np.ndarray:
        return face_embedding(self.recognizer, frame, face)

    def process(self, frame: np.ndarray) -> TrackingResult:
        if (not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3
                or frame.dtype != np.uint8 or min(frame.shape[:2]) < 1):
            raise ValueError("frame must be a nonempty BGR uint8 image of shape (height, width, 3).")
        timestamp = time.monotonic()
        frame_size = (frame.shape[1], frame.shape[0])
        faces = self._detect(frame)
        features = np.stack([self._feature(frame, face) for face in faces]) if len(faces) else np.empty((0, 0))
        scores = reference_scores(features, self.reference_embeddings)
        boxes = [clip_bbox(face, frame_size) for face in faces]
        continuing = False
        if len(scores) and np.all(np.isfinite(scores)) and self.last_bbox is not None:
            best = int(np.argmax(scores))
            continuing = (frame_size == self.last_frame_size
                          and 0 <= timestamp - self.last_verified_time <= self.tracking_timeout
                          and bbox_iou(self.last_bbox, boxes[best]) >= self.tracking_iou)
        # A jump, stale frame, or changed resolution requires acquisition again.
        if self.last_bbox is not None and not continuing:
            self.identity_lock.matches = 0
        threshold = self.tracking_threshold if continuing else self.identity_lock.threshold
        index, status, similarity = self.identity_lock.update(scores, threshold)
        bbox = center = center_error = None
        if index is not None:
            bbox = boxes[index]
            x, y, w, h = bbox
            center = (x + w / 2, y + h / 2)
            center_error = (2 * center[0] / frame_size[0] - 1, 2 * center[1] / frame_size[1] - 1)
        self.last_bbox = bbox
        self.last_frame_size = frame_size if bbox is not None else None
        self.last_verified_time = timestamp if bbox is not None else None
        self.last_diagnostics = {
            "timestamp": timestamp, "status": status, "reason": self.identity_lock.reason,
            "face_count": len(faces), "active_threshold": threshold, "continuing": continuing,
            "faces": [{"bbox": box, "detection_score": float(face[14]), "similarity": float(score)}
                      for face, box, score in zip(faces, boxes, scores)],
        }
        return TrackingResult(status, bbox, center, center_error, similarity, frame_size, timestamp)


def draw_preview(frame: np.ndarray, result: TrackingResult, fps: float, diagnostics: dict | None = None) -> np.ndarray:
    preview = frame.copy()
    width, height = result.frame_size
    cv2.drawMarker(preview, (width // 2, height // 2), (255, 255, 255), cv2.MARKER_CROSS, 16, 1)
    if result.bbox is not None:
        x, y, w, h = result.bbox
        cv2.rectangle(preview, (x, y), (x + w - 1, y + h - 1), (0, 255, 0), 2)
        cv2.circle(preview, tuple(round(value) for value in result.center), 3, (0, 255, 0), -1)
    score = "--" if result.similarity is None else f"{result.similarity:.2f}"
    cv2.putText(preview, f"{result.status}  sim:{score}  {fps:.1f} FPS", (5, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 255, 255), 1, cv2.LINE_AA)
    if diagnostics:
        cv2.putText(preview, f"{diagnostics['reason']}  min:{diagnostics['active_threshold']:.3f}",
                    (5, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
    return preview


def run_capture(tracker: FaceTracker, source: str, display: bool, emit_json: bool, diagnostics: bool = False) -> None:
    is_camera = source.isdecimal()
    capture = cv2.VideoCapture(int(source) if is_camera else source)
    window = "Face tracker (q to quit)"
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open {'camera' if is_camera else 'video'}: {source}")
        if is_camera:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, INPUT_SIZE[0])
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, INPUT_SIZE[1])
        fps = 0.0
        frame_count = 0
        while True:
            start = time.monotonic()
            ok, frame = capture.read()
            if not ok:
                if is_camera or frame_count == 0:
                    raise ValueError(f"Cannot read a frame from source: {source}")
                print("Video complete.", file=sys.stderr)
                break
            if frame_count == 0:
                print(f"Input: {frame.shape[1]}x{frame.shape[0]}; detection: 320x240. Press q to quit.", file=sys.stderr)
            result = tracker.process(frame)
            if diagnostics:
                print(json.dumps(tracker.last_diagnostics, allow_nan=False), file=sys.stderr, flush=True)
            elapsed = time.monotonic() - start
            current_fps = 1 / max(elapsed, 1e-9)
            fps = current_fps if frame_count == 0 else 0.9 * fps + 0.1 * current_fps
            if emit_json:
                print(json.dumps(result.to_dict(), allow_nan=False), flush=True)
            if display:
                cv2.imshow(window, draw_preview(frame, result, fps, tracker.last_diagnostics if diagnostics else None))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
            frame_count += 1
    finally:
        capture.release()
        if display:
            cv2.destroyAllWindows()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-models", action="store_true", help="download/verify models and exit")
    parser.add_argument("--prepare-references", action="store_true", help="augment/enroll once, save gallery cache, and exit")
    parser.add_argument("--cache-path", type=Path, help="override prepared gallery path (default: reference folder/face_gallery.npz)")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    references = parser.add_mutually_exclusive_group()
    references.add_argument("--references", nargs="+", metavar="PHOTO")
    references.add_argument("--reference-dir", metavar="DIRECTORY")
    parser.add_argument("--source", default="0", help="webcam index or video file (default: 0)")
    parser.add_argument("--detection-threshold", type=float, default=0.8)
    parser.add_argument("--similarity-threshold", type=float, default=0.5)
    parser.add_argument("--tracking-threshold", type=float,
                        help="identity cutoff for a recent overlapping verified face (default: min(0.363, similarity threshold))")
    parser.add_argument("--tracking-iou", type=float, default=0.2, help="minimum box overlap for continuation (default: 0.2)")
    parser.add_argument("--tracking-timeout", type=float, default=0.5, help="maximum continuation gap in seconds (default: 0.5)")
    parser.add_argument("--similarity-margin", type=float, default=0.05)
    parser.add_argument("--confirmation-frames", type=int, default=3)
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--json", action="store_true", help="emit one JSON object per processed frame")
    parser.add_argument("--diagnostics", action="store_true", help="show rejection reason and write per-face scores to stderr")
    add_augmentation_arguments(parser)
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(arguments)
    try:
        if args.download_models:
            if args.prepare_references:
                parser.error("Choose --download-models or --prepare-references separately.")
            download_models(args.model_dir)
            return 0
        paths = collect_references(args.references, args.reference_dir)
        augmentation = augmentation_from_args(args)
        if args.prepare_references:
            prepare_gallery_cache(paths, args.model_dir, augmentation, args.detection_threshold, args.cache_path)
            return 0
        custom_settings = any(flag.split("=")[0] in {"--augmentation-angles", "--augmentation-lighting", "--augmentation-gamma"}
                              for flag in arguments)
        # Default tracking accepts the settings used when the cache was prepared.
        requested = augmentation if custom_settings or not args.augment_references else None
        tracker = FaceTracker(paths, args.model_dir, args.detection_threshold,
                              args.similarity_threshold, args.similarity_margin, args.confirmation_frames,
                              args.tracking_threshold, args.tracking_iou, args.tracking_timeout, requested, args.cache_path)
        print(f"Loaded prepared gallery: {len(paths)} photos, {len(tracker.reference_embeddings)} fixed embeddings.", file=sys.stderr)
        run_capture(tracker, args.source, not args.no_display, args.json, args.diagnostics)
        return 0
    except KeyboardInterrupt:
        print("Stopped.", file=sys.stderr)
        return 0
    except (ValueError, OSError, cv2.error) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
