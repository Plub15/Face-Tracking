# Face tracking with multiple reference photos

A small Python tracker for one selected person: **YuNet** finds faces and landmarks,
and **SFace** compares them with several reference photos. Detection input and the
requested webcam capture size are **320×240**. It returns a box only when the target
is verified. This first version provides position data for future camera control.

## Run in your Conda `iot` environment

Use Python 3.11 or newer. From PowerShell in `D:\University_Works\Face-Tracking`:

```powershell
conda activate iot
python -m pip install -r requirements.txt
python face_tracker.py --download-models
```

You can manually put several photos of **the same person** in `references/alice`. Use clear
photos with exactly one visible face, varying lighting and moderate head angles.
Every photo must pass detection; the program names any invalid photo and stops.
The program trusts that you supplied the same person in every photo.

```powershell
python face_tracker.py --reference-dir references/alice --prepare-references
python face_tracker.py --reference-dir references/alice --source 0
```

Or list photos explicitly:

```powershell
python face_tracker.py --references front.jpg side.jpg dim_light.jpg --prepare-references
python face_tracker.py --references front.jpg side.jpg dim_light.jpg --source 0 --json
```

After collection releases the camera, the collector prepares each reference for recognition:
brightness multipliers **0.65 / 1.35**, gamma values **0.7 / 1.4**, and sideways
rotations **−30°, −15°, +15°, +30°**. These create up to nine fixed embeddings per
photo, including the original. Embeddings and metadata are saved atomically in
**`references/NAME/face_gallery.npz`**. Original photos are unchanged and no extra
JPEGs are written. Pretrained YuNet/SFace weights remain fixed.

Tracking reads this prepared gallery without decoding reference photos, generating
variants, or running reference inference. It checks photo content hashes and the
cache/model format first. Added, removed, edited photos or a corrupt cache require
explicit preparation; tracking never rebuilds the gallery during startup.
You can copy the person folder and cache together to the Pi.

The original reference must still contain exactly one detectable face. Rotation
uses an expanded canvas and transforms all five landmarks together. YuNet's
landmarks from the rotated view are used when they correspond to the same face;
transformed original landmarks provide a fallback if detection fails. Variants
whose cosine similarity to their original is below 0.363 are discarded.

Settings are configurable:

```powershell
python collect_reference.py -u alice --prepare-only --augmentation-angles -30 -15 15 30 --augmentation-lighting 0.65 1.35 --augmentation-gamma 0.7 1.4
python face_tracker.py --reference-dir references/alice --source 0
python face_tracker.py --reference-dir references/alice --no-augment-references
```

An augmentation option with no values disables that variant group. Duplicate
values and identity transforms (0° / factor 1 / gamma 1) are omitted. Augmentation
adds preparation time and gallery comparisons, particularly with many photos on
the Pi. Preparation reports the photo count, embedding count, and skipped variants.
Tracking stays offline and the gallery remains fixed during a run.

For older or manually managed reference folders, prepare once without collecting:

```powershell
python collect_reference.py -u alice --prepare-only
# Also supports reference folders outside the collector's references directory:
python face_tracker.py --reference-dir references/alice --prepare-references
```

`--prepare-references` exits without opening a camera. `--cache-path PATH` overrides
the cache location; use the same path for preparation and tracking. Ordinary tracking
uses the augmentation settings saved in the cache. To change settings, prepare
again with the desired options. `--no-augment-references` while tracking selects
only the original-photo embeddings already stored in the cache.

Rotations cover sideways head tilt (roll). Real left/right/up/down photos are
still useful for views a 2D rotation cannot create. Augmentation helps recognition
when the face is detected; a large tilt can still stop YuNet from finding a face.
Use `--diagnostics` to distinguish detection loss from an identity rejection.

Without activating the environment:

```powershell
conda run --no-capture-output -n iot python face_tracker.py --reference-dir references/alice --source 0 --json
```

Camera `0` is the default; try `--source 1` if you have another webcam. A video path
also works. Press **q**, close the preview, or press **Ctrl+C** to stop.

```powershell
python face_tracker.py --reference-dir references/alice --source recording.mp4 --no-display --json
python face_tracker.py --help
```

`--download-models` is a separate setup action that exits after downloading and
verifying the two ONNX files. Tracking has no network activity. Model paths default
to the `models` directory beside the script; `--model-dir PATH` overrides this.
Downloads are atomic and checked against fixed SHA-256 values. Upstream model
licenses are saved alongside them. Missing/corrupt models produce a repair command.

## Collect references automatically

The collector uses **YuNet** for face detection and five facial landmarks,
**OpenCV solvePnP** for approximate head direction, and **SFace** to keep the same
person. It uses the same dependencies and two models as the tracker:

```powershell
conda activate iot
python -m pip install -r requirements.txt
python collect_reference.py --download-models
```

If you installed the older `opencv-python` requirements, uninstall that package
before installing `requirements.txt`. Use a single OpenCV distribution:
`opencv-contrib-python` 4.x. The environment used for development has already been
migrated. Tracker and collector share `requirements.txt`; no additional collector
packages or model bundles are needed.

```powershell
python collect_reference.py -n alice
python collect_reference.py -u alice --photos-per-direction 5
python face_tracker.py --reference-dir references/alice --source 0 --json
```

- `-n/--name NAME` creates a new `references/NAME` collection. If the name already
  exists, it reports the conflict and exits before opening the camera. The folder
  is created only when the first photo is ready to save.
- `-u/--update NAME` requires an existing folder and **appends** a new complete
  session, preserving existing photos. An empty folder starts fresh enrollment.
  Folder names may contain spaces (quote them); paths and traversal are rejected.
- Choose exactly one mode. `--download-models` is a separate setup command.
- The automatic sequence is **front → your left → your right → up → down**,
  with **20 photos per direction** in the current configuration (100 photos total). Use
  `--photos-per-direction N` to choose any positive integer. Use moderate
  turns with your whole face in view. Left/right mean your own directions; the
  preview and saved photos are unmirrored.
- Hold each requested pose for one continuous second. Failed checks or
  movement over 5° reset the hold. The sharpest qualifying frame in the hold is
  saved. Each photo requires a new hold interval; the collector stays on the same
  direction until its requested count is saved. `--samples-per-stage` remains an
  alias for `--photos-per-direction`.
- The first frontal photo establishes the pose baseline and, for a fresh folder,
  the fixed identity anchor. Update sessions validate all existing photos and use
  their fixed gallery. Later captures must reach the configured similarity
  threshold, and newly captured photos do not expand the session's identity gallery.
  The collector CLI's current default is 0.3; use `--similarity-threshold 0.5`
  for stricter matching.
- The dashboard shows the detected face, instructions, approximate angles,
  quality feedback, hold progress, and completed categories.
  Its display size is independent of the 320×240 requested camera resolution.
- Press **q**, close the preview, or use **Ctrl+C** to stop. Successful photos
  remain available after cancellation or failure. The summary lists remaining
  categories. Updating starts a new full session, allowing different lighting.
- After completion, **q**, or video completion, the camera is released and the
  entire person folder's prepared gallery is refreshed, including partial sessions
  with saved photos. Ctrl+C or a collection failure stops immediately and preserves
  the photos; use `-u NAME --prepare-only` afterward. Preparation errors preserve
  photos and the previous cache; a stale cache cannot be used for tracking.

Each JPEG has a unique session/category filename and contains the original frame
without preview annotations. A `session_*.json` manifest records settings, accepted
photos, pose estimates, quality measurements, and completion status;
the tracker ignores these JSON files. Capture timestamps are monotonic seconds
for webcams or elapsed video seconds for recorded inputs. Session creation time
is recorded separately as a UTC timestamp. Exit codes: 0 for complete/setup,
1 for errors, 2 for incomplete collection, and 130 for Ctrl+C.

Default acceptance checks:

| Check | Default |
| --- | --- |
| Face visibility | Exactly one face, valid YuNet landmarks, no clipped face box |
| Minimum face size | Shorter side ≥64 original-frame pixels |
| Brightness | Grayscale face mean between 40 and 215 |
| Sharpness | Laplacian variance ≥50 on the face resized to 128×128 |
| Front head pose | Within ±10° yaw/pitch of the frontal baseline |
| Left / right | 20–35° yaw, pitch within ±10° |
| Up / down | Up: −25° to −8° pitch; down: 12–25°; yaw within ±10° |
| Roll | Within ±15° |

Initial frontal calibration permits ±15° yaw/pitch before the baseline exists.
Angles are approximate guidance based on YuNet's five landmarks, a generic 3D face
template, and estimated camera intrinsics. They are not calibrated measurements.
The first frontal capture reduces person-specific pose bias in subsequent stages.
The collector does not measure or constrain eyelid or mouth opening.

Quality and identity thresholds are configurable; see `python collect_reference.py --help`.
For example:

```powershell
python collect_reference.py -u alice --source 1 --min-sharpness 40
python collect_reference.py -n bob --photos-per-direction 5 --hold-seconds 1.5
python collect_reference.py -n recorded --source guided_recording.mp4 --no-display
```

Recorded-video holds use the video's FPS, so processing faster than playback does
not shorten the required source-video interval. Live capture resets after a gap
over `--max-frame-gap` (default 0.5 seconds); increase it for genuinely slower
capture hardware. Use update sessions in different lighting to expand the
reference gallery, then validate recognition accuracy with your own footage.

## Identity matching and position output

Original and augmented reference embeddings are calculated once, normalized, and kept fixed. For every
frame, every detected face is aligned using its five landmarks and compared with
every reference. The face's score is its **best cosine similarity** to any reference.
Keeping separate references lets a photo in similar lighting provide the match.

To acquire the target, the highest-scoring visible face must reach
`--similarity-threshold` (default **0.5**) and beat the second-best face by
`--similarity-margin` (default **0.05**).
Detection uses `--detection-threshold` (default **0.8**). Acquisition requires
`--confirmation-frames` consecutive accepted frames (default **3**). Once acquired,
each accepted frame updates the box. While the best matching face overlaps the
last verified box by at least `--tracking-iou` (default **0.2** IoU) and the last
verification was within `--tracking-timeout` (default **0.5 seconds**), recognition
uses `--tracking-threshold` (default **0.363**, capped at the acquisition threshold).
This allows an already verified face's recognition score to dip during a head turn.
The second-face margin still applies. A jump to a different location, changed frame
size, or expired verification requires acquisition again.

A missing, rejected, or ambiguous match
immediately clears all position fields; reacquisition again requires three frames.
There is no position prediction or automatic reference updating.

The continuation cutoff follows OpenCV's [SFace example cosine threshold](https://docs.opencv.org/4.x/d0/dd4/tutorial_dnn_face.html).
It still needs tuning with your own recordings; a lower cutoff increases the chance
of an incorrect match. Use `--tracking-threshold 0.5` for the original strict cutoff.

If a head turn loses the box, run with diagnostics:

```powershell
python face_tracker.py --reference-dir references/alice --source 0 --diagnostics
```

The preview shows the rejection reason and active cutoff. Per-frame diagnostics
go to stderr with every detected face's box, detection score, and best reference
similarity. `no_faces` means YuNet found no face above the detection threshold;
`identity_below_threshold` means recognition rejected it; `ambiguous_faces` means
two faces scored too similarly; `confirming` means acquisition is still pending.
You can use `--json` alongside diagnostics; its stdout format is unchanged.

Enroll photos covering the angles and lighting you need. Repeated photos at the
same angle add less coverage than distinct angles. Full profile views or strong
occlusion may still lose detection; diagnostics reveal whether detection or
recognition needs attention.

The preview shows the verified box, frame-center crosshair, status, similarity,
and a smoothed rate of capture + processing. It excludes rendering/output overhead;
video files are processed as fast as possible, without playback pacing.

`--json` writes one JSON object per processed frame to stdout. Startup messages,
errors, and video completion go to stderr. Example:

```json
{"status":"tracking","bbox":[120,70,80,100],"center":[160.0,120.0],"center_error":[0.0,0.0],"similarity":0.82,"frame_size":[320,240],"timestamp":1234.5}
```

- `bbox` is **[x, y, width, height]**, in original input-frame pixels, clipped to
  frame boundaries. It is not [x1, y1, x2, y2].
- `center_error` is [(center_x − frame_width/2)/(frame_width/2),
  (center_y − frame_height/2)/(frame_height/2)]. Positive means right/down;
  values lie within −1 to +1.
- `status` starts as `searching`, becomes `tracking` after confirmation, and is
  `lost` after an acquired target is no longer verified, including reacquisition.
- `bbox`, `center`, and `center_error` are **null** unless status is `tracking`.
  `similarity` can still report the best rejected/pending match; it is a cosine
  score, not a probability. It is null when no valid face is detected.
- `frame_size` is [width, height]. `timestamp` is monotonic seconds at the start
  of processing, useful for elapsed times rather than calendar dates.

Webcams may ignore a requested resolution. The program reports the actual input
size. Every frame is fitted proportionally onto a 320×240 detection canvas and
boxes/landmarks are mapped back to the actual frame before recognition and output.
Other aspect ratios are letterboxed without stretching. Recognition uses the
original frame; OpenCV handles YuNet's internal padding to its network stride.

## Use from another Python program

```python
import cv2
from face_tracker import FaceTracker

# Prepare this exact photo set once via --prepare-references before running.
tracker = FaceTracker(["front.jpg", "side.jpg", "dim_light.jpg"])
capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
try:
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        result = tracker.process(frame)  # uint8 BGR image
        if result.bbox is not None:
            print(result.bbox, result.center_error)
        # Future motor control should use position only when a box is present.
finally:
    capture.release()
```

`FaceTracker.process(frame)` returns an immutable `TrackingResult` with the fields
above; `result.to_dict()` provides a JSON-compatible dictionary. Use one tracker
instance per sequential camera stream. Importing the module opens no camera and
performs no downloads. Constructing the tracker loads a previously prepared gallery.
`prepare_gallery_cache(paths)` is the corresponding preparation API; call it after
collecting/updating photos, outside the recurring tracking startup. For explicit
in-memory experiments, `FaceTracker(paths, use_cache=False)` computes enrollment.

## Validate before camera control

```powershell
conda run --no-capture-output -n iot python -m unittest discover -s tests -v
```

Regression tests cover shared enrollment, multiple-reference scoring, invalid
references, degraded augmentation rejection, cache-only startup, changed photos,
atomic cache write failures, confirmation, ambiguity, continuation, loss and
reacquisition, coordinate mapping, folder rules, parser defaults, hold resets,
pose boundaries, and append/cancellation behavior.

Run the offline real-model smoke check with a prepared reference folder:

```powershell
conda run --no-capture-output -n iot python -m tests.smoke_models --reference-dir references/alice
```

It checks existing-cache compatibility, temporary gallery preparation, identity
matching, loss/reacquisition, collector enrollment, and tracking a generated video
with JSON output. Temporary files are removed; source photos and the existing
gallery remain unchanged. Models must already be downloaded. Live pose coverage
still needs validation with a webcam or a representative guided recording.

For your own references, record clips containing two people crossing, the target
leaving while someone else stays, returning, partial occlusion, and lighting changes.
Inspect JSON and preview for wrong-person boxes and missed matches; measure FPS.
Tune thresholds on these clips. Raising the identity threshold/margin reduces
acceptance; lowering them increases the chance of following someone else. Several
poor or inconsistent reference photos can also increase false matches. Recognition
reduces identity switches but does not guarantee identity.

## Raspberry Pi 4B

The implementation uses CPU inference and has no CUDA, PyTorch, or C++ build
requirement. Processing is reusable with frames supplied by Picamera2, but CSI
camera capture and servo control are not implemented in this version. A USB webcam
can use OpenCV's `VideoCapture` on the Pi. Use `--no-display --json` without a desktop.
Install dependencies for the Pi's Python/architecture rather than copying the
Windows Conda environment. Pi performance must be measured on the actual device:
recognizing every visible face costs more than face detection alone.

## Sources and model licenses

- [libfacedetection / YuNet](https://github.com/ShiqiYu/libfacedetection)
- [OpenCV YuNet model](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
  (`face_detection_yunet_2023mar.onnx`, MIT)
- [OpenCV SFace model](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface)
  (`face_recognition_sface_2021dec.onnx`, Apache 2.0)
- [OpenCV detection and recognition API](https://docs.opencv.org/4.x/d0/dd4/tutorial_dnn_face.html)
- [OpenCV pose estimation](https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html)

Reference images and downloaded model binaries are excluded from version control.
