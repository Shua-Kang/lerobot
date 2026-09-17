# Peg insertion — complete dual-camera recording

The two MP4 files contain the complete recorded session, including the unsuccessful first grasp, re-grasp, transfer, insertion, release verification, withdrawal, parking, and shutdown.

| Version | File |
| --- | --- |
| 中文 | [peg_full_camera01_zh_small.mp4](peg_full_camera01_zh_small.mp4) |
| English | [peg_full_camera01_en_small.mp4](peg_full_camera01_en_small.mp4) |

Download the files if GitHub does not provide an inline player.

## Format and limitations

- H.264 / AVC in MP4, yuv420p, 1280 × 506, 10 fps, no audio.
- Approximately 24 minutes per file, two-pass target bitrate 420 kbit/s.
- Camera0 is on the left; camera1 is on the right. The lower panel has 20 phase-level summaries from the action log, not a verbatim internal reasoning transcript.
- The entire timeline is retained. Spatial resolution and frame rate are reduced for a smaller download; the original 15 fps high-quality files remain local.
- The source recordings last 1418.933333 and 1439.733333 seconds. Their timelines were linearly normalized to the longer duration before pairing. Synchronization and caption timing are approximate because acquisition timestamps were not saved for each frame.
- These are full-length videos, not the earlier five-minute excerpts.

## Reproduce

The session-specific render script uses the original `camera0.avi`, `camera1.avi`, and `decisions.jsonl` under `outputs/peg_20260917_023406/`. These raw files are intentionally ignored by Git. Rendering requires Python, Pillow, ffmpeg/ffprobe, and the macOS Hiragino Sans GB font (edit the font path on other platforms).

```bash
python scripts/render_peg_full.py
python scripts/compress_peg_video.py \
  outputs/peg_20260917_023406/peg_full_camera01_zh.mp4 \
  /path/to/new/peg_full_camera01_zh_small.mp4
python scripts/compress_peg_video.py \
  outputs/peg_20260917_023406/peg_full_camera01_en.mp4 \
  /path/to/new/peg_full_camera01_en_small.mp4
```

The compressor refuses to overwrite an existing destination. Rendering and compression do not connect to the robot or cameras.

`scripts/peg_session.py` is the original hardware-session recorder, not an autonomous task replay. It opens camera indices 0, 1, and 2 on startup and accepts explicitly supplied JSON commands on stdin. Verify camera identities and hardware safety before using it. Do not run it to view or regenerate the videos. Recorded open-loop motions must not be replayed against a changed scene.
