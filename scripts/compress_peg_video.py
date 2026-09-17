"""Compress a full session MP4 with two-pass H.264; never touch hardware."""

import argparse
import subprocess
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    if args.source.resolve() == args.destination.resolve():
        parser.error('Source and destination must differ.')
    if args.destination.exists():
        parser.error('Destination already exists; choose a new path.')
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='peg-h264-') as scratch:
        base = [
            'ffmpeg', '-hide_banner', '-nostdin', '-i', str(args.source),
            '-map', '0:v:0', '-an', '-vf', 'scale=1280:-2:flags=lanczos,fps=10',
            '-c:v', 'libx264', '-preset', 'fast', '-threads', '4',
            '-b:v', '420k', '-pix_fmt', 'yuv420p',
            '-passlogfile', str(Path(scratch) / 'pass'),
        ]
        subprocess.run(base + ['-pass', '1', '-f', 'null', '-y', '/dev/null'], check=True)
        subprocess.run(base + ['-pass', '2', '-movflags', '+faststart', '-n', str(args.destination)], check=True)


if __name__ == '__main__':
    main()
