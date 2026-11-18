"""Open the tunable Godot landscape preview."""
import argparse
import subprocess

from config import GODOT_PATH, PROJECT_PATH


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    subprocess.run([GODOT_PATH, '--path', PROJECT_PATH, 'res://landscape_preview.tscn',
                    '--', f'--seed={args.seed}'], check=True)


if __name__ == '__main__':
    main()
