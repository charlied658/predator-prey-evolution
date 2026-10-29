"""Quick ecosystem demos and reproducible, short headless experiments."""
import argparse
import json
import subprocess
from pathlib import Path
from config import GODOT_PATH, PROJECT_PATH


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('ecosystem.json'))
    parser.add_argument('--seed', type=int)
    parser.add_argument('--duration', type=float)
    parser.add_argument('--predators', type=int)
    parser.add_argument('--prey', type=int)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('results/ecosystem-summary.json'))
    args = parser.parse_args()
    settings = json.loads(args.config.read_text())
    for key in ('seed', 'duration', 'predators', 'prey'):
        if getattr(args, key) is not None:
            settings[key] = getattr(args, key)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Store the exact settings alongside the summary for repeatable comparisons.
    config_path = args.output.with_suffix('.config.json').resolve()
    config_path.write_text(json.dumps(settings, indent=2) + '\n')
    command = [GODOT_PATH, '--log-file', str(args.output.with_suffix('.log').resolve())]
    if args.headless:
        command += ['--headless', '--fixed-fps', '60']
    command += ['--path', PROJECT_PATH, 'res://ecosystem.tscn', '--',
                '--ecosystem-config', str(config_path), '--ecosystem-output', str(args.output.resolve())]
    subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
