"""Replay an identified study controller and an explicit scenario."""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from config import GODOT_PATH, PROJECT_PATH


def resolve_controller(path, condition, seed):
    if path.is_file():
        return path
    if (path / 'best_predator.json').exists():
        return path / 'best_predator.json'
    candidates = sorted(path.glob(f'{condition}-seed-*/best_predator.json'))
    if seed is not None:
        candidates = [p for p in candidates if p.parent.name == f'{condition}-seed-{seed}']
    if not candidates:
        raise ValueError(f'No matching controllers in {path}')
    def score(candidate):
        report = json.loads((candidate.parent / 'report.json').read_text())
        return report['history'][-1]['champion_validation_capture_rate']
    return max(candidates, key=score)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('controller', type=Path, help='Champion JSON, run directory, or study directory')
    parser.add_argument('--condition', choices=['fixed', 'diverse'], default='diverse')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--opponent', choices=['saved', 'stationary', 'straight', 'flee', 'neural'], default='saved')
    parser.add_argument('--case-index', type=int, default=0, help='Index within the selected opponent type')
    args = parser.parse_args()
    try:
        path = resolve_controller(args.controller, args.condition, args.seed).resolve()
        data = json.loads(path.read_text())
        results_path = path.parent / 'test_encounters.json'
        if results_path.exists():
            cases = json.loads(results_path.read_text())
            if args.opponent != 'saved':
                cases = [case for case in cases if case.get('prey_controller', 'neural') == args.opponent]
            if not 0 <= args.case_index < len(cases):
                raise ValueError(f'case-index must be between 0 and {len(cases) - 1}')
            case = cases[args.case_index]
            print(f"Recorded outcome: captured={case['captured']}, time={case['time']:.2f}s", flush=True)
        else:
            if args.case_index != 0:
                raise ValueError('This controller has only one saved scenario')
            case = data.get('replay_encounter', {'prey_genome': data['prey_genome']})
            if args.opponent != 'saved':
                case = {**case, 'prey_controller': args.opponent}
        case = {**case, 'predator_genome': data['genome']}
        print(f"Controller: {path}\nCondition: {data.get('condition', 'legacy')}; seed: {data.get('seed', 'unknown')}\n"
              f"Prey: {case.get('prey_controller', 'neural')}; case: {args.case_index}", flush=True)
        if args.opponent == 'saved':
            print('Saved case may be a failed encounter. Use --opponent stationary to inspect basic pursuit.', flush=True)
        with tempfile.TemporaryDirectory(prefix='predator-replay-') as directory:
            replay_path = Path(directory) / 'replay.json'
            replay_path.write_text(json.dumps({'replay_encounter': case}), encoding='utf-8')
            subprocess.run([GODOT_PATH, '--path', PROJECT_PATH, '--', '--replay', str(replay_path)], check=True)
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
