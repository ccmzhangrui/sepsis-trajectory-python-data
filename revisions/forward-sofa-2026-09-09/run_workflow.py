"""Run unchanged analysis scripts in documented order; never download data."""
from pathlib import Path
import argparse
import os
import shlex
import subprocess
import sys


def commands(stage):
    source = [[p] for p in [
        'work/mimiciii/01_extract.py', 'work/mimiciii/02_score_sofa.py',
        'work/mimiciii/03_cohort.py', 'work/multicohort/prepare_mimiciii.py',
        'prepare_forward_inputs.py']]
    forward = [
        ['work/future_trajectory/run_forward.py', '--time', '12'],
        ['work/future_trajectory/run_forward.py', '--time', '24'],
        ['work/future_trajectory/run_forward.py', '--time', '12', '--mode', 'separated'],
        ['work/future_trajectory/run_forward.py', '--time', '24', '--mode', 'separated'],
        ['work/future_trajectory/verify_forward.py']]
    secondary = [[p] for p in [
        'work/mimiciii/04_fit_trajectories.py', 'work/mimiciii/05_features.py',
        'work/multicohort/rerun_mimiciii.py']]
    secondary += [
        ['work/multicohort/mortality_models.py', 'mimiciii'],
        ['work/reanalysis/run_analysis.py', '--input', os.environ.get('MIMICIV_XLSX', '${MIMICIV_XLSX}')],
        ['work/multicohort/eicu_extract.py'], ['work/multicohort/eicu_features.py'],
        ['work/multicohort/mortality_models.py', 'eicu']]
    secondary += [[p] for p in [
        'work/multicohort/aggregate_checks.py', 'work/multicohort/figure_inference.py',
        'work/multicohort/clinical_deterioration.py', 'work/multicohort/landmark_corrected_analysis.py',
        'work/multicohort/time_risk_analysis.py', 'work/reviewer_revision/verify_time_risk.py']]
    figures = [[p] for p in [
        'work/multicohort/plot_decimal_figures.py', 'work/multicohort/plot_time_risk.py',
        'work/future_trajectory/plot_forward.py']]
    return {'source': source, 'forward': forward, 'secondary': secondary, 'figures': figures}[stage]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['source','forward','secondary','figures'], required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if Path.cwd().resolve() != root:
        parser.error('Run from the published revision directory.')
    steps = commands(args.stage)
    for step in steps:
        if not (root / step[0]).is_file():
            parser.error(f'Missing source file: {step[0]}')
    if args.dry_run:
        for step in steps:
            print(shlex.join(['python'] + step))
        return
    required = ['MIMICIII_DIR'] if args.stage in ['source','forward'] else []
    if args.stage == 'secondary':
        required = ['MIMICIII_DIR', 'EICU_DIR', 'MIMICIV_XLSX']
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        parser.error('Set local authorized input paths: ' + ', '.join(missing))
    for name in required:
        p = Path(os.environ[name])
        valid = p.is_file() if name == 'MIMICIV_XLSX' else p.is_dir()
        if not valid:
            parser.error(f'Input location is unavailable: {name}')
    for name in ['work/mimiciii', 'work/multicohort', 'work/multicohort/mimiciii',
                 'work/reanalysis', 'work/reviewer_revision', 'work/future_trajectory']:
        parent = root / name
        for sub in ['private', 'results', 'figures']:
            folder = parent / sub
            folder.mkdir(parents=True, exist_ok=True)
            if sub == 'private':
                os.chmod(folder, 0o700)
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    for step in steps:
        print('Running ' + step[0], flush=True)
        subprocess.run([sys.executable] + step, check=True, cwd=root, env=env)


if __name__ == '__main__':
    main()
