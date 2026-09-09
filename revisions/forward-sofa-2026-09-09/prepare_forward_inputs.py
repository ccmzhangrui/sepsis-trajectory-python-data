"""Build the same cutoff-first vital cache without fitting secondary models.

Publication helper: statistical/extraction functions are imported unchanged.
"""
from pathlib import Path
import os
import sys


def main():
    import pandas as pd
    root = Path(__file__).resolve().parent
    if Path.cwd().resolve() != root:
        raise SystemExit('Run from the published revision directory.')
    sys.path.insert(0, str(root / 'work/multicohort'))
    from landmark_corrected_analysis import build_vital_hourly
    cohort = pd.read_pickle('work/multicohort/mimiciii/private/cohort.pkl')
    cohort = cohort.set_index('icustay_id').sort_values('subject_id')
    assert len(cohort) == 4856 and cohort.subject_id.is_unique
    folder = root / 'work/multicohort/private'
    folder.mkdir(parents=True, exist_ok=True)
    os.chmod(folder, 0o700)
    output = folder / 'time_risk_vitals.pkl'
    if output.exists():
        raise SystemExit('Vital cache already exists. Use a fresh private working copy to rebuild it.')
    build_vital_hourly(cohort.index).to_pickle(output)
    os.chmod(output, 0o600)
    print('Cutoff-first vital cache written to the local private directory.')


if __name__ == '__main__':
    main()
