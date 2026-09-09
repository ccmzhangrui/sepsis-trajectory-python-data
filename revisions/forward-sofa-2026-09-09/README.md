# Forward SOFA trajectory reanalysis: September 2026 Reply

This source-code snapshot accompanies the revised Reply concerning *Machine
learning predicts sepsis deterioration trajectories*. The main question is
forecasting **future SOFA trajectories at actual ICU hours 12 and 24**.
Hospital mortality and organ-support events are secondary analyses.

中文说明：本目录为本次回复信的分析代码。主分析在12或24小时当时发出预测，
仅使用此前信息预测随后SOFA轨迹。它使用重建的MIMIC-III队列；MIMIC-IV和eICU
仅用于各自单独训练的住院死亡次要分析。代码公开不代表原四队列或原17.6小时
预警结果已经复现。原仓库其他目录中的示例数据不是本次计算的输入。

## Scientific scope

- At 12 h: prefix SOFA grid 6/12 h; future grid 18/24/30/36/42/48/54/60 h.
- At 24 h: prefix grid 6/12/18/24 h; future grid 30/36/42/48/54/60 h.
- Later-target sensitivities start at 36 h and 48 h, respectively, to exclude
  pre-origin measurements from the rolling 24-hour future SOFA windows.
- Five patient folds and three inner folds separately fit prefix GBTMs,
  future-label GBTMs, preprocessing and classifiers within training partitions.
- Future labels require at least three observed future grid scores. Patients
  with unassessable future trajectories are not coded as negative outcomes.
- The posterior-addition comparator excludes SOFA slope. Separate slope and
  no-SOFA models assess shared information and incremental performance.

The executed analyses found little posterior gain beyond latest SOFA and did
not consistently recover a rising deterioration class. High future SOFA is not
automatically deterioration. Future class membership does not provide a unique
onset timestamp for the original 17.6-hour warning estimator. Conditional
bootstrap intervals omit full training uncertainty. Read the protocol and
Supplementary Information for the scientific limitations.

## Contents and access

This directory contains source code, protocols, environment information and
documentation only. No raw clinical records, patient identifiers, individual
predictions, data tables, trained model parameters or reviewer correspondence
are included. There is no automatic patient-data download.

The original analysis scripts are copied byte-for-byte from the executed
Supplementary Code archive; their SHA-256 digests are recorded in
`PUBLICATION_MANIFEST.json`. The two root helper scripts arrange execution and
prepare the same vital-history cache using the existing extraction function;
they do not change the statistical methods. They have been syntax and dry-run
checked, not used to repeat the complete clinical analyses during publication.

`work/future_trajectory` contains the principal analysis. `work/mimiciii`
contains source extraction and earlier intermediates. `work/multicohort`
contains corrected MIMIC-III processing, eICU processing, secondary outcomes,
statistics and plotting dependencies. `work/reanalysis` handles the supplied
MIMIC-IV workbook. `work/reviewer_revision` contains verification code only.

## Local environment and input contract

Use a private working copy and run commands **from this directory**, with the
Python and dependency versions recorded in `environment.json` and
`requirements.txt`. Dependencies must be installed in the reader's own local
environment. `requirements.txt` describes the recorded runtime, not a promise
of compatibility with arbitrary later versions.

Set `MIMICIII_DIR` to the local directory of the authorized MIMIC-III CSV exports.
The extraction script names its required files, including the supplied derived
tables. It reads `chartevents.csv` once; the numbered files are not concatenated.
`chartevents_1.csv` is used only for the source-overlap audit. The support analyses
also require the supplied ventilation and vasopressor duration CSV exports.
These derived exports are prerequisites; the code does not claim to regenerate
all device/pump intervals from raw records.

For secondary analyses, set `EICU_DIR` to the directory containing the authorized
`patient.csv`, `diagnosis.csv` and `nurseCharting.csv` exports, and set
`MIMICIV_XLSX` to the author-supplied workbook. The latter's original extraction
query, version and nominal Day1 ICU time anchor were not verified. A fresh generic
MIMIC-IV download cannot be assumed to reproduce that workbook. No Ruijin
clinical input is included or used by these scripts.

The fixed cohort assertions are intentional. A mismatched export or cohort must
be investigated; removing assertions to force agreement would change the study.
Data-use agreements and access permissions remain applicable to any local run.
See [DATA_ACCESS.md](DATA_ACCESS.md).

## Execution

Preview any stage first; `--dry-run` performs no analysis:

```bash
python run_workflow.py --stage source --dry-run
python run_workflow.py --stage source
python run_workflow.py --stage forward
```

`source` builds the local extraction database, rolling SOFA and operational
cohort, applies effective clinical exit, then builds cutoff-first vital histories.
`forward` runs the two main and two later-target configurations and independently
checks labels and numerical results. This is computational work using authorized
data; it is not a small demonstration.

To reproduce secondary outcomes and the full figure set after those stages:

```bash
python run_workflow.py --stage secondary
python run_workflow.py --stage figures
```

The secondary stage includes the earlier overlapping-window diagnostic and
legacy event sensitivity because figure scripts require their outputs. Those
legacy tasks do not define the current principal endpoint. The final corrected
MIMIC-III historical sensitivity outputs are under
`work/multicohort/mimiciii/results`, not `work/mimiciii/results`.

All generated outputs remain local and are ignored by the supplied `.gitignore`.
In particular, some fitted GBTM parameters are written to JSON files under
`results`; JSON is not automatically safe to publish. No generated folder should
be added to GitHub wholesale.

## Provenance and licensing

Four unchanged pivot SQL files originate from
[MIT-LCP/mimic-code, commit 303d26c623dcc9c49cc0f204468d4acc2f063797](https://github.com/MIT-LCP/mimic-code/tree/303d26c623dcc9c49cc0f204468d4acc2f063797/mimic-iii/concepts_duckdb).
Their MIT license and attribution are retained in
`THIRD_PARTY_LICENSES/MIT-LCP-mimic-code-LICENSE`.
This snapshot does not assign a new license to the authors' other code or grant
any rights to the clinical databases.

For an archived, citable publication version, a fixed commit can be deposited
in a DOI-minting repository. No Zenodo DOI or archive deposit is asserted here.
