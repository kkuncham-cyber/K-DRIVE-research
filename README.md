# K-DRIVE — two-arm data and decision experiment

**Author:** Keerthi Kuncham  
**Corresponding email:** kkuncham@midwest.edu

**Status:** Research-development manuscript, **not a validated business deployment**. This code regenerates the two figures and two tables in the accompanying manuscript. All findings and claims must be checked by the named author before journal submission.

## Study design

- **Observed survey arm:** 306 public anonymized respondent records derived from A. Clim (2026), Zenodo DOI [10.5281/zenodo.21235494](https://doi.org/10.5281/zenodo.21235494), source `data/public/public_anonymised_dataset.csv` in the author's public GitHub repository. This is a CC BY 4.0 research-data derivative with credit, **not firm-level longitudinal performance data**. 174 participants are students. The target is a respondent's stated intention to consider or definitely plan investment in AI within the next 1–2 years. Original survey categories are mapped to documented codes in `data/survey_public_projection.csv`; see `METHODS.md`. Measured outcome: reported intention, not future investing.
- **Simulation arm:** 6,000 researcher-generated enterprise examples (3,600 train, 1,200 validation, 1,200 test) plus 1,500 independent shifted cases. It tests how three action options, four policies and invented business costs behave in a controlled artificial environment. These are not real enterprises, and the action cost assumptions are not expert-validated.

## Run locally

Python 3.11+ recommended.

```bash
python -m venv .venv
# activate venv, then:
pip install -r requirements.txt
python empirical.py
python main.py
```

Both scripts use pre-specified random seeds and overwrite `outputs/` reproducibly. Version variations may affect low-order digits. No network access is required; the attributed 306-row public-survey projection is bundled.

## Input/output inventory

**Minimal source files:** `empirical.py`, `main.py`, `requirements.txt`, `README.md`, `data/survey_public_projection.csv`.

**Manuscript figure 1:** `outputs/figure_1_real_survey.png` and `.pdf`: held-out ROC/PR on *observed respondent intention*, n=77.

**Manuscript figure 2:** `outputs/figure_2_policy_validation.png` and `.pdf`: loss/response mix from 1,200 *simulated enterprises*.

**Research Table 1:** `outputs/survey_table_predictive.csv`, with numerical master `outputs/survey_results.json` and per-respondent probabilities `outputs/survey_holdout_predictions.csv`.

**Research Table 2:** `outputs/results.json`, with row-level synthetic decisions `outputs/test_predictions_and_decisions.csv`.

**Supplementary sensitivity:** `outputs/supp_penalty_sensitivity.csv`, `outputs/supp_threshold_sensitivity.csv` and independent generated shift metrics in `results.json`. `outputs/synthetic_enterprise_records.csv` and `outputs/stress_synthetic_records.csv` fully specify the generated example cases.

**Graphical abstract:** built programmatically as a standalone companion figure in the manuscript package (not one of the two research figures).

## Research integrity

1. The observational survey prediction and simulated intervention outcomes belong to **two different datasets**; no empirical transfer or treatment effect was shown.
2. Gradient boosting was selected for the observational survey via five-fold training-only Brier score; final 77 cases were held out. Brier results are very close to the random-forest comparator, and the 95% AUC interval is wide.
3. The safety-rule model wins under the chosen penalty **but loses its advantage at zero penalty**. Treat the loss improvement as conditional on model assumptions.
4. The work is not original surveillance/cybersecurity research from the user's earlier residency paper, nor does it reuse the professor's accepted UAV-paper results or figures.
5. The named author and corresponding email are confirmed; institutional affiliation, ethics approvals, journal submission and a public repository URL must not be represented as completed without evidence.

## Source attribution

Clim A. *AI in Business – EU contextual regulations and future workforce survey data: Public anonymised analysis snapshot and reproducibility package*, v2.1.2. Zenodo (2026). DOI 10.5281/zenodo.21235494. Data: CC BY 4.0. This repository's data projection reuses an anonymized subset of these published public records; all original categorical labels were encoded according to the accompanying methods appendix.

## Public repository notes

The two figures, the two main results tables, and their source scripts are provided here. Large synthetic case-generation CSVs and duplicate PDF/PNG figures are omitted from Git history; `python main.py` and `python empirical.py` recreate the full set of outputs. Do not infer that a source-code release represents journal submission, peer review, scientific validation, or deployment. This repository includes material derived from a public anonymized survey, credited above and documented in `METHODS.md`. Copyright in original researcher-generated code remains with its author; no general code reuse license is granted here.