# When Simulation-Based Sample Supplementation Fails in Cross-Domain Bearing Fault Diagnosis

**The Recording-Fingerprint Effect and an Evaluation Protocol That Reveals It**
Fangyu Zhang, Julong He

This repository holds the code, the per-table artifacts and the LaTeX sources behind the
paper. Its point is not to publish a method but to make a negative result checkable: on the
CWRU cross-load benchmark the class label is carried by the individual recording rather than
by the fault physics, so simulation-based sample supplementation cannot work there, and the
standard few-shot protocol hides this by letting support and test windows share a recording.

Everything the paper claims numerically is recomputed here from stored artifacts or from the
raw signals, and `tools/verify_paper_numbers.py` fails loudly on any mismatch.

---

## 1. Check the numbers: no dataset download needed

```bash
python tools/verify_paper_numbers.py artifacts
```

Reads every artifact in `results_paper/`, `results_pb/` and `results/final/`, recomputes the
quantity each table reports, compares it with the value printed in the manuscript, and prints
one line per check. **318 checks**; exit code 0 means all agree, exit code 1 means a mismatch
and the offending rows are printed.

Requirements: Python 3 with numpy and scipy (`requirements.txt`).

## 2. The remaining checks need the raw data

```bash
python tools/verify_paper_numbers.py raw     # 40 further checks
python tools/verify_paper_numbers.py all     # all 360
```

`raw` recomputes the deterministic statistics of Table 2, Table 4, the "Diagnosis II" section
and the bench-tone paragraph directly from the signals. The datasets are public and are **not**
redistributed here (Paderborn alone is ~15 GB).

### CWRU 12 kHz drive end

```bash
python tools/prepare_cwru.py            # downloads the 40 official files into data/cwru/
python tools/prepare_cwru.py --verify   # checks an existing data/cwru/ file by file
```

**One warning, and it matters.** CWRU's `99.mat` (the 2 hp normal baseline) contains *two*
recordings in a single file: `X098` (1 hp) and `X099` (2 hp), with `X098` first. Any loader
that takes "the first DE channel" silently assigns the load-2 NORMAL class the **1 hp**
recording — a byte-for-byte duplicate of `Normal_1.mat`. We hit exactly that during this work:
the CWRU shaft rates, recovered from the recordings themselves, are 1795.5 rpm for `X097` and
1772 / 1750 rpm for `X098` / `X099`, which identifies the two channels unambiguously.
`tools/prepare_cwru.py` picks `X099` explicitly for that file, and `data_utils.load_de_channel`
now refuses any file with more than one DE channel rather than guessing.

### Paderborn (KAt)

22 bearings, `data/paderborn/<code>/<code>/<condition>_<code>_<index>.mat`, downloaded from
`https://groups.uni-paderborn.de/kat/BearingDataCenter/<code>.rar`. Three pitfalls we hit, all
of which change conclusions if missed:

- the class definition used here (`pb_utils.CLASSES_LOC`) is by *damage location* (healthy /
  inner race / outer race), parsed from the fact sheet PDF shipped with each bearing. `KA08`'s
  fact sheet writes the location as `AR`, which is not in its own legend — a typo in the source
  data, so that bearing is excluded;
- at 64 kHz a 1024-point window is only 16 ms — about one BPFO period at 1500 rpm (76 Hz,
  13 ms), so a window rarely holds more than one impact. Every bearing then looks spectrally
  identical (pairwise cosine ≈ 1.000) and any spectral diagnosis fails. **Use a 4096-point
  window here** (about five periods);
- in the subset downloaded for this paper every artificial-damage bearing is outer-ring and
  every run-to-failure bearing is inner-ring, so damage location and damage origin are
  confounded. The paper's spectral claim is an absence of class structure and is unaffected,
  but it supports no separate statement about outer versus inner race.

## 3. What the checks establish, and what they do not

Two different kinds of claim; they should not be conflated.

| | how it is verified | checks |
|---|---|---|
| Tables 2 and 4, "Diagnosis II", bench tone | recomputed from the raw signals — deterministic given the data | 40 |
| Tables 1, 3, 5, 6, 7, 8, 9 | the manuscript agrees with the stored artifacts | 318 |

For the second group the check establishes **manuscript ↔ artifact** agreement, not
**artifact ↔ fresh run**. Those tables come from training, and this hardware is not run-to-run
deterministic: we measured the same command producing 10-seed means up to 0.033 apart, and a
3-seed standard deviation wrong by more than an order of magnitude (0.005 measured against
0.077 measured properly). The paper says so, and it is why the secondary tables are reported
over `n_repeat × n_seed` runs rather than over three seeds. Re-running the training scripts on
other hardware will give values close to, but not identical to, the printed ones.

## 4. Which artifact backs which table

Every numbered table has an on-disk artifact and a computing function in
`tools/verify_paper_numbers.py`.

| Manuscript table | Artifact | Producing command | Verified by |
|---|---|---|---|
| Table 1 `tab:synth` | `results_paper/tab_synth.json` (merged from `tab_synth_10a.json`, `tab_synth_10b.json`; n_repeat=2 × 10 seeds) | `python tools/probe_fp.py 3 5 400 40 10 2` | `table_synth` |
| Table 2 `tab:classsep` | none — recomputed from `data/cwru` | `python tools/probe_classsep.py` | `table_classsep` (raw) |
| Table 3 `tab:leak` | `results_paper/tab_leak.json` (n_repeat=2 × 10 seeds; the superseded 6-run artifact is kept as `tab_leak_n6legacy.json`) | `python tools/probe_leakage.py all 60 10 5 2 --suffix=_s10`, then copy `tab_leak_s10.json` to `tab_leak.json` | `table_leak` |
| Table 4 `tab:bench` | `results_pb/spectra_bench.json` (recomputed anyway) | `python tools/probe_spectra.py` | `table_bench` (raw) |
| Table 5 `tab:pb_leak` | `results_pb/pb_levels.json` (10 seeds, 200 epochs), plus `results_pb/pb_split_coverage.json` for the bearing coverage quoted in the text (15 of 21, `python tools/pb_split_coverage.py 10`) | `python tools/probe_pb_levels.py 200 10` | `table_pb_levels` |
| Table 6 `tab:pb_levels` | `results_pb/pb_levels.json`, plus `results_pb/pb_levels_notch.json` and `pb_levels_highpass.json` for the two filtered columns (`pb_levels.log`, `pb_levels_notch.log`, `pb_levels_highpass.log`) | `python tools/probe_pb_levels.py 200 10`; add `--filter=notch` and `--filter=highpass` for the last two columns | `table_pb_levels`, `table_pb_filter` |
| Table 7 `tab:pb_fewshot` | `results_pb/d_twin_k5.json`, `d_twin_k20.json`, `d_twin_k50.json` (10 seeds each; the superseded 5/3-seed runs are kept as `*_n5legacy`/`*_n3legacy`) | `python tools/probe_paderborn.py d --k=5,20,50 --seeds=10 --suffix=_s10`, then copy the `_s10` files to the names in column 2 | `table_pb_fewshot` |
| Table 8 `tab:main` | `results/final/*.csv` (7 methods × 5 seeds) | `bash run_s2.sh all` then `python tools/collect_results.py` | `table_main` |
| Table 9 `tab:fewshot` | `results_paper/tab_fewshot.json` (n_repeat=2 × 10 seeds; the superseded 3-seed run is kept as `tab_fewshot_n6legacy.json`) | `python tools/probe_fewshot.py 3 0 40 10 2 --suffix=_s10`, then copy `tab_fewshot_s10.json` to `tab_fewshot.json` | `table_fewshot` |
| Fig. 1 `fig:leakage` | `paper/figs/fig_leakage.pdf` | `python tools/make_figures.py` | the same numbers as Table 3 |
| Fig. 2 `fig:spectra` | `paper/figs/fig_spectra.pdf` | `python tools/make_figures.py` | the same numbers as "Diagnosis II" |
| Backbone check | `results_pb/pb_levels_smallresnet.json` | `python tools/probe_pb_backbone.py 200 10 --arch=smallresnet` | `table_pb_levels` |
| Fig. 3 `fig:pb_levels` | `paper/figs/fig_pb_levels.pdf` | `python tools/make_figures.py` | the same numbers as Tables 5 and 6 |

Notes.

- Table 7's three rows each average 10 seeds (see the array lengths in the JSON files). The
  earlier 5/3-seed run that the manuscript used before 2026-10-27 is kept as
  `d_twin_k5_n5legacy.json`, `d_twin_k20_n3legacy.json` and `d_twin_k50_n3legacy.json`; it is
  not the source of any number in the paper, and its k=20/50 rows are systematically
  optimistic (that is why the table was extended).
- The taxonomy check in Section V uses `results_paper/tab_leak_c4s10.json` (four
  classes, three recordings per class, ten seeds) and `results_paper/tab_leak_c4single.json`
  (four classes at one fault size, ten seeds); each artifact records its command in
  `provenance`, and both are read by `table_leak_taxonomy` in the verification script.
- Table 3's cells each average 20 runs (10 seeds, repeated twice). The earlier 6-run
  artifact used before 2026-10-31 is kept as `results_paper/tab_leak_n6legacy.json`; it is not
  the source of any number in the paper. Its load-3 fingerprint row was the outlier that made
  the earlier text speak of an "only sometimes" benefit; at 20 runs the same effect is
  +0.099 and the loads 0--2 rows are stable at --0.02 to +0.00.
- Table 9's cells each average 20 runs (10 seeds, repeated twice). The earlier 3-seed run
  used before 2026-10-30 is kept as `results_paper/tab_fewshot_n6legacy.json`; it is not the
  source of any number in the paper, and its support-only row is optimistic by 0.05 at k=1
  (seeds 0--2 are an easy draw).
- Tables 5 and 6 come from one run by construction: the same training set is evaluated against
  three test sets that differ by one factor each, which is what makes the session/specimen
  decomposition meaningful.
- One of the ten unfiltered runs (seed 6) failed to converge and is *kept* in Table 5; its
  training-set macro-F1 is 0.60 against 0.98--1.00 for the others. The same configuration
  re-run converges (`results_pb/pb_levels_s6rerun.json`, produced with
  `python tools/probe_pb_levels.py 200 1 --seed_start=6 --suffix=s6rerun`), which is how we know
  the failure is transient rather than a property of the split. The text says so.
- The notch and high-pass columns of Table 6 are separate runs of the same script with
  `--filter=notch` and `--filter=highpass`, which mask the window in the frequency domain and
  renormalise it before training. They are comparable seed by seed with the unfiltered column
  because every run draws its split from `RandomState(seed)` with the same code path.
- The merged files (`tab_synth.json`, `tab_leak.json`) were produced either by the runner's
  `n_repeat` argument or by `python tools/merge_runs.py out.json runA.json runB.json`, which
  refuses to merge runs whose configurations differ. The per-run inputs are kept alongside them.
- `results_paper/tab_synth_run1.json`, `tab_synth_10a.json`, `tab_synth_10b.json` and
  `tab_synth_pooled.json` are the provenance chain of the merged Table 1 artifact. For Table 3
  the current artifact comes from a single ten-seed, two-repeat invocation; its console
  transcript is `results_paper/tab_leak_s10.log` (the earlier 6-run transcript is
  `tab_leak.log`) and the command is recorded in the artifact's own `provenance` field.

## 5. Layout

```
tools/            analysis code; entry point is verify_paper_numbers.py
results_paper/    artifacts for Tables 1, 3, 9 (+ the per-run files they merge)
results_pb/       artifacts for Tables 4, 5, 6, 7 and the cross-benchmark spectra
results/final/    per-run CSVs for Table 8 (7 methods x 5 seeds)
paper/            LaTeX sources, bibliography, figures, compiled PDF
data_utils.py     CWRU reading, windowing, few-shot split, cache
twin_calib.py     spectral-envelope calibration used by the constructive method
dt_synth.py       physics twin (McFadden-Smith impulse model), kept as the negative control
models.py         ResNet1D backbone, classifier, domain discriminator, gradient reversal
train.py          training entry point for every method
pb_utils.py       Paderborn reading pipeline and damage-location parsing
run_s2.sh/.bat    CWRU load0 -> load3 runs (smoke / all / single method)
install.bat       Windows one-shot environment setup (torch cu128 for RTX 50 series)
```

## 6. Reproducing the training tables

```bash
bash run_s2.sh smoke     # 2 epochs per method, checks the pipeline
bash run_s2.sh all       # all methods, CWRU 10-class S0 -> S3
```

Method names used by `train.py --method`: `source_only`, `dann`, `dt_dann`, `rand_dann`,
`fp_dann`, `fp_phys_dann`. Other settings: `--classes 4|10`, `--k`, `--seed`, `--epochs`,
`--window`, `--synth_per_class`, `--data_root`.

## 7. Environment

Windows 11, Python 3.11.1, numpy 2.4.6, scipy 1.17.1, matplotlib 3.11.2, torch 2.11.0 (CUDA).
Training tables used one consumer GPU; no GPU-specific code paths are involved. On an RTX 50
series card, torch must be built for CUDA 12.8 or later (`sm_120`); `install.bat` does this.

## 8. Building the manuscript

```bash
python tools/to_elsarticle.py     # IEEEtran source -> Elsevier elsarticle version
python tools/check_submission.py  # compiles both and checks abstract length, keywords,
                                  # highlights, overfull boxes, undefined references
```

`paper/manuscript.tex` (IEEEtran) is the only hand-edited source; `paper/manuscript_mssp.tex`
(elsarticle) is generated from it, and the converter verifies that the body text is identical
modulo its mechanical substitutions.
