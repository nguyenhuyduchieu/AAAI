# GEF Paper — Implementation & Writing Plan

**Paper:** What Do Time Series Foundation Models Actually Learn? A Geometric Explainability Framework  
**Venue:** AAAI-26  
**Status:** Experiments not yet run. LaTeX skeleton complete at `paper/main.tex`.

Legend: 🔴 blocker · 🟡 can parallelise · 🟢 independent of experiments

---

## Phase A — Fix Blockers
> Must complete in order before anything else. Every downstream metric is invalid until Task 1 is done.

### Task 1 — Fix LCS labels: replace `% 8` with STL-derived trend labels 🔴
**File:** `src/eval/metrics.py`  
**Why:** The current `phase_labels = (np.arange(...) % 8)` are arbitrary cyclic indices with no semantic meaning. Every LCS number produced by the current code is methodologically invalid and will be caught by any reviewer who reads the implementation.  
**What to do:**
- Apply STL decomposition to each window using `statsmodels.tsa.seasonal.STL`
- Extract the trend component and compute OLS slope `beta_i`
- Assign labels: `up` if `beta_i > +0.05 * sigma_w`, `down` if `beta_i < -0.05 * sigma_w`, `flat` otherwise
- Replace `phase_labels` everywhere it appears in `metrics.py` and `run_pipeline.py`

**Success criterion:** `compute_lcs()` accepts STL-derived 3-class labels; silhouette score is computed against semantically meaningful clusters.

---

### Task 2 — Scale window sampling from N=256 to N=3000 🔴
**Files:** `notebooks/timesfm_latent_demo.ipynb`, `scripts/run_pipeline.py`  
**Why:** TwoNN intrinsic dimensionality requires N ≥ 1000 for stable estimates (Facco et al. 2017). At N=256 with k=10, the k-NN graph covers 3.9% of all points per node — nearly fully connected — making geodesic distances trivial and uninterpretable.  
**What to do:**
- Set `MAX_WINDOWS = 3000`, `stride = 64`, `context_length = 512`
- Verify memory fits on GPU for all 4 datasets (batch extraction if needed)
- Update notebook config cell and pipeline `--max-windows` default

**Success criterion:** All datasets produce embeddings of shape `[3000, D]` before any geometric analysis runs.

---

## Phase B — Implement Missing Experiment Modules
> Tasks 3–8 are independent of each other and can be worked in parallel. All blocked by Task 1.

### Task 3 — Implement `src/eval/id_twonn.py` — TwoNN intrinsic dimensionality 🟡
**Why:** Layer-wise ID is the paper's primary mechanistic finding. It shows whether TSFMs progressively compress representations into a low-dimensional manifold — the geometric "smoking gun" that supervised baselines lack.  
**What to do:**
- Implement TwoNN estimator: `id = -N / sum(ln(mu_i))` where `mu_i = dist_2nd / dist_1st`
- Function `compute_id(embeddings: np.ndarray) -> float`
- Function `compute_id_profile(layer_embeddings: np.ndarray) -> dict` — takes `[n_layers, N, D]`, returns `{layer_ids, peak_layer, terminal_id, compression_ratio}`
- Use `scikit-dimension` (`skdim.id.TwoNN`) if installed; provide manual fallback
- Handle edge cases: duplicate points, very small N

**Success criterion:** Returns stable ID estimates (variance < 10% across 5 random subsamples of same embeddings) for N=3000, D=1280.

---

### Task 4 — Implement `src/eval/geodesic.py` — geodesic distance + Spearman correlation 🟡
**Why:** The geodesic-forecast error Spearman ρ is the paper's headline XAI result and the primary number in the abstract. Without it there is no paper.  
**What to do:**
- `build_knn_graph(embeddings, k=10)` → scipy sparse distance matrix using `sklearn.neighbors.NearestNeighbors`
- `compute_geodesic_distances(graph)` → dense matrix via `scipy.sparse.csgraph.shortest_path(method='D')`
- Check for disconnected graph (infinite distances); warn and increase k automatically if found
- `compute_geodesic_forecast_correlation(embeddings, forecast_errors, k=10)`:
  - For each window i, find nearest manifold neighbour j* = argmin d_G(i,j)
  - Compute Spearman ρ between `d_G(i, j*)` and `|e_i - e_{j*}|`
  - Return `{rho, p_value, ci_lower, ci_upper}` (Fisher z-transform CI)
- `get_failure_cases(d_geo, errors, q_geo=0.25, q_err=0.75)` → indices where geodesic distance is in lowest quartile but error is in highest quartile

**Success criterion:** Returns finite ρ with p-value < 0.05 on at least one dataset for pretrained TimesFM. Returns `nan` (not crash) for random-weight control if graph is degenerate.

---

### Task 5 — Implement `src/eval/probe.py` — linear probing for temporal primitives 🟡
**Why:** Probing classifiers establish that geometric clusters encode interpretable temporal semantics (trend, volatility, seasonality) — not just arbitrary structure. This is the "what do they actually learn" answer.  
**What to do:**
- `derive_labels(windows: np.ndarray) -> dict` — three binary label arrays derived independently from any model:
  - **Trend:** OLS slope `beta` on window; positive if `|beta| > 0.05 * sigma_w`
  - **High volatility:** window std vs global median; positive if `sigma_w > median(all sigma)`
  - **Seasonality:** dominant FFT frequency in `[1/48, 1/6]` hours; positive if peak power > 2× mean spectral power
- `probe_layer(embeddings, labels, C=1.0, n_seeds=5)`:
  - Fit L2 logistic regression on 70/30 stratified split, repeated 5 times
  - Return `{balanced_accuracy_mean, balanced_accuracy_ci, auroc_mean, auroc_ci}`
- `probe_all_layers(layer_embeddings, labels)` → profile across `[embed, L/4, L/2, L_max]`
- `permutation_test(embeddings, labels, n_perm=1000)` → p-value vs stratified chance baseline

**Success criterion:** Random-weight TimesFM returns balanced accuracy ≈ 0.50 (chance) for all three primitives. This validates the control.

---

### Task 6 — Implement `src/eval/failure.py` — failure case characterisation 🟡
**Why:** AAAI-26 guidelines explicitly require analysis of cases where the proposed approach fails. A paper without failure analysis risks desk rejection. This section also provides the falsifiable boundary conditions for GEF's reliability signal.  
**What to do:**
- Accept failure case indices from `geodesic.get_failure_cases()`
- For each failure window compute:
  - **Structural break:** PELT change-point detection via `ruptures` library; return number of detected breakpoints
  - **Distributional shift:** MMD with RBF kernel (`sigma = median heuristic`) between failure window and training window distribution
  - **Spectral complexity:** FFT spectral entropy `H = -sum(p_k * log(p_k))` where `p_k` are normalised power spectral densities
- Return `pd.DataFrame` with one row per failure case and columns `[window_idx, n_breakpoints, mmd_score, spectral_entropy]`
- Comparison function: compute same three metrics for non-failure windows to enable statistical contrast

**Success criterion:** Failure windows show statistically higher scores on at least one of the three characterisation axes versus non-failure windows (Mann-Whitney U test, p < 0.05).

---

### Task 7 — Implement `src/models/tsfm_chronos.py` — Chronos-T5 extractor 🟡
**Why:** Two TSFMs are needed to show findings are not TimesFM-specific. Chronos-T5 is already cloned under `external/chronos-forecasting` and uses a different architecture (T5 encoder-decoder vs decoder-only), making it the right architectural contrast.  
**What to do:**
- Use `ChronosPipeline.from_pretrained('amazon/chronos-t5-large')` from the cloned repo
- Extract encoder hidden states via `output_hidden_states=True` in the underlying T5 model
- Document Chronos tokenisation: it quantises values into bins before embedding — windows must be passed as raw float tensors, not pre-normalised
- Match the output dict interface: `{token_embeddings, sequence_embeddings, layer_embeddings}`
- Add `ChronosLatentExtractor` class following the same API as `TimesFMLatentExtractor`

**Success criterion:** Produces `layer_embeddings` array of shape `[n_layers, B, T, D]` for a batch of windows without errors on the Electricity dataset.

---

### Task 8 — Add random-weight TimesFM control to pipeline 🟡
**Why:** Without this control, a reviewer can argue that all observed geometric structure comes from the patch tokeniser or Transformer architecture, not from pretraining. This is a fatal objection. The control is one additional forward pass and costs almost nothing.  
**What to do:**
- Add `--random-weight` flag to `scripts/run_pipeline.py`
- When set: load TimesFM config, instantiate model with `AutoModel.from_config(config)` (random init, no pretrained weights)
- Run the identical extraction and metric pipeline on the random-weight model
- Compute `delta_lcs = lcs_pretrained - lcs_random` for each dataset
- Report alongside pretrained results in all tables

**Success criterion:** `delta_lcs > 0.10` across datasets (criterion from the experimental claims table in `main.tex`). If delta is small, the paper's scope must shift to "patch tokenisers impose structure" — document this outcome regardless.

---

## Phase C — Supporting Infrastructure
> Can be built in parallel with Phase B.

### Task 9 — Implement `scripts/run_ablations.py` — four ablation suites 🟡
**Why:** Four ablations defend against specific reviewer objections identified in the AAAI-26 critical evaluation: pooling choice, geodesic k sensitivity, LCS threshold sensitivity, and baseline data-scale fairness.  
**What to do:**
1. **Pooling ablation** (on Electricity): recompute LCS, TSI, terminal ID under mean-pool, last-token, max-pool. Confirm findings hold across all three.
2. **Geodesic k-sensitivity** (all 4 datasets): recompute Spearman ρ for `k ∈ {5, 10, 15, 20}`. A robust finding stays positive and significant across all k.
3. **LCS threshold sweep** (all 4 datasets): recompute LCS with STL slope thresholds `{0.02, 0.05, 0.10} × sigma_w`. Report sensitivity of TSFM vs baseline gap to threshold choice.
4. **Baseline data-scale** (Transformer on Electricity): train at `{10%, 25%, 50%, 100%}` of training data; track terminal ID and LCS. Tests whether geometric gap is explained by training data volume alone.

Save all outputs to `outputs/metrics/ablations/`. Each ablation gets its own CSV.

**Success criterion:** Core findings (TSFM LCS > baseline, positive geodesic ρ) replicate across all ablation conditions. If not, document which conditions break the finding.

---

### Task 10 — Add bootstrap CI and Wilcoxon significance testing to all metrics 🟡
**Files:** `src/eval/metrics.py`, `scripts/run_pipeline.py`  
**Why:** The current code returns point estimates only. AAAI-26 empirical standards require confidence intervals. Without them, reviewers cannot assess whether differences are statistically meaningful.  
**What to do:**
- `bootstrap_ci(metric_fn, *args, n_bootstrap=2000, ci=0.95)` → `(mean, lower, upper)` using stratified resampling over windows
- `wilcoxon_compare(tsfm_scores, baseline_scores)` → p-value; apply Holm–Bonferroni correction when comparing across 4 datasets × 3 metrics (12 simultaneous tests)
- Update `run_pipeline.py` to collect per-dataset metric vectors (not just per-run point estimates) and call both functions
- All tables in `main.tex` must show `mean ± CI` format; update LaTeX table templates accordingly

**Success criterion:** Every reported metric in the paper has a 95% CI and a corrected p-value for the TSFM vs best-baseline comparison.

---

### Task 11 — Add new visualization functions to `src/latents/visualize.py` 🟡
**Why:** Four new experiment types each need a dedicated figure. The existing `visualize.py` only covers UMAP scatter and perturbation bars.  
**What to do:**
1. `save_id_profile(id_per_layer_per_model, out_file)` — line plot of intrinsic dimensionality vs transformer layer depth, all models on one axes, with shaded CI bands
2. `save_geodesic_scatter(d_geo, delta_error, out_file)` — scatter plot of `d_G(i,j*)` vs `|e_i - e_{j*}|` with LOESS smoother; separate panels per dataset
3. `save_probing_heatmap(accuracy_grid, out_file)` — heatmap of probe balanced accuracy at `{embed, L/4, L/2, L_max}` × `{trend, volatility, seasonality}`, annotated with values
4. `save_failure_violins(failure_df, nonfailure_df, out_file)` — paired violin plots of PELT/MMD/spectral entropy for failure vs non-failure windows

All plots must use matplotlib with 300 DPI output and publication-appropriate font sizes (≥9pt axis labels).

**Success criterion:** All four functions produce clean figures that can be inserted directly into the LaTeX without editing.

---

## Phase D — Run Experiments
> Blocked by all of Phase B and C. Expected GPU time: several hours.

### Task 12 — Run full experiment pipeline across all models and datasets 🔴
**Command:** `python scripts/run_pipeline.py` with each combination  
**Scope:** `{TimesFM-pretrained, TimesFM-random, Chronos-T5, LSTM, TCN, Transformer}` × `{Electricity, ETTh1, Weather, METR-LA}` × `H ∈ {96, 192, 336, 720}`  
**What to collect per run:**
- Window embeddings `[3000, D]` from final layer
- Full layer embeddings `[n_layers, 3000, D]` for ID and probing profiles
- LCS, TSI, GSP with STL labels + bootstrap CI
- TwoNN ID profile across all layers
- Spearman ρ geodesic-forecast correlation + CI + p-value at each horizon
- Probing accuracy at 4 layer depths × 3 primitives
- Failure case characterisation (PELT, MMD, spectral entropy)
- Perturbation bar chart data (noise, dropout, timewarp)

Save all CSVs to `outputs/metrics/` with naming `{model}_{dataset}_{horizon}.csv`.  
Log all runs to `outputs/logs/`.

**Success criterion:** All 6 × 4 = 24 model-dataset combinations complete without error. No NaN values in metric outputs (investigate and fix any that appear).

---

### Task 13 — Run ablation experiments and verify robustness
**Command:** `python scripts/run_ablations.py`  
**Blocked by:** Task 12 (need baseline results to compare ablations against)  
**What to do:**
- Execute all four ablation suites defined in Task 9
- Verify that core findings (TSFM LCS > baseline, ρ > 0.40, terminal ID compression) replicate across all ablation conditions
- If any ablation breaks a finding, document it honestly — this becomes a limitation, not a reason to hide the result

**Success criterion:** Findings are robust across pooling strategies, k values, and LCS thresholds. Data-scale ablation shows Transformer geometric regularity does not converge to TSFM level even at 100% data (or documents that it does, which changes the paper's framing).

---

## Phase E — Fill the Paper
> Tasks 15 and 16 are independent and can be started immediately. Tasks 17–19 are blocked by Task 14.

### Task 14 — Fill all `\RESULT{}` macros in `main.tex` with real numbers 🔴
**Blocked by:** Tasks 12 and 13  
**What to do:**
- Search `main.tex` for every `\RESULT{description}{placeholder}` — there are currently 4 in the abstract alone
- Replace with actual experimental values:
  - Spearman ρ + CI for geodesic-forecast correlation (headline abstract number)
  - Probing balanced accuracy for TSFM vs supervised baseline (abstract)
  - Terminal ID comparison across models (Section 4.3)
  - Full LCS/TSI/GSP table (Section 4.4)
- Do not estimate or approximate. If an experiment has not run, it does not get filled.

**Success criterion:** Zero `\RESULT{}` tokens remaining in `main.tex`. Document compiles to clean PDF.

---

### Task 15 — Write Section 2: Background and Related Work 🟢
**Independent of experiments — can start immediately**  
**Target length:** ~0.75 page  
**Structure:**
1. TSFM taxonomy — decoder-only (TimesFM), encoder-decoder (Chronos), patch-based (MOMENT); cite Das 2024, Ansari 2024, Goswami 2024
2. The sceptical position — Zeng et al. 2023 showed linear models match transformers on standard benchmarks; frame this as motivation for geometric analysis rather than performance claims
3. Geometric XAI precedents in NLP/vision — Ansuini 2019 (ID in CNNs), Reif 2019 (BERT geometry), Birdal 2021 (ID explains fine-tuning); establish that no equivalent analysis exists for TSFMs
4. TwoNN estimator — Facco 2017; brief mathematical description
5. Linear probing — Alain & Bengio 2017
6. Takens as motivating intuition — flag explicitly as analogy, not as theoretical foundation (the theorem's conditions are not satisfied by general heterogeneous time series)

**Success criterion:** Section cites and directly engages with Zeng et al. 2023. Does not claim TSFMs are universally better than simpler models — this would contradict the paper it cites.

---

### Task 16 — Write Section 3: Geometric Explainability Framework 🟢
**Independent of experiments — can start immediately**  
**Target length:** ~1 page  
**Structure:**
1. **Definition 1 — Temporal Representation Manifold (TRM):** the image of the TSFM encoder `f_theta: R^L → R^D` restricted to a given temporal distribution, approximated empirically by the set `{z_i}_{i=1}^N`. Keep this precise but not overreaching — it is an empirical approximation, not a theorem about the model.
2. **Definition 2 — Geometric regularity metrics:** formal restatement of the LCS, TSI, GSP equations from Section 4 (move equation definitions here, reference them in Section 4).
3. **Proposition 1 — Consistency bound:** if `TSI_P >= 1 - eps`, then for any two windows `(w_i, w_j)`, `E[|e_i - e_j|] <= g(eps, d_G(i, j))` for some monotone function g. Frame as empirically motivated conjecture tested in Section 4.5, not a full theorem. Acknowledge the proposition is not proven from first principles.
4. **GEF pipeline overview:** a figure or TikZ diagram showing the three-lens analysis flow (extraction → ID profiling → regularity metrics → geodesic correlation).

**Success criterion:** Definition 1 is formally stated. Proposition 1 is clearly labelled as a conjecture with "we test this empirically in Section 4.5." No causal claims in the formal statements.

---

### Task 17 — Write Section 5: Discussion
**Blocked by:** Task 14 (needs real numbers to interpret)  
**Target length:** ~0.75 page  
**Structure:**
1. **What the geometry tells us about TSFM pretraining:** interpret the ID compression profile and LCS separation results. What does a low terminal ID mean about what the model has learned to ignore? What does high LCS mean about how pretraining organises representations?
2. **When geometry fails:** synthesise the failure analysis results from Section 4.8. State as concrete, falsifiable conditions: "The geodesic reliability signal should not be trusted when [condition from PELT/MMD/spectral entropy analysis]."
3. **Practitioner guidance:** concretely, how to use GEF at deployment time. Compute embeddings for new series. Check geodesic distance to training manifold. If distance falls outside the reliable zone identified in the failure analysis, treat the forecast with scepticism.

**Success criterion:** The failure boundary is stated as a testable condition, not vague hedging. The practitioner guidance is actionable in ≤ 3 steps.

---

### Task 18 — Write Section 6: Conclusion and complete mandatory AAAI-26 statements
**Blocked by:** Tasks 14, 15, 16, 17  
**What to do:**
- **Conclusion (~0.3 page):** summarise what was found (not what was done). Restate the two core claims: geometric structure of TSFM latent space; reliability signal from geodesic proximity. Acknowledge explicitly that findings are observational across two architectures and four datasets.
- **Author Contributions (CRediT):** assign standard CRediT taxonomy roles to each author
- **Funding Acknowledgment:** fill institutional grant details
- **Data Availability:** confirm all datasets and code are public; replace `[anonymous repository]` with actual repo URL after review period
- **Ethics Declaration:** already drafted — verify it is complete
- **Conflict of Interest:** already drafted — verify it is complete

**Success criterion:** All six mandatory AAAI-26 statements are present and complete. Conclusion does not introduce new claims.

---

## Phase F — Final Submission Check

### Task 19 — AAAI-26 compliance check and submission package
**Blocked by:** All previous tasks  
**Checklist:**

| Check | Criterion |
|-------|-----------|
| Page count | ≤ 7 pages content + 1 page references (compile with `pdflatex` and measure) |
| `\RESULT{}` macros | Zero remaining — search with `grep RESULT main.tex` |
| Citation completeness | Every `\cite{}` key exists in `refs.bib`; every `refs.bib` entry has a DOI if one exists |
| Table captions | One-sentence finding summary **below** each table, not just a title above |
| Abstract language | No causal "explains" — must say "correlates with" or "characterises" |
| Limitations paragraph | Present in Section 5; covers: 2 architectures only, 4 datasets, observational correlation, Takens analogy scope |
| Zeng et al. 2023 citation | Present in Section 1 or 2 with direct engagement |
| Double-blind compliance | No author names, institution names, or identifying repo URLs in main body or bibliography |
| Figure resolution | All figures ≥ 300 DPI when rendered in PDF |
| Random-weight control | Appears in at least one main table and in the abstract/intro as a validity check |

**Success criterion:** All 10 checklist items pass. Clean PDF compiles from `pdflatex main.tex` without errors or warnings about missing references.

---

## Dependency Map

```
Task 1 (fix LCS labels)  ◄── START HERE
  │
  └── Task 2 (scale N=3000)
        │
        ├── Task 3 (TwoNN id_twonn.py)      ─┐
        ├── Task 4 (geodesic.py)             ─┤
        ├── Task 5 (probe.py)                ─┤
        ├── Task 6 (failure.py)              ─┤── all parallel
        ├── Task 7 (Chronos extractor)       ─┤
        ├── Task 8 (random-weight control)   ─┤
        ├── Task 9 (run_ablations.py)        ─┤
        ├── Task 10 (bootstrap CI + Wilcoxon)─┤
        └── Task 11 (visualize.py additions) ─┘
              │
              └── Task 12 (full pipeline run)
                    │
                    └── Task 13 (ablation run)
                          │
                          └── Task 14 (fill \RESULT{} macros)
                                │
                                ├── Task 17 (Discussion)
                                │     │
                                │     └── Task 18 (Conclusion + statements)
                                │           │
                                │           └── Task 19 (compliance check)
                                │
Tasks 15 (Background) ──────────┘  ← start now, independent
Tasks 16 (GEF framework) ──────────┘  ← start now, independent
```

---

## Quick Reference: Start Points

| You have time for... | Start with |
|----------------------|------------|
| 30 min | Task 1 — fix the LCS labels (one file, clear change) |
| 2 hours | Tasks 1 + 2 + 3 — fix labels, scale N, implement TwoNN |
| Half day | Tasks 1–8 complete (all code, no experiments yet) |
| Writing session | Tasks 15 and 16 — Background and Framework sections, no experiment results needed |
| GPU available | Tasks 12 + 13 after all Phase B/C code is done |
