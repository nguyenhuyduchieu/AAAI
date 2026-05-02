# Figure Production TODO (Technical Checklist)

Mục tiêu: hoàn thiện toàn bộ hình theo `FIGURE_ROADMAP.md` với pipeline rõ ràng:
- mỗi task có **script cụ thể**
- nêu rõ **input CSV / input figure**
- nêu rõ **output PDF**
- có **lệnh chạy** tái lập

> Quy ước output: tất cả file PDF lưu tại `outputs/figures/paper/`  
> Quy ước script đề xuất: đặt trong `scripts/figures/`

---

## 0) One-time setup

- [ ] **Tạo thư mục script và output**
  - Input: None
  - Output:
    - `scripts/figures/`
    - `outputs/figures/paper/`
  - Command:
    - `mkdir -p scripts/figures outputs/figures/paper`

- [ ] **Chuẩn hóa môi trường**
  - Input: `requirements.txt`
  - Output: môi trường đủ `matplotlib`, `seaborn`, `pandas`, `numpy`, `scikit-learn`
  - Command:
    - `pip install -r requirements.txt`

---

## 1) Figures đang được include trong `paper_overleaf/main.tex` (ưu tiên cao nhất)

## Task F1 — `fig_umap_alldatasets.pdf`
- Script: `scripts/figures/make_fig_umap_alldatasets.py`
- Input:
  - `outputs/figures/figure_timesfm_umap_electricity_pretrained.png`
  - `outputs/figures/figure_timesfm_umap_ett_pretrained.png`
  - `outputs/figures/figure_timesfm_umap_traffic_pretrained.png`
  - `outputs/figures/figure_timesfm_umap_weather_pretrained.png`
- Output:
  - `outputs/figures/paper/fig_umap_alldatasets.pdf`
- Command:
  - `PYTHONPATH=. python scripts/figures/make_fig_umap_alldatasets.py`

## Task F2 — `fig_ltc_profile.pdf` (strict-consistent)
- Script: `scripts/figures/make_fig_ltc_profile.py`
- Input CSV:
  - `outputs/metrics/timesfm_geometry_missing_validations_electricity.csv`
  - `outputs/metrics/timesfm_geometry_missing_validations_weather.csv`
  - `outputs/metrics/timesfm_geometry_missing_validations_ett.csv`
  - `outputs/metrics/timesfm_geometry_missing_validations_traffic.csv`
- Metric dùng:
  - `LTC_step_baseline_cmp` cho các model (`tsfm_pretrained`, `lstm`, `tcn`, `transformer`)
- Output:
  - `outputs/figures/paper/fig_ltc_profile.pdf`
- Command:
  - `PYTHONPATH=. python scripts/figures/make_fig_ltc_profile.py --source geometry_extensions`

## Task F3 — `fig_geodesic_reliability.pdf`
- Script: `scripts/figures/make_fig_geodesic_reliability.py`
- Input CSV:
  - `outputs/metrics/timesfm_control_metrics_electricity.csv`
  - `outputs/metrics/timesfm_control_metrics_ett.csv`
  - `outputs/metrics/timesfm_control_metrics_weather.csv`
  - `outputs/metrics/timesfm_control_metrics_traffic.csv`
- Metric dùng:
  - `PDS_spearman_mae_havg_raw_corr`
  - `FSS_spearman_havg_raw_corr`
  - CI/p-value từ các cột tương ứng (`*_ci_low`, `*_ci_high`, `*_pvalue`)
  - lọc `label_mode='na'`, `perturb_setting='clean'` cho FSS; `clean_k20` cho PDS
- Output:
  - `outputs/figures/paper/fig_geodesic_reliability.pdf`
- Command:
  - `PYTHONPATH=. python scripts/figures/make_fig_geodesic_reliability.py --k 20`

## Task F4 — `fig_probing_heatmap.pdf`
- Script: `scripts/figures/make_fig_probing_heatmap.py`
- Input CSV:
  - `outputs/metrics/timesfm_control_metrics_electricity.csv`
  - `outputs/metrics/timesfm_control_metrics_ett.csv`
  - `outputs/metrics/timesfm_control_metrics_weather.csv`
  - `outputs/metrics/timesfm_control_metrics_traffic.csv`
- Metric dùng:
  - `CSS_trend_bal_acc`, `CSS_seasonality_bal_acc`, `CSS_volatility_bal_acc`
  - p-value tương ứng để gắn ký hiệu significance
- Output:
  - `outputs/figures/paper/fig_probing_heatmap.pdf`
- Command:
  - `PYTHONPATH=. python scripts/figures/make_fig_probing_heatmap.py`

## Task F5 — OOD figure đang include trong paper (`fig_stock_ood.pdf`)
- Script: `scripts/figures/make_fig_stock_ood.py`
- Input:
  - `outputs/figures/figure_timesfm_umap_csi300_pretrained.png`
  - `outputs/figures/figure_timesfm_umap_csi300_random-reset.png`
  - `outputs/figures/figure_pds_error_umap_csi300.png`
- Output:
  - `outputs/figures/paper/fig_stock_ood.pdf`
- Command:
  - `PYTHONPATH=. python scripts/figures/make_fig_stock_ood.py`

---

## 2) Stub figures trong roadmap (đưa từ placeholder -> có file thật)

## Task S6 — `fig_lgf_pipeline.pdf`
- Script: `scripts/figures/make_fig_lgf_pipeline.py`
- Input CSV (để annotate số tốt/xấu):  
  - `outputs/metrics/timesfm_geometry_missing_validations_cross_dataset.csv`
  - `outputs/metrics/timesfm_control_summary_electricity.csv` (hoặc summary đại diện)
- Output:
  - `outputs/figures/paper/fig_lgf_pipeline.pdf`
- Command:
  - `PYTHONPATH=. python scripts/figures/make_fig_lgf_pipeline.py`

## Task S7 — `fig_ltc_intuition.pdf`
- Script: `scripts/figures/make_fig_ltc_intuition.py`
- Input: không bắt buộc (cartoon/schematic), optional lấy số đại diện từ:
  - `outputs/metrics/timesfm_control_metrics_{electricity,weather,ett,traffic}.csv`
- Output:
  - `outputs/figures/paper/fig_ltc_intuition.pdf`
- Command:
  - `PYTHONPATH=. python scripts/figures/make_fig_ltc_intuition.py`

## Task S8 — `fig_lcs_clusters.pdf`
- Script: `scripts/figures/make_fig_lcs_clusters.py`
- Input CSV:
  - `outputs/metrics/timesfm_control_metrics_weather.csv` (hoặc dataset đại diện)
- Input figure/data bổ sung:
  - UMAP/PCA source từ embeddings (nếu chưa lưu embeddings, dùng surrogate từ UMAP PNG hiện có)
- Output:
  - `outputs/figures/paper/fig_lcs_clusters.pdf`
- Command:
  - `PYTHONPATH=. python scripts/figures/make_fig_lcs_clusters.py --dataset weather`

## Task S9 — `fig_pds_concept.pdf`
- Script: `scripts/figures/make_fig_pds_concept.py`
- Input: schematic + số annotate từ:
  - `outputs/metrics/timesfm_control_metrics_electricity.csv`
  - `outputs/metrics/timesfm_control_metrics_traffic.csv`
- Output:
  - `outputs/figures/paper/fig_pds_concept.pdf`
- Command:
  - `PYTHONPATH=. python scripts/figures/make_fig_pds_concept.py`

## Task S10 — `fig_css_probe.pdf`
- Script: `scripts/figures/make_fig_css_probe.py`
- Input CSV:
  - `outputs/metrics/timesfm_control_metrics_electricity.csv`
  - `outputs/metrics/timesfm_control_metrics_ett.csv`
  - `outputs/metrics/timesfm_control_metrics_weather.csv`
  - `outputs/metrics/timesfm_control_metrics_traffic.csv`
- Output:
  - `outputs/figures/paper/fig_css_probe.pdf`
- Command:
  - `PYTHONPATH=. python scripts/figures/make_fig_css_probe.py`

## Task S11 — `fig_ood_summary.pdf` (nếu quyết định dùng thay `fig_stock_ood`)
- Script: `scripts/figures/make_fig_ood_summary.py`
- Input CSV:
  - `outputs/metrics/timesfm_geometry_missing_validations_rebuttal.csv` (ưu tiên)
  - hoặc `outputs/metrics/timesfm_geometry_missing_validations_cross_dataset.csv`
  - `outputs/metrics/timesfm_control_metrics_csi300.csv`
  - `outputs/metrics/timesfm_control_metrics_csi800.csv`
- Output:
  - `outputs/figures/paper/fig_ood_summary.pdf`
- Command:
  - `PYTHONPATH=. python scripts/figures/make_fig_ood_summary.py`

---

## 3) Recommended figures (supplement / camera-ready mở rộng)

## Task R12 — `fig_umap_pre_vs_rand.pdf`
- Script: `scripts/figures/make_fig_umap_pre_vs_rand.py`
- Input:
  - `outputs/figures/figure_timesfm_umap_{dataset}_pretrained.png`
  - `outputs/figures/figure_timesfm_umap_{dataset}_random-reset.png`
- Output:
  - `outputs/figures/paper/fig_umap_pre_vs_rand.pdf`
- Command:
  - `PYTHONPATH=. python scripts/figures/make_fig_umap_pre_vs_rand.py`

## Task R13 — `fig_ltc_by_layer.pdf`
- Script: `scripts/figures/make_fig_ltc_by_layer.py`
- Input CSV:
  - `outputs/metrics/timesfm_geometry_missing_validations_cross_dataset.csv`
  - lọc `CSS_layer*` (nếu cần layer-wise proxy) hoặc bổ sung output layer-wise LTC riêng
- Output:
  - `outputs/figures/paper/fig_ltc_by_layer.pdf`
- Command:
  - `PYTHONPATH=. python scripts/figures/make_fig_ltc_by_layer.py`

## Task R14 — `fig_pds_scatter.pdf`
- Script: `scripts/figures/make_fig_pds_scatter.py`
- Input:
  - `outputs/figures/figure_pds_density_umap_*.png`
  - `outputs/figures/figure_pds_error_umap_*.png`
  - và/hoặc CSV raw từ control metrics để overlay regression summary
- Output:
  - `outputs/figures/paper/fig_pds_scatter.pdf`
- Command:
  - `PYTHONPATH=. python scripts/figures/make_fig_pds_scatter.py`

## Task R15 — `fig_tsfm_taxonomy.pdf`
- Script: `scripts/figures/make_fig_tsfm_taxonomy.py`
- Input: schematic (không phụ thuộc CSV)
- Output:
  - `outputs/figures/paper/fig_tsfm_taxonomy.pdf`
- Command:
  - `PYTHONPATH=. python scripts/figures/make_fig_tsfm_taxonomy.py`

## Task R16 — `fig_css_gap_vs_params.pdf`
- Script: `scripts/figures/make_fig_css_gap_vs_params.py`
- Input CSV:
  - `outputs/metrics/timesfm_control_metrics_*.csv` (TimesFM)
  - thêm CSV từ model khác khi có (Chronos/MOMENT/Moirai/TinyTimeMixer)
- Output:
  - `outputs/figures/paper/fig_css_gap_vs_params.pdf`
- Command:
  - `PYTHONPATH=. python scripts/figures/make_fig_css_gap_vs_params.py`

---

## 4) Validation + paper compile checklist

- [ ] Kiểm tra đủ file PDF đang được `\includegraphics` gọi trong `paper_overleaf/main.tex`
  - Command:
    - `python - <<'PY'\nfrom pathlib import Path\nreq=['fig_umap_alldatasets','fig_ltc_profile','fig_geodesic_reliability','fig_probing_heatmap','fig_stock_ood']\nbase=Path('outputs/figures/paper')\nfor r in req:\n    p=base/f'{r}.pdf'\n    print(r, 'OK' if p.exists() else 'MISSING')\nPY`

- [ ] Build PDF paper (2 lần)
  - Command:
    - `cd paper_overleaf && pdflatex main.tex && pdflatex main.tex`

- [ ] Kiểm tra caption/text không còn số cũ sau rerun strict-consistent
  - Command:
    - `rg "16\\.16|3\\.5\\\\times|4\\.64|3\\.90|5\\.56|4\\.75" paper_overleaf/main.tex`

---

## 5) Delivery order đề xuất (thực thi nhanh)

1. F1 -> F5 (để paper hết placeholder ngay).  
2. S11 hoặc giữ `fig_stock_ood` (chốt 1 chuẩn OOD).  
3. S6 -> S10 (nâng đủ roadmap).  
4. R12 -> R16 (supplement/camera-ready mở rộng).

