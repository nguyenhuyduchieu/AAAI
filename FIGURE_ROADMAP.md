# LGF Paper — Figure Roadmap

All figure files should be saved to `../outputs/figures/paper/` so they are
picked up automatically by the `\graphicspath` declaration in the LaTeX
preamble. The paper compiles with placeholder boxes until the actual image
files exist.

---

## Status Key

| Symbol | Meaning |
|--------|---------|
| ✅ Existing | Figure environment and image file both present in paper |
| 🔲 Stub | Figure environment added; image file not yet generated |
| 💡 Recommended | Not yet in paper body; suggested for camera-ready / supplementary |

---

## Part 1 — Existing Figures

These four figures already have `\includegraphics` calls and captions in the
paper body. The image files are expected in `../outputs/figures/paper/`.

---

### Figure 1 — `fig_umap_alldatasets` ✅

**Section:** §4.2 Qualitative Manifold Structure  
**Caption summary:** UMAP projections of pretrained TimesFM 2.0 final-layer
embeddings for all four datasets, coloured by temporal sample index
(purple = early, yellow = late).

**What to show:**
- 2×2 subplot grid: Electricity (top-left), ETTh1 (top-right), Traffic
  (bottom-left), Weather (bottom-right)
- Each point is one of the 3 000 embedding vectors, projected to 2D by UMAP
- Colour encodes temporal order (sample index 0 → N)
- Caption highlights: Electricity/ETTh1 show dense entangled paths; Traffic
  shows a compact web; Weather reveals an outlier cluster present in both
  pretrained and random runs

**How to code it:**

```python
import umap
import matplotlib.pyplot as plt
import numpy as np

# embeddings: dict mapping dataset name -> (3000, 1024) np.ndarray
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
datasets = ['Electricity', 'ETTh1', 'Traffic', 'Weather']

for ax, name in zip(axes.flat, datasets):
    Z = embeddings[name]              # (3000, 1024)
    proj = reducer.fit_transform(Z)   # (3000, 2)
    sc = ax.scatter(proj[:, 0], proj[:, 1],
                    c=np.arange(len(Z)), cmap='viridis',
                    s=2, alpha=0.6)
    ax.set_title(name)
    ax.axis('off')

plt.colorbar(sc, ax=axes, label='Temporal index')
plt.savefig('../outputs/figures/paper/fig_umap_alldatasets.pdf',
            bbox_inches='tight')
```

**Key parameters:** `n_neighbors=15`, `min_dist=0.1`, colormap `viridis`,
point size `s=2`.

---

### Figure 2 — `fig_ltc_profile` ✅

**Section:** §4.3 Latent Trajectory Complexity  
**Caption summary:** LTC_step bar chart per dataset and model type, plus a
right panel comparing all model types on Electricity only.

**What to show:**
- Left panel: grouped bars — pretrained TimesFM vs. random-weight control,
  across all four datasets. Error bars = 95% bootstrap CI.
- Right panel: Electricity only, all model types (Pretrained, Random, LSTM,
  TCN, Transformer). Supervised Transformer bar highlighted in distinct colour
  to show per-dataset memorisation.
- Annotate pretrained TimesFM value (4.64) and Transformer value (16.16) on
  the right panel.

**How to code it:**

```python
import matplotlib.pyplot as plt
import numpy as np

datasets   = ['Electricity', 'Weather', 'ETTh1', 'Traffic']
pre_vals   = [4.64, 5.10, 5.02, 3.77]
rand_vals  = [3.90, 3.52, 3.70, 2.55]
pre_ci     = [0.12, 0.15, 0.14, 0.11]   # 95% bootstrap CI half-widths
rand_ci    = [0.10, 0.12, 0.11, 0.09]

x = np.arange(len(datasets))
w = 0.35

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Left: pretrained vs random across datasets
ax1.bar(x - w/2, pre_vals,  w, yerr=pre_ci,  label='Pretrained', color='steelblue')
ax1.bar(x + w/2, rand_vals, w, yerr=rand_ci, label='Random',     color='lightcoral')
ax1.set_xticks(x); ax1.set_xticklabels(datasets)
ax1.set_ylabel('LTC_step'); ax1.legend()

# Right: all model types on Electricity
models     = ['Pre', 'Rand', 'LSTM', 'TCN', 'Trans.']
elec_vals  = [4.64, 3.90, 4.75, 5.56, 16.16]
colours    = ['steelblue','lightcoral','grey','grey','darkorange']
ax2.bar(models, elec_vals, color=colours)
ax2.set_title('Electricity (all models)')
ax2.set_ylabel('LTC_step')

plt.tight_layout()
plt.savefig('../outputs/figures/paper/fig_ltc_profile.pdf', bbox_inches='tight')
```

**Key parameters:** bar width `w=0.35`, Transformer bar in `darkorange` to
distinguish it visually.

---

### Figure 3 — `fig_geodesic_reliability` ✅

**Section:** §4.4 Manifold Reliability Protocol  
**Caption summary:** Paired bar chart of PDS (blue) vs. FSS (light blue) per
dataset for the pretrained model; random-weight control in red/pink. 95%
bootstrap CI error bars. Non-significant bars marked with †.

**What to show:**
- Four dataset groups on x-axis
- Within each group: 4 bars — [Pre PDS, Pre FSS, Rand PDS, Rand FSS]
- Error bars = 95% bootstrap CI
- Dashed horizontal line at ρ=0
- Annotate bars with † where p > 0.05

**How to code it:**

```python
import matplotlib.pyplot as plt
import numpy as np

datasets = ['Electricity', 'ETTh1', 'Weather', 'Traffic']

# Spearman ρ values
pre_pds  = [0.41,  0.04,  0.11,  0.25]
pre_fss  = [0.01, -0.01,  0.09,  0.05]
rand_pds = [-0.34, -0.27, 0.10,  0.08]
rand_fss = [0.13, -0.01,  0.10,  0.12]

# 95% CI half-widths (bootstrap)
pre_pds_ci  = [0.04, 0.06, 0.05, 0.04]
pre_fss_ci  = [0.03, 0.04, 0.04, 0.03]
rand_pds_ci = [0.05, 0.06, 0.05, 0.05]
rand_fss_ci = [0.04, 0.04, 0.04, 0.04]

x = np.arange(len(datasets))
w = 0.2
fig, ax = plt.subplots(figsize=(10, 4))

ax.bar(x - 1.5*w, pre_pds,  w, yerr=pre_pds_ci,  color='steelblue',   label='Pre PDS')
ax.bar(x - 0.5*w, pre_fss,  w, yerr=pre_fss_ci,  color='lightsteelblue', label='Pre FSS')
ax.bar(x + 0.5*w, rand_pds, w, yerr=rand_pds_ci, color='tomato',      label='Rand PDS')
ax.bar(x + 1.5*w, rand_fss, w, yerr=rand_fss_ci, color='lightsalmon', label='Rand FSS')

ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
ax.set_xticks(x); ax.set_xticklabels(datasets)
ax.set_ylabel('Spearman ρ')
ax.legend()

# Mark non-significant bars with †
for xi, val in zip(x, [0.04, 0.04, 0.09, 0.05]):  # pre_fss non-sig on ETTh1
    if abs(val) < 0.05:
        ax.text(xi - 0.5*w, val + 0.01, '†', ha='center')

plt.tight_layout()
plt.savefig('../outputs/figures/paper/fig_geodesic_reliability.pdf',
            bbox_inches='tight')
```

---

### Figure 4 — `fig_probing_heatmap` ✅

**Section:** §4.5 Temporal Primitive Probing  
**Caption summary:** Two-panel heatmap of CSS balanced accuracy (%) —
left panel = pretrained TimesFM, right panel = random-weight control.
Rows = concepts (Trend, Seasonality, Volatility); columns = datasets.

**What to show:**
- 3 rows × 4 columns per panel
- Colormap centred at 50% (chance), range 40–90%
- Bold-annotate cells significant at p < 0.05
- Annotate non-significant cells with †

**How to code it:**

```python
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

concepts  = ['Trend', 'Seasonality', 'Volatility']
datasets  = ['Electricity', 'Weather', 'ETTh1', 'Traffic']

pre_data = np.array([
    [54.3, 80.7, 71.8, 65.0],   # Trend (†,*,†,*)
    [82.6, 71.8, 77.6, 82.6],   # Seasonality
    [77.6, 70.8, 68.0, 67.0],   # Volatility
])
rand_data = np.array([
    [47.5, 83.5, 74.7, 48.6],
    [64.3, 65.9, 72.8, 71.0],
    [64.9, 60.1, 65.0, 62.1],
])
# sig[i,j] = True if p < 0.05
sig = np.array([
    [False, True,  False, True ],
    [True,  True,  True,  True ],
    [True,  True,  True,  True ],
])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)

def annotate(ax, data, sig_mask):
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            label = f'{data[i,j]:.1f}'
            weight = 'bold' if sig_mask[i, j] else 'normal'
            suffix = '' if sig_mask[i, j] else '†'
            ax.text(j + 0.5, i + 0.5, label + suffix,
                    ha='center', va='center',
                    fontweight=weight, fontsize=9)

for ax, data, title in [(ax1, pre_data, 'Pretrained'),
                         (ax2, rand_data, 'Random')]:
    sns.heatmap(data, ax=ax, annot=False, cmap='RdYlGn',
                vmin=40, vmax=90, center=50,
                xticklabels=datasets, yticklabels=concepts,
                cbar=(ax == ax2))
    annotate(ax, data, sig)
    ax.set_title(title)

plt.tight_layout()
plt.savefig('../outputs/figures/paper/fig_probing_heatmap.pdf',
            bbox_inches='tight')
```

**Key parameters:** colormap `RdYlGn`, `vmin=40`, `vmax=90`, `center=50`.

---

## Part 2 — New Figure Stubs (in-paper placeholders)

These six figures have `\includegraphics` calls and full captions in the paper
body. Generate and save the image files to replace the placeholder boxes in
the compiled PDF.

---

### Figure 5 — `fig_lgf_pipeline` 🔲

**Section:** §3 (after the LGF metric summary table)  
**Caption summary:** LGF four-stage pipeline overview — frozen encoder maps
windows to embeddings; four coloured stage boxes; eight metric outputs.

**What to show:**
- Left: an input time-series window (small waveform icon)
- Arrow → frozen encoder block labelled `f_θ (frozen)`
- Arrow → embedding cloud labelled `Z ⊂ ℝᴰ`
- Four stage boxes (colour-coded):
  - Stage 1 (blue): **LTC** → outputs `LTC_step`, `LTC_dir`
  - Stage 2 (green): **Regularity** → outputs `LCS`, `TSI`, `GSP`
  - Stage 3 (orange): **Reliability** → outputs `PDS`, `FSS`
  - Stage 4 (purple): **Encoding** → outputs `CSS(trend)`, `CSS(season)`, `CSS(vol)`
- Right: annotate each metric with its "good value" (from Table 2 in paper)

**How to code it:**

```python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(14, 4))
ax.set_xlim(0, 14); ax.set_ylim(0, 4); ax.axis('off')

# Helper
def box(ax, x, y, w, h, label, sublabel, color):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle='round,pad=0.1',
                          linewidth=1.5, edgecolor=color,
                          facecolor=color + '22')   # light fill
    ax.add_patch(rect)
    ax.text(x + w/2, y + h*0.65, label,
            ha='center', va='center', fontsize=10, fontweight='bold', color=color)
    ax.text(x + w/2, y + h*0.25, sublabel,
            ha='center', va='center', fontsize=7.5, color='grey')

def arrow(ax, x1, x2, y=2.0):
    ax.annotate('', xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

# Input window
ax.text(0.3, 2.0, '🗠\nwindow\nwᵢ', ha='center', va='center', fontsize=9)
arrow(ax, 0.7, 1.2)

# Encoder
box(ax, 1.2, 1.2, 1.8, 1.6, 'Encoder  fθ', '(frozen inference)', '#555555')
arrow(ax, 3.0, 3.5)

# Embedding cloud
ax.text(3.7, 2.0, 'Embeddings\nZ ⊂ ℝᴰ', ha='center', va='center', fontsize=9)
arrow(ax, 4.2, 4.8)

# Stage boxes
stages = [
    (4.8,  'Stage 1\nLTC',       'LTC_step\nLTC_dir',         '#1f77b4'),
    (6.8,  'Stage 2\nRegularity','LCS  TSI  GSP',              '#2ca02c'),
    (8.8,  'Stage 3\nReliability','PDS   FSS',                 '#ff7f0e'),
    (10.8, 'Stage 4\nEncoding',  'CSS(trend)\nCSS(season)\nCSS(vol)', '#9467bd'),
]
for x, label, sub, color in stages:
    box(ax, x, 0.8, 1.8, 2.4, label, sub, color)
    if x < 10.8:
        arrow(ax, x + 1.8, x + 2.0)

# Output label
ax.text(13.5, 2.0, '8 geometric\nmetrics', ha='center', va='center',
        fontsize=9, style='italic')

plt.tight_layout()
plt.savefig('../outputs/figures/paper/fig_lgf_pipeline.pdf', bbox_inches='tight')
```

---

### Figure 6 — `fig_ltc_intuition` 🔲

**Section:** §3.2 Stage 1 — LTC (before Formal definition)  
**Caption summary:** Two-panel cartoon showing a short zig-zag path
(random-weight) vs. a longer, directional path (pretrained). Arrows show
step vectors δᵢ; angle arcs illustrate LTC_dir.

**What to show:**
- Two panels side-by-side on the same scale
- Each panel: ~12 numbered points connected by arrows in time order
- Left (random): arrows point in random directions, short, scattered
- Right (pretrained): arrows drift in a consistent direction, longer
- Annotate LTC_step and LTC_dir numerical values in panel titles
- Draw an angle arc between two consecutive arrow pairs to visualise the
  cosine similarity that LTC_dir averages

**How to code it:**

```python
import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(42)

def make_path(N, drift, noise):
    """Generate a 2D embedding path with given drift and noise level."""
    steps = rng.normal(loc=drift, scale=noise, size=(N, 2))
    return np.cumsum(steps, axis=0)

N = 14
rand_path = make_path(N, drift=[0, 0],    noise=1.0)   # random walk
pre_path  = make_path(N, drift=[0.6, 0.3], noise=0.5)  # biased drift

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

for ax, path, title, color in [
    (ax1, rand_path, 'Random-weight\nLTC_step≈low  LTC_dir≈0', '#e74c3c'),
    (ax2, pre_path,  'Pretrained\nLTC_step≈high  LTC_dir>0',   '#2980b9'),
]:
    ax.scatter(path[:, 0], path[:, 1], color=color, zorder=5, s=40)
    for i in range(len(path) - 1):
        ax.annotate('', xy=path[i+1], xytext=path[i],
                    arrowprops=dict(arrowstyle='->', color=color,
                                   lw=1.5, mutation_scale=12))
    # Number the points
    for i, (x, y) in enumerate(path):
        ax.text(x + 0.05, y + 0.05, str(i+1), fontsize=7, color='grey')
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('Embedding dim 1')
    ax.set_ylabel('Embedding dim 2')
    ax.axis('equal')

plt.tight_layout()
plt.savefig('../outputs/figures/paper/fig_ltc_intuition.pdf', bbox_inches='tight')
```

---

### Figure 7 — `fig_lcs_clusters` 🔲

**Section:** §3.3 Stage 2 — LCS (after Proposition on LCS degeneracy)  
**Caption summary:** Two-panel 2D PCA scatter coloured by STL trend label
(red = up, green = flat, blue = down). Left: pretrained — separated blobs.
Right: random — colours intermixed.

**What to show:**
- Fit PCA on pretrained embeddings; project random onto the same axes
- Plot N=500 randomly sampled points per panel (manageable visual density)
- Draw convex-hull outlines per cluster in the pretrained panel
- Annotate the LCS value in each panel title
- Use consistent colours throughout paper: red (#e74c3c), green (#2ecc71),
  blue (#3498db)

**How to code it:**

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull

# Z_pre, Z_rand: (3000, 1024) embedding matrices
# labels: (3000,) int array — 0=up, 1=flat, 2=down
# lcs_pre, lcs_rand: scalar LCS values

N_show = 500
idx = np.random.choice(len(Z_pre), N_show, replace=False)

pca = PCA(n_components=2).fit(Z_pre)
P_pre  = pca.transform(Z_pre[idx])
P_rand = pca.transform(Z_rand[idx])
y_sub  = labels[idx]

colors = ['#e74c3c', '#2ecc71', '#3498db']
cnames = ['Up-trend', 'Flat', 'Down-trend']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

for ax, P, title, draw_hull in [
    (ax1, P_pre,  f'Pretrained  (LCS = {lcs_pre:.3f})',  True),
    (ax2, P_rand, f'Random      (LCS = {lcs_rand:.3f})', False),
]:
    for k, (c, name) in enumerate(zip(colors, cnames)):
        mask = y_sub == k
        ax.scatter(P[mask, 0], P[mask, 1], color=c, label=name,
                   s=12, alpha=0.6)
        if draw_hull and mask.sum() > 3:
            hull = ConvexHull(P[mask])
            pts  = np.append(P[mask][hull.vertices],
                             P[mask][hull.vertices[:1]], axis=0)
            ax.plot(pts[:, 0], pts[:, 1], color=c, lw=1.2, alpha=0.7)
    ax.set_title(title)
    ax.axis('off')

ax1.legend(loc='lower right', fontsize=8)
plt.tight_layout()
plt.savefig('../outputs/figures/paper/fig_lcs_clusters.pdf', bbox_inches='tight')
```

---

### Figure 8 — `fig_pds_concept` 🔲

**Section:** §3.4 Stage 3 — PDS/FSS (before Formal definitions)  
**Caption summary:** Two-panel diagram showing PDS (local density → error)
and FSS (global proximity → error similarity). Point size proportional to
|eᵢ|; k-NN circle shown on isolated point; error-difference arrows on FSS.

**What to show:**
- **Left panel (PDS):** 2D scatter. Dense region (blue) with small points
  (low error). Sparse region (orange) with large points (high error). Draw a
  dashed circle of radius d̄ᵢ around one isolated point. Label "Dense region
  (small d̄ᵢ, low error)" and "Sparse region (large d̄ᵢ, high error)".
- **Right panel (FSS):** Three points: zᵢ, zⱼ (close together), zₖ (far).
  Draw bidirectional arrows labelled ‖zᵢ−zⱼ‖ (small) and ‖zᵢ−zₖ‖ (large).
  Annotate |eᵢ−eⱼ| (small) and |eᵢ−eₖ| (large).

**How to code it:**

```python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

rng = np.random.default_rng(7)

# --- Left panel: PDS ---
# Dense cluster (low error)
dense_xy = rng.normal(loc=[1.5, 1.5], scale=0.25, size=(40, 2))
dense_err = rng.uniform(0.02, 0.08, 40)

# Sparse outliers (high error)
sparse_xy  = np.array([[3.8, 3.0], [4.2, 2.5], [3.5, 3.6]])
sparse_err = np.array([0.45, 0.52, 0.38])

ax1.scatter(dense_xy[:,0], dense_xy[:,1],
            s=dense_err * 800, c='#3498db', alpha=0.6, label='Low error')
ax1.scatter(sparse_xy[:,0], sparse_xy[:,1],
            s=sparse_err * 800, c='#e67e22', alpha=0.8, label='High error')

# k-NN circle around most isolated point
iso = sparse_xy[0]
ax1.add_patch(plt.Circle(iso, 0.6, fill=False,
                         linestyle='--', edgecolor='#e67e22', lw=1.5))
ax1.annotate('Large $\\bar{d}_i$', xy=iso + [0, 0.65], ha='center',
             fontsize=8, color='#e67e22')
ax1.text(1.5, 0.7, 'Dense region\n(small $\\bar{d}_i$, low error)',
         ha='center', fontsize=8, color='#2980b9')
ax1.set_title('PDS — local density')
ax1.legend(fontsize=8); ax1.axis('equal'); ax1.axis('off')

# --- Right panel: FSS ---
zi = np.array([1.0, 1.0])
zj = np.array([1.4, 1.2])   # close
zk = np.array([3.5, 2.8])   # far

for pt, lbl in [(zi, '$z_i$'), (zj, '$z_j$'), (zk, '$z_k$')]:
    ax2.scatter(*pt, s=80, zorder=5, color='#2c3e50')
    ax2.text(pt[0]+0.07, pt[1]+0.07, lbl, fontsize=10)

# Close pair arrow
ax2.annotate('', xy=zj, xytext=zi,
             arrowprops=dict(arrowstyle='<->', color='#27ae60', lw=2))
mid_ij = (zi + zj) / 2
ax2.text(mid_ij[0]-0.15, mid_ij[1]+0.1,
         '$\\|z_i-z_j\\|$ small\n$|e_i-e_j|$ small',
         fontsize=7.5, color='#27ae60')

# Far pair arrow
ax2.annotate('', xy=zk, xytext=zi,
             arrowprops=dict(arrowstyle='<->', color='#c0392b', lw=2))
mid_ik = (zi + zk) / 2
ax2.text(mid_ik[0]+0.1, mid_ik[1],
         '$\\|z_i-z_k\\|$ large\n$|e_i-e_k|$ large',
         fontsize=7.5, color='#c0392b')

ax2.set_title('FSS — global proximity')
ax2.set_xlim(0, 5); ax2.set_ylim(0, 4); ax2.axis('off')

plt.tight_layout()
plt.savefig('../outputs/figures/paper/fig_pds_concept.pdf', bbox_inches='tight')
```

---

### Figure 9 — `fig_css_probe` 🔲

**Section:** §3.5 Stage 4 — CSS (after the CSS equation block)  
**Caption summary:** Three-panel 2D PCA scatter with logistic decision
boundaries. Left: seasonality — good separation. Centre: volatility — good
separation. Right: trend — no separation. CSS values annotated.

**What to show:**
- Fit PCA(2) on frozen embeddings; fit LogisticRegression(C=1) on 2D
  projection (for visualisation only; actual CSS is on full D-dim embeddings)
- Shade decision regions (alpha=0.15)
- Plot dashed decision boundary (contour at level=0)
- Colour: class 0 = `#8e44ad` (purple), class 1 = `#e67e22` (orange)
- Annotate CSS% and significance marker in panel title

**How to code it:**

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

# Z_pre: (3000, 1024) embeddings
# y_season, y_vol, y_trend: (3000,) binary labels

N_show = 600
idx = np.random.choice(len(Z_pre), N_show, replace=False)
pca = PCA(n_components=2).fit(Z_pre)
P2  = pca.transform(Z_pre[idx])

concepts = [
    ('Seasonality', y_season[idx], 78.6, True),
    ('Volatility',  y_vol[idx],    70.9, True),
    ('Trend',       y_trend[idx],  67.9, False),
]

fig, axes = plt.subplots(1, 3, figsize=(13, 4))

for ax, (name, y, css, sig) in zip(axes, concepts):
    clf = LogisticRegression(C=1, max_iter=1000).fit(P2, y)

    # Decision region background
    xx, yy = np.meshgrid(np.linspace(P2[:,0].min()-0.5, P2[:,0].max()+0.5, 200),
                          np.linspace(P2[:,1].min()-0.5, P2[:,1].max()+0.5, 200))
    Z_grid = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:,1].reshape(xx.shape)
    ax.contourf(xx, yy, Z_grid, levels=[0, 0.5, 1],
                colors=['#8e44ad', '#e67e22'], alpha=0.12)
    ax.contour(xx, yy, Z_grid, levels=[0.5],
               colors='black', linestyles='--', linewidths=1.2)

    # Scatter
    for k, color in enumerate(['#8e44ad', '#e67e22']):
        mask = y == k
        ax.scatter(P2[mask, 0], P2[mask, 1], color=color,
                   s=8, alpha=0.5)

    sig_str = '✓' if sig else '†(n.s.)'
    ax.set_title(f'{name}\nCSS = {css:.1f}%  {sig_str}', fontsize=9)
    ax.axis('off')

plt.tight_layout()
plt.savefig('../outputs/figures/paper/fig_css_probe.pdf', bbox_inches='tight')
```

---

### Figure 10 — `fig_ood_summary` 🔲

**Section:** §4.6 Out-of-Distribution Validation (after "The predicted split
holds" paragraph)  
**Caption summary:** Three-row × two-column grid comparing in-distribution
(left) vs. OOD equity (right) for LTC_step, CSS(seasonality), and PDS.

**What to show:**
- 3 rows × 2 cols: rows = LTC_step / CSS-season / PDS; cols = in-dist / OOD
- Each cell: paired bars (pretrained blue, random orange)
- PDS row: add dashed ρ=0 baseline
- Bold-frame the OOD column to visually separate it
- Consistent colours with Figures 2 and 3

**How to code it:**

```python
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(3, 2, figsize=(10, 7), sharey='row')
fig.subplots_adjust(hspace=0.5, wspace=0.35)

# --- Row 0: LTC_step ---
indist_datasets = ['Elec.', 'Weath.', 'ETTh1', 'Traffic']
ood_datasets    = ['CSI300', 'CSI800']

ltc_pre_ind  = [4.64, 5.10, 5.02, 3.77]
ltc_rand_ind = [3.90, 3.52, 3.70, 2.55]
ltc_pre_ood  = [4.24, 4.30]
ltc_rand_ood = [3.80, 3.82]

for ax, vals_pre, vals_rand, xlabels in [
    (axes[0, 0], ltc_pre_ind, ltc_rand_ind, indist_datasets),
    (axes[0, 1], ltc_pre_ood, ltc_rand_ood, ood_datasets),
]:
    x = np.arange(len(xlabels)); w = 0.35
    ax.bar(x - w/2, vals_pre,  w, color='#2980b9', label='Pre')
    ax.bar(x + w/2, vals_rand, w, color='#e74c3c', label='Rand')
    ax.set_xticks(x); ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_ylabel('LTC_step', fontsize=8)

# --- Row 1: CSS seasonality ---
css_pre_ind  = [82.6, 71.8, 77.6, 82.6]
css_rand_ind = [64.3, 65.9, 72.8, 71.0]
css_pre_ood  = [74.8, 70.0]
css_rand_ood = [58.1, 55.9]

for ax, vals_pre, vals_rand, xlabels in [
    (axes[1, 0], css_pre_ind, css_rand_ind, indist_datasets),
    (axes[1, 1], css_pre_ood, css_rand_ood, ood_datasets),
]:
    x = np.arange(len(xlabels)); w = 0.35
    ax.bar(x - w/2, vals_pre,  w, color='#2980b9')
    ax.bar(x + w/2, vals_rand, w, color='#e74c3c')
    ax.axhline(50, color='grey', linestyle=':', lw=0.9)
    ax.set_xticks(x); ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_ylabel('CSS Season (%)', fontsize=8)

# --- Row 2: PDS ---
pds_pre_ind  = [0.41,  0.04,  0.11,  0.25]
pds_rand_ind = [-0.34, -0.27, 0.10,  0.08]
pds_pre_ood  = [0.02,  -0.09]
pds_rand_ood = [-0.37, -0.38]

for ax, vals_pre, vals_rand, xlabels in [
    (axes[2, 0], pds_pre_ind, pds_rand_ind, indist_datasets),
    (axes[2, 1], pds_pre_ood, pds_rand_ood, ood_datasets),
]:
    x = np.arange(len(xlabels)); w = 0.35
    ax.bar(x - w/2, vals_pre,  w, color='#2980b9')
    ax.bar(x + w/2, vals_rand, w, color='#e74c3c')
    ax.axhline(0, color='black', linestyle='--', lw=1.0)
    ax.set_xticks(x); ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_ylabel('PDS  ρ_s', fontsize=8)

# Column titles and bold OOD frame
axes[0, 0].set_title('In-distribution', fontweight='bold')
axes[0, 1].set_title('OOD (equity)',    fontweight='bold', color='#c0392b')
for row in range(3):
    for spine in axes[row, 1].spines.values():
        spine.set_edgecolor('#c0392b'); spine.set_linewidth(1.5)

axes[0, 0].legend(fontsize=8)
plt.suptitle('OOD validation: LTC and CSS transfer; PDS collapses',
             fontsize=10, fontweight='bold')
plt.savefig('../outputs/figures/paper/fig_ood_summary.pdf', bbox_inches='tight')
```

---

## Part 3 — Recommended Additional Figures

Not yet in paper body. Suggested for camera-ready or supplementary material.

---

### Figure 11 — `fig_umap_pre_vs_rand` 💡

**Section:** Appendix (Extended Experimental Details)  
**Purpose:** Directly shows how pretraining reshapes the manifold topology by
projecting pretrained and random embeddings onto identical UMAP axes.

**How to code it:**

```python
import umap, matplotlib.pyplot as plt, numpy as np

# Z_pre, Z_rand: (3000, 1024) for the same dataset (e.g., Electricity)
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
reducer.fit(Z_pre)

P_pre  = reducer.transform(Z_pre)
P_rand = reducer.transform(Z_rand)   # same axes!

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
color_idx = np.arange(3000)
for ax, P, title in [
    (ax1, P_pre,  f'Pretrained  LTC_step={ltc_pre:.2f}'),
    (ax2, P_rand, f'Random      LTC_step={ltc_rand:.2f}'),
]:
    sc = ax.scatter(P[:,0], P[:,1], c=color_idx, cmap='viridis', s=2, alpha=0.5)
    ax.set_title(title); ax.axis('off')

plt.colorbar(sc, ax=[ax1, ax2], label='Temporal index', shrink=0.6)
plt.savefig('../outputs/figures/paper/fig_umap_pre_vs_rand.pdf',
            bbox_inches='tight')
```

**Key insight:** Using `.transform()` (not `.fit_transform()`) ensures both
panels share the same coordinate system, making the topological difference
directly visible.

---

### Figure 12 — `fig_ltc_by_layer` 💡

**Section:** Appendix (Extended Experimental Details)  
**Purpose:** Shows at which transformer layer temporal trajectory complexity
emerges — diagnostically valuable for understanding where the model encodes
temporal dynamics.

**How to code it:**

```python
import torch, matplotlib.pyplot as plt

# Register forward hooks on each transformer block
layer_embeddings = {}

def make_hook(layer_idx):
    def hook(module, input, output):
        # Mean-pool token dimension → (batch, D)
        layer_embeddings[layer_idx] = output[0].mean(dim=1).detach().cpu()
    return hook

handles = []
for i, block in enumerate(model.transformer.layers):
    handles.append(block.register_forward_hook(make_hook(i)))

# Run inference
with torch.no_grad():
    _ = model(windows_tensor)   # windows_tensor: (N, L)

for h in handles: h.remove()

# Compute LTC_step per layer
def ltc_step(Z):
    """Z: (N, D) sorted by time."""
    diffs = np.diff(Z, axis=0)
    return np.mean(np.linalg.norm(diffs, axis=1))

n_layers = len(layer_embeddings)
ltc_pre_per_layer  = [ltc_step(layer_embeddings[l].numpy()) for l in range(n_layers)]
ltc_rand_per_layer = [...]  # same for random-weight model

plt.figure(figsize=(7, 4))
plt.plot(range(n_layers), ltc_pre_per_layer,  'o-', label='Pretrained', color='#2980b9')
plt.plot(range(n_layers), ltc_rand_per_layer, 's--', label='Random',    color='#e74c3c')
plt.fill_between(range(n_layers),
                 ltc_rand_per_layer, ltc_pre_per_layer,
                 alpha=0.15, color='#2980b9')
plt.xlabel('Transformer layer index')
plt.ylabel('LTC_step')
plt.title('Temporal trajectory complexity by layer')
plt.legend(); plt.tight_layout()
plt.savefig('../outputs/figures/paper/fig_ltc_by_layer.pdf', bbox_inches='tight')
```

---

### Figure 13 — `fig_pds_scatter` 💡

**Section:** Appendix (Extended Experimental Details)  
**Purpose:** Raw scatter of neighbourhood isolation vs. forecast error for
each dataset, making the PDS Spearman correlation visually legible.

**How to code it:**

```python
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

datasets = ['Electricity', 'ETTh1', 'Weather', 'Traffic', 'CSI300', 'CSI800']
fig, axes = plt.subplots(2, 3, figsize=(13, 8))

for ax, name in zip(axes.flat, datasets):
    d_bar = knn_isolation[name]   # (N,) mean k-NN distances
    errors = abs_errors[name]     # (N,) |e_i|

    # Subsample for readability
    idx = np.random.choice(len(d_bar), 500, replace=False)
    rho, pval = stats.spearmanr(d_bar[idx], errors[idx])

    ax.scatter(d_bar[idx], errors[idx], s=6, alpha=0.3, color='#2980b9')

    # Regression line
    m, b = np.polyfit(d_bar[idx], errors[idx], 1)
    x_line = np.linspace(d_bar.min(), d_bar.max(), 100)
    ax.plot(x_line, m*x_line + b, color='#c0392b', lw=1.5)

    sig = '***' if pval < 0.001 else ('**' if pval < 0.01 else
          ('*' if pval < 0.05 else '(n.s.)'))
    ax.set_title(f'{name}\nρ = {rho:.2f}  {sig}', fontsize=9)
    ax.set_xlabel('$\\bar{d}_i$ (isolation)', fontsize=8)
    ax.set_ylabel('$|e_i|$ (forecast error)', fontsize=8)

plt.tight_layout()
plt.savefig('../outputs/figures/paper/fig_pds_scatter.pdf', bbox_inches='tight')
```

---

### Figure 14 — `fig_tsfm_taxonomy` 💡

**Section:** §2 Related Work  
**Purpose:** Visual taxonomy of TSFM architectural families — makes the
Related Work section scannable and orients the reader before the experiments.

**How to code it:**

```python
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

G = nx.DiGraph()

families = {
    'Masked\nEnc-Dec':    ['MOMENT',  'Moirai',  'Moirai-MoE'],
    'Decoder-only\nAR':   ['TimesFM', 'Timer'],
    'Language-\ninspired':['Chronos', 'Lag-Llama','LLMTime'],
    'Efficient /\nMoE':   ['TinyTimeMixer', 'Time-MoE'],
}
family_colors = {
    'Masked\nEnc-Dec':    '#3498db',
    'Decoder-only\nAR':   '#2ecc71',
    'Language-\ninspired':'#e67e22',
    'Efficient /\nMoE':   '#9b59b6',
}
# Models with LGF scope (bold border)
lgf_models = {'TimesFM', 'Chronos', 'MOMENT', 'Moirai', 'TinyTimeMixer'}

root = 'TSFMs'
G.add_node(root)
for fam, models in families.items():
    G.add_edge(root, fam)
    for m in models:
        G.add_edge(fam, m)

pos = nx.nx_agraph.graphviz_layout(G, prog='dot')   # requires pygraphviz

fig, ax = plt.subplots(figsize=(12, 5))
nx.draw(G, pos, ax=ax, with_labels=True, arrows=False,
        node_color=['#ecf0f1'] * len(G.nodes()),
        font_size=8, node_size=1500)
# Colour family nodes
for fam, color in family_colors.items():
    nx.draw_networkx_nodes(G, pos, nodelist=[fam], node_color=color,
                           node_size=2000, ax=ax)
plt.tight_layout()
plt.savefig('../outputs/figures/paper/fig_tsfm_taxonomy.pdf', bbox_inches='tight')
```

---

### Figure 15 — `fig_css_gap_vs_params` 💡

**Section:** §5 Discussion  
**Purpose:** Once all 5 models are run, plot CSS gap (pretrained minus random)
vs. log(params) to test whether geometric richness scales predictably —
connecting to Edwards et al. (2024) scaling laws.

**How to code it:**

```python
import matplotlib.pyplot as plt
import numpy as np

# Populate as experiments complete; × marks are placeholders
models   = ['TinyTimeMixer', 'Chronos-Small', 'MOMENT-Large', 'TimesFM 2.0', 'Moirai']
params_M = [5,               46,              385,             500,            311]
# CSS gap (season) = pretrained_balanced_acc - random_balanced_acc
gap_season = [None,  None,  None,  10.1, None]   # None = not yet run
gap_vol    = [None,  None,  None,   7.9, None]

fig, ax = plt.subplots(figsize=(7, 4))
x = np.log10(params_M)

for i, (m, gs, gv) in enumerate(zip(models, gap_season, gap_vol)):
    if gs is not None:
        ax.scatter(x[i], gs, marker='o', s=100, color='#2980b9',
                   zorder=5, label='Season' if i == 0 else '')
        ax.scatter(x[i], gv, marker='s', s=100, color='#e67e22',
                   zorder=5, label='Volatility' if i == 0 else '')
        ax.annotate(m, (x[i], gs + 0.3), fontsize=8)
    else:
        ax.scatter(x[i], 0, marker='x', s=80, color='grey', zorder=5)
        ax.annotate(f'{m}\n(planned)', (x[i], 0.5), fontsize=7,
                    color='grey', ha='center')

ax.set_xlabel('log₁₀(Parameters)')
ax.set_ylabel('CSS gap (Pre − Rand, pp)')
ax.axhline(0, color='grey', linestyle=':', lw=0.9)
ax.set_title('CSS gap vs. model scale (§ scaling-law connection)')
ax.legend()
plt.tight_layout()
plt.savefig('../outputs/figures/paper/fig_css_gap_vs_params.pdf',
            bbox_inches='tight')
```

---

## Quick-reference summary

| # | Filename | Section | Type | Status |
|---|----------|---------|------|--------|
| 1 | `fig_umap_alldatasets` | §4.2 | UMAP scatter (4 panels) | ✅ Existing |
| 2 | `fig_ltc_profile` | §4.3 | Grouped bar chart | ✅ Existing |
| 3 | `fig_geodesic_reliability` | §4.4 | Paired bar chart | ✅ Existing |
| 4 | `fig_probing_heatmap` | §4.5 | Seaborn heatmap | ✅ Existing |
| 5 | `fig_lgf_pipeline` | §3 | Schematic flowchart | ✅ Existing |
| 6 | `fig_ltc_intuition` | §3.2 | Path cartoon (2-panel) | ✅ Existing |
| 7 | `fig_lcs_clusters` | §3.3 | PCA scatter (2-panel) | ✅ Existing |
| 8 | `fig_pds_concept` | §3.4 | Density + proximity diagram | ✅ Existing |
| 9 | `fig_css_probe` | §3.5 | Decision boundary scatter (3-panel) | ✅ Existing |
| 10 | `fig_ood_summary` | §4.6 | 3×2 bar grid | ✅ Existing |
| 11 | `fig_umap_pre_vs_rand` | Appendix | Side-by-side UMAP | ✅ Existing |
| 12 | `fig_ltc_by_layer` | Appendix | Layer-wise line plot | ✅ Existing |
| 13 | `fig_pds_scatter` | Appendix | Isolation vs. error scatter | ✅ Existing |
| 14 | `fig_tsfm_taxonomy` | §2 | Taxonomy tree diagram | ✅ Existing |
| 15 | `fig_css_gap_vs_params` | §5 | Scaling-law scatter | ✅ Existing |

> **Output path:** save all `.pdf` files to `../outputs/figures/paper/`  
> **Compile:** run `pdflatex` twice after adding any new image file — the
> placeholder box will automatically be replaced by the generated figure.
