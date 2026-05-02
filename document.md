# Step-by-Step Guide: Running a Random-Weight Control Experiment for TimesFM 2.0

This guide reproduces the core experiment behind the paper’s Figure 11:

> comparing pretrained TimesFM embeddings vs random-weight embeddings
> using the SAME architecture, SAME dataset, and SAME UMAP projection.

The goal is to isolate the effect of pretraining on latent geometry.

---

# 1. Install Dependencies

```bash
pip install torch transformers datasets umap-learn matplotlib scikit-learn numpy pandas
```

Optional:

```bash
pip install seaborn
```

---

# 2. Load Dataset

Example: Electricity dataset.

You can use:
- Monash forecasting repository
- HuggingFace datasets
- custom CSV

Example using CSV:

```python
import pandas as pd

df = pd.read_csv("electricity.csv")
series = df["value"].values
```

---

# 3. Create Sliding Windows

TimesFM consumes fixed-length windows.

Example:
- context length = 512
- stride = 1

```python
import numpy as np

WINDOW = 512

windows = []

for i in range(len(series) - WINDOW):
    w = series[i:i+WINDOW]
    windows.append(w)

windows = np.array(windows)
```

Shape:

```python
(num_windows, 512)
```

---

# 4. Load Pretrained TimesFM

Example pseudocode:

```python
from transformers import AutoModel

pretrained_model = AutoModel.from_pretrained(
    "google/timesfm-2.0"
)

pretrained_model.eval()
```

---

# 5. Create Random-Weight Control

CRITICAL:
- same architecture
- same config
- random initialization only

DO NOT load pretrained weights.

Example:

```python
from transformers import AutoConfig, AutoModel

config = AutoConfig.from_pretrained(
    "google/timesfm-2.0"
)

random_model = AutoModel.from_config(config)

random_model.eval()
```

At this point:
- architecture identical
- weights random

---

# 6. Extract Final-Layer Embeddings

Usually:
- mean pooled hidden states
or
- CLS token
or
- final timestep embedding

Example:

```python
import torch

def extract_embeddings(model, windows):
    
    all_embeddings = []

    with torch.no_grad():

        for w in windows:

            x = torch.tensor(w).float().unsqueeze(0)

            outputs = model(x)

            hidden = outputs.last_hidden_state

            embedding = hidden.mean(dim=1)

            all_embeddings.append(
                embedding.squeeze(0).cpu().numpy()
            )

    return np.array(all_embeddings)
```

---

# 7. Generate Embeddings

## Pretrained

```python
Z_pretrained = extract_embeddings(
    pretrained_model,
    windows
)
```

## Random

```python
Z_random = extract_embeddings(
    random_model,
    windows
)
```

Typical shape:

```python
(num_windows, hidden_dim)
```

Example:

```python
(10000, 1024)
```

---

# 8. Fit UMAP ONLY on Pretrained Embeddings

IMPORTANT:
DO NOT use fit_transform separately.

Correct:

```python
import umap

reducer = umap.UMAP(
    n_neighbors=30,
    min_dist=0.1,
    metric="cosine",
    random_state=42
)

Z_pretrained_2d = reducer.fit_transform(
    Z_pretrained
)
```

This creates the shared coordinate system.

---

# 9. Transform Random Embeddings into SAME Space

IMPORTANT:
use `.transform()`

NOT `.fit_transform()`

```python
Z_random_2d = reducer.transform(
    Z_random
)
```

Now both embeddings share:
- same axes
- same manifold basis
- same projection geometry

This is essential for scientific comparison.

---

# 10. Create Temporal Colors

Color points by temporal index.

Early:
- purple

Late:
- yellow

```python
colors = np.linspace(
    0,
    1,
    len(Z_pretrained_2d)
)
```

---

# 11. Plot Results

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Pretrained
sc1 = axes[0].scatter(
    Z_pretrained_2d[:,0],
    Z_pretrained_2d[:,1],
    c=colors,
    cmap="viridis",
    s=5
)

axes[0].set_title("Pretrained TimesFM")

# Random
sc2 = axes[1].scatter(
    Z_random_2d[:,0],
    Z_random_2d[:,1],
    c=colors,
    cmap="viridis",
    s=5
)

axes[1].set_title("Random-Weight Control")

plt.colorbar(sc1, ax=axes)
plt.show()
```

---

# 12. Expected Result

## Pretrained

You should observe:
- smooth trajectories
- coherent flows
- gradual color transitions
- clustered manifold structure

Meaning:
- nearby times map to nearby latent states

---

## Random Control

You should observe:
- scattered points
- mixed colors
- weak trajectory structure
- more isotropic distribution

Meaning:
- no learned temporal organization

---

# 13. Compute LTC (Latent Temporal Coherence)

Basic intuition:

Measure whether neighboring timesteps stay close in latent space.

Simple version:

```python
from sklearn.metrics.pairwise import cosine_similarity

def compute_ltc(Z):

    sims = []

    for i in range(len(Z)-1):

        sim = cosine_similarity(
            Z[i:i+1],
            Z[i+1:i+2]
        )[0][0]

        sims.append(sim)

    return np.mean(sims)
```

---

# 14. Compare LTC

```python
ltc_pretrained = compute_ltc(Z_pretrained)

ltc_random = compute_ltc(Z_random)

print("Pretrained LTC:", ltc_pretrained)
print("Random LTC:", ltc_random)
```

Expected:

```python
Pretrained LTC > Random LTC
```

---

# 15. Why This Experiment Works

This experiment isolates:

| Variable | Same? |
|---|---|
| Architecture | ✅ |
| Dataset | ✅ |
| Input windows | ✅ |
| UMAP projection | ✅ |
| Hyperparameters | ✅ |
| Weights | ❌ |

Therefore:

Any geometry difference comes from:
- learned temporal representations
- not architecture

---

# 16. Common Mistakes

## ❌ Wrong: Separate UMAP fit

```python
umap.fit_transform(Z_pretrained)
umap.fit_transform(Z_random)
```

This destroys comparability.

---

## ❌ Wrong: Different random seeds

Must keep:
- projection seed fixed
- data ordering fixed

---

## ❌ Wrong: Different preprocessing

Normalization must match exactly.

---

## ❌ Wrong: Comparing different architectures

The control only works if:
- architecture identical
- only weights differ

---

# 17. Interpretation

If pretrained embeddings show:
- smoother trajectories
- higher LTC
- denser manifolds

then:

pretraining has learned:
- temporal continuity
- latent state organization
- forecasting-relevant geometry

This is the paper’s core claim.

---

# 18. Deeper Interpretation

The experiment suggests pretrained TSFMs learn:

- temporal manifolds
- latent dynamical systems
- semantically organized state spaces

similar to:
- LLM semantic embeddings
- vision latent manifolds
- self-supervised representation learning