<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a>
    <img src="https://github.com/gerryc-0/Why-so-Time-Serious/blob/main/weasel_icon.png" 
         alt="Weasel v2.0 Icon" width="400">
  </a>
  <p align="center">
    <br />
  </p>
</p>

## Revisiting WEASEL 2.0: Reproduction, Sensitivity, and an Adaptive Ensemble-Size Rule

This paper was accepted to, and will appear in the proceedings of the 11th Workshop on Advanced Analytics and Learning on Temporal Data (AALTD 2026) part of ECML PKDD 2026, held in Naples, Italy.

### ArXiv paper: https://arxiv.org/abs/2608.18021

## Citing

If you use this algorithm or publication, please cite:

```bibtex
@misc{higgins2026revisitingweasel20reproduction,
      title={Revisiting WEASEL 2.0: Reproduction, Sensitivity, and an Adaptive Ensemble-Size Rule}, 
      author={Cian Higgins and Gerard Carrigan and Pinar Sungu Isiacik and Georgiana Ifrim},
      year={2026},
      eprint={2608.18021},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2608.18021}, 
}
```

This is the citation of the **original WEASEL 2.0** paper:
```bibtex
@article{schaefer2023weasel,
  title={WEASEL 2.0: a random dilated dictionary transform for fast, accurate and memory constrained time series classification},
  author={Sch{\"a}fer, Patrick and Leser, Ulf},
  journal={Machine Learning},
  volume={112},
  number={12},
  pages={4763--4788},
  year={2023},
  publisher={Springer}
}
```

## Contact
Cian Higgins - cian.higgins1@ucdconnect.ie

Gerard Carrigan - gerard.carrigan1@ucdconnect.ie

## Tasks

1. **Reproduce WEASEL 2.0** on 114 UCR datasets and compare test accuracy, runtime, and feature counts against the published results.
2. **Reproduce seven competitor classifiers** (MultiRocket, MiniRocket, Rocket, R-DST, Hydra, WEASEL 1.0, cBOSS) under identical conditions.
3. **Statistical analysis**: Wilcoxon signed-rank tests, critical difference diagrams, pairwise scatter plots, and a multi-comparison matrix.
4. **Investigate the ensemble-sizing heuristic**: evaluate fixed, halved, and adaptive ensemble variants to assess sensitivity and potential improvements.

---

## Built With

* Python 3.12
* [aeon](https://github.com/aeon-toolkit/aeon): UCR dataset loading and competitor classifiers
* [weasel](https://github.com/patrickzib/dictionary): WEASEL 1.0 and WEASEL 2.0 implementations
* scikit-learn 1.6.1
* NumPy / Pandas / Matplotlib / Seaborn
* SciPy: Wilcoxon tests
* Jupyter Notebook



## Getting Started

### Prerequisites

* Python 3.12+
* scikit-learn 1.6.1
* aeon 1.3.0

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/gerryc-0/Why-so-Time-Serious.git
   cd Why-so-Time-Serious
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate        # Linux / macOS
   venv\Scripts\activate           # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Reproducing the Results

#### Step 1: Run the 8-classifier benchmark (reproduction)

This runs WEASEL 2.0, WEASEL 1.0, cBOSS, Hydra, MultiRocket, MiniRocket, Rocket and R-DST on all 114 UCR datasets. Uses `seed=1379` and `n_jobs=4`.

```bash
python test_ucr_v2.py
```

Outputs two CSV files: `ucr-8-classifiers-accuracy-1.csv` and `ucr-8-classifiers-runtime-1.csv`.

#### Step 2: Run the experiments

This evaluates WEASEL 2.0 variants (adaptive ensemble, halved ensemble, fixed ensemble sizes, TF-IDF weighting, etc.) on the UCR archive.

```bash
python run_experiments.py
```

Results are saved to `experiments/results/`.

#### Step 3: Generate analysis and figures

Open and run the two notebooks:

1. **`evaluation_v2.ipynb`** - Loads the reproduced results (2 CSVs), computes summary statisics, Wilcoxon tests and generates CD diagrams, scatter plots, and the multi-comparison matrix.
2. **`report-visuals.ipynb`** - Loads the experiment results and prduces ensemble-size comparison plots and statistical significance tables.

---

## Configuration

Key parameters shared across all scripts:

| Parameter       | Value  | Location                        |
|-----------------|--------|---------------------------------|
| `seed`  | 1379   | `test_ucr_v2.py`, `run_experiments.py` |
| `n_jobs`        | 4      | Both scripts                    |

---
