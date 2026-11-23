# AI for Drug Discovery — Computational Notebooks

A collection of end-to-end computational drug discovery notebooks covering the full pipeline from target characterisation to clinical analysis. Each notebook is self-contained, heavily commented, and designed to run on Google Colab or a local Python environment.

---

## Repository Structure

```
my_repo/
│
├── 1-Future-of-Drug-Discovery/
│   └── 1_future_drug_discovery.ipynb
│
├── 2-Dose-Response-Analysis-and-IC50-Fitting/
│   ├── 2_dose_response_ic50.ipynb
│   └── synthesize_data/
│       ├── 2_00_synthesize_data.py
│       └── data/
│           ├── compound_A_clean.csv
│           ├── compound_B_noisy.csv
│           ├── compound_C_hook.csv
│           ├── compound_D_incomplete.csv
│           └── compound_E_inactive.csv
│
├── 3-Molecular-Representations-for-ML/
│   └── 3_molecular_representations.ipynb
│
├── 4-QSAR-Modeling/
│   └── 4_qsar_modeling.ipynb
│
├── 5-Virtual-Screening-for-QSAR/
│   └── 5_virtual_screening.ipynb
│
├── 6-Cellpainting/
│   └── 6_cellpainting.ipynb
│
├── 7-ADMET-Property/
│   └── 7_admet_prediction.ipynb
│
├── 8-Molecular-Generation/
│   └── 8_molecular_generation.ipynb
│
├── 9-LLMs-for-Scientific-Literature-Mining/
│   └── 9_llm_literature_mining.ipynb
│
├── 10-Molecular-Docking/
│   └── 10_molecular_docking.ipynb
│
├── 11-End-to-End-Report/
│   └── 11_end_to_end_report.ipynb
│
├── 12-Preclinical-Studies/
├── 13-Clinical-Trials/
├── 14-Protein-Structure-Prediction/
├── 15-Zero-Shot-Drug-Binding-Proteins/
└── 16-Spatial-Pharmacokinetics/
```

---

## Notebooks at a Glance

| # | Topic | Key Tools | GPU Needed |
|---|-------|-----------|------------|
| 1 | Protein language models — ESM2 embeddings, UMAP, zero-shot mutation scoring | `transformers`, `torch`, `umap-learn` | Recommended |
| 2 | Dose-response analysis — 4PL curve fitting, IC50, confidence intervals, LLM QC | `scipy`, `anthropic` | No |
| 3 | Molecular representations — SMILES, Morgan fingerprints, physicochemical descriptors, ChEMBL | `rdkit`, `chembl-webresource-client` | No |
| 4 | QSAR modelling — XGBoost regression/classification, neural network, scaffold split, SHAP | `xgboost`, `torch`, `shap` | Optional |
| 5 | Virtual screening — Lipinski filter, feature computation, compound ranking | `rdkit`, `xgboost` | No |
| 6 | Cell Painting — morphological profiling, PCA, UMAP, K-means clustering | `scikit-learn`, `umap-learn` | No |
| 7 | ADMET prediction — Lipinski/Veber/PAINS filters, radar charts | `rdkit` | No |
| 8 | Molecular generation — VAE, fragment-based design, scaffold hopping with LLM | `torch`, `anthropic` | Optional |
| 9 | LLM literature mining — NER, drug-target extraction, PubMed API | `anthropic`, `beautifulsoup4` | No |
| 10 | Molecular docking — PDB download, 3D ligand prep, AutoDock Vina workflow | `rdkit`, `biopython` | No |
| 11 | End-to-end report — composite scoring, multi-panel figure, compound prioritisation | `rdkit`, `matplotlib` | No |

---

## Detailed Notebook Descriptions

### 1 — Protein Language Models and EGFR Mutation Analysis

Uses **ESM2** (Meta AI, 650M parameters) to embed protein sequences as 1280-dimensional vectors. Demonstrates how single amino acid changes — such as the T790M drug resistance mutation — create measurable shifts in embedding space without any labelled disease data.

**What you will run:**
- Load ESM2 from HuggingFace (`facebook/esm2_t33_650M_UR50D`)
- Fetch the EGFR kinase domain from UniProt (P00533) and build T790M, C797S, L858R mutants
- Compute pairwise cosine similarity between EGFR variants, HER2, KRAS, and a random control
- Project all proteins to 2D with UMAP
- Zero-shot mutation scoring: rank resistance mutations by evolutionary disfavour

**Key outputs:** similarity heatmap, UMAP scatter plot, mutation distance bar chart, MLM mutation scores

---

### 2 — Dose-Response Analysis and IC50 Fitting

Implements the **4-Parameter Logistic (4PL)** model, the industry standard for fitting dose-response curves. Covers five realistic experimental scenarios: clean sigmoid, noisy data, hook effect, incomplete curve, and inactive compound.

**What you will run:**
- Generate or load five synthetic compound datasets (pre-generated CSVs are included)
- Fit 4PL curves with `scipy.optimize.curve_fit` using data-driven initial guesses
- Compute IC50 with 95% confidence intervals (t-distribution)
- Calculate R² and RMSE goodness-of-fit metrics
- Send curve images to Claude Sonnet for visual quality assessment (requires API key)

**Data files:**
- `synthesize_data/data/compound_A_clean.csv` — ideal reference curve
- `synthesize_data/data/compound_B_noisy.csv` — realistic lab noise
- `synthesize_data/data/compound_C_hook.csv` — hook effect artifact
- `synthesize_data/data/compound_D_incomplete.csv` — IC50 outside tested range
- `synthesize_data/data/compound_E_inactive.csv` — inactive compound

Run `synthesize_data/2_00_synthesize_data.py` to regenerate all CSV files from scratch.

---

### 3 — Molecular Representations for Machine Learning

Covers the three main ways to convert molecules into ML-ready numerical vectors. Uses real EGFR inhibitors (erlotinib, gefitinib, osimertinib) as examples throughout.

**What you will run:**
- Parse SMILES strings with RDKit and inspect molecular graphs
- Generate Morgan (circular) fingerprints — radius 2, 2048 bits
- Compute Tanimoto similarity between all drug pairs
- Calculate Lipinski Rule of Five descriptors (MW, logP, HBD, HBA, TPSA)
- Pull EGFR IC50 data from ChEMBL (CHEMBL203) and assemble an ML-ready CSV

---

### 4 — QSAR Modelling

Complete **Quantitative Structure-Activity Relationship** pipeline. Merged from two stages (data preparation + modelling) into a single notebook.

**What you will run:**
- Fetch and deduplicate EGFR IC50 data from ChEMBL (geometric mean per compound)
- Compute pIC50 (−log10 of IC50 in Molar)
- Murcko scaffold split — ensures test set contains novel scaffolds (not random split)
- **XGBoost regression** — predict continuous pIC50 values
- **XGBoost classification** — predict active (pIC50 ≥ 6) vs inactive
- **PyTorch neural network** — fully connected, 512→256→128→64 architecture with BatchNorm and Dropout
- **SHAP feature importance** — which fingerprint bits and descriptors drive predictions?

---

### 5 — Virtual Screening for QSAR

Applies a trained QSAR model to rank an entire compound library without running any lab experiments.

**What you will run:**
- Retrain a quick QSAR model on ChEMBL EGFR data (self-contained, no saved model needed)
- Load a screening library from ChEMBL approved drugs
- Apply Lipinski pre-filter to remove non-drug-like compounds
- Compute features for all filtered compounds
- Score every compound and rank by predicted pIC50
- Visualise the distribution and top-20 predicted EGFR inhibitors

---

### 6 — Cell Painting

**High-content imaging** analysis — inferring drug mechanism of action from cell morphology without knowing the target. Covers both image feature extraction concepts (Part A) and downstream analysis (Part B) in one notebook.

**What you will run:**
- Simulate a 150-compound × 200-feature Cell Painting dataset with five known mechanism groups
- Normalise features with StandardScaler
- Reduce dimensions with PCA (200 → 50 components)
- Project to 2D with UMAP coloured by mechanism of action
- K-means clustering to recover MoA groups (silhouette score validation)

---

### 7 — ADMET Property Prediction

Predicts **Absorption, Distribution, Metabolism, Excretion, and Toxicity** properties — the properties responsible for ~90% of clinical trial failures.

**What you will run:**
- Lipinski Rule of Five filter
- Veber rules (rotatable bonds + TPSA)
- PAINS (Pan-Assay INterference Substances) filter via RDKit FilterCatalog
- Rule-based ADMET scoring: oral absorption, BBB penetration, metabolic stability, solubility, hERG risk
- Radar (spider) chart comparison of ADMET profiles across compounds

---

### 8 — Molecular Generation

Three approaches to **designing new drug-like molecules** computationally.

**What you will run:**
- **LLM-based scaffold hopping** — ask Claude Sonnet to propose T790M-overcoming modifications to erlotinib (requires API key)
- **Fragment library analysis** — characterise EGFR inhibitor pharmacophore fragments (quinazoline hinge binders, covalent warheads, solubilising groups)
- **Variational Autoencoder (VAE)** — train a fingerprint VAE, then sample new molecular fingerprint vectors from the latent space

---

### 9 — LLMs for Scientific Literature Mining

Automates extraction of structured drug discovery data from unstructured text using large language models.

**What you will run:**
- Fetch PubMed abstracts via the E-utilities API (PMID-based, no key required)
- Named entity recognition (NER) — extract drugs, targets, diseases, mutations, assays
- Structured relation extraction — build a drug-target database from free text
- Output is saved to `drug_target_relations.csv`

Requires an Anthropic API key. Cells skip gracefully if no key is found.

---

### 10 — Molecular Docking

Predicts **how and where a small molecule binds** to a protein structure.

**What you will run:**
- Download EGFR kinase domain structure from PDB (1IVO — erlotinib co-crystal)
- Prepare ligands: SMILES → 3D coordinates (ETKDGv3) → MMFF94 energy minimisation → SDF files
- Define AutoDock Vina docking box around the ATP binding site
- Analyse docking scores and rank compounds by predicted binding affinity
- Visualise score comparison across erlotinib, gefitinib, osimertinib, and a negative control

For real docking runs, install AutoDock Vina and use the saved SDF files with PDBQT conversion.

---

### 11 — End-to-End Drug Discovery Report

Integrates all previous analyses into a **single prioritisation pipeline** and generates a multi-panel summary figure.

**What you will run:**
- Assemble potency (pIC50), docking score, hERG risk, and solubility for six candidate compounds
- Compute a composite priority score with weighted contributions from each dimension
- Generate a 6-panel summary figure: potency bars, docking bars, composite score, chemical space scatter, IC50 bars, top-3 table
- Save `compound_report.csv` and `drug_discovery_report.png`

---

## Setup and Installation

### Option A — Google Colab (recommended)

Each notebook installs its own dependencies in the first cell using `pip`. Simply open the notebook in Colab and run all cells.

For notebooks that use an LLM (2, 8, 9), store your API key in a `.env` file on Google Drive:

```
ANTHROPIC_API_KEY=sk-ant-...
```

Then update the `ENV_PATH` variable in the first cell to point to that file.

### Option B — Local Python

Python 3.10+ recommended.

```bash
pip install rdkit chembl-webresource-client xgboost shap torch torchvision \
            transformers biopython umap-learn scikit-learn pandas numpy \
            matplotlib seaborn scipy anthropic openai requests beautifulsoup4
```

For notebooks that require an API key, either set the environment variable directly:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

or create a `.env` file in the same directory as the notebook:

```
ANTHROPIC_API_KEY=sk-ant-...
```

---

## Dependencies Summary

| Library | Used in | Purpose |
|---------|---------|---------|
| `rdkit` | 2–8, 10–11 | Cheminformatics — SMILES, fingerprints, descriptors, 3D structure |
| `chembl-webresource-client` | 3–5 | Pull IC50 data from ChEMBL |
| `xgboost` | 4–5 | Gradient boosted trees for QSAR |
| `torch` | 1, 4, 8 | Neural network training (VAE, MolNet) |
| `transformers` | 1 | Load ESM2 protein language model |
| `shap` | 4 | Feature importance explanation |
| `umap-learn` | 1, 6 | 2D projection of high-dimensional embeddings |
| `scikit-learn` | 4–7 | PCA, StandardScaler, metrics |
| `scipy` | 2 | 4PL curve fitting |
| `biopython` | 1, 10 | Fetch EGFR sequence from NCBI/UniProt |
| `anthropic` | 2, 8, 9 | Claude Sonnet API for LLM-assisted analysis |
| `beautifulsoup4` | 9 | Parse PubMed HTML abstracts |
| `matplotlib` / `seaborn` | all | Plotting |
| `pandas` / `numpy` | all | Data handling |

---

## Data Sources

| Source | Used in | Description |
|--------|---------|-------------|
| [ChEMBL](https://www.ebi.ac.uk/chembl/) | 3, 4, 5 | Experimental IC50 data for EGFR (CHEMBL203) |
| [UniProt / NCBI](https://www.uniprot.org/uniprot/P00533) | 1 | EGFR protein sequence (P00533) |
| [PDB](https://www.rcsb.org/) | 10 | EGFR crystal structure (PDB 1IVO) |
| [PubMed](https://pubmed.ncbi.nlm.nih.gov/) | 9 | Biomedical abstracts via E-utilities API |
| [HuggingFace](https://huggingface.co/facebook/esm2_t33_650M_UR50D) | 1 | ESM2 protein language model weights |
| Synthetic (included) | 2 | Five dose-response CSV datasets |

---

## Notes

- All notebooks use `warnings.filterwarnings('ignore')` to suppress non-critical messages — remove this line if you want to see full output during debugging.
- The ESM2 model in notebook 1 is ~2.6 GB. It downloads automatically on first run and is cached by HuggingFace.
- Notebooks that call ChEMBL (3, 4, 5) require an internet connection. All include fallback data if the API is unavailable.
- LLM cells (2, 8, 9) skip gracefully with a message if no API key is set — the rest of the notebook still runs.
- Docking in notebook 10 is simulated with realistic literature-matched scores. For live docking, install AutoDock Vina and use the SDF files the notebook generates.
