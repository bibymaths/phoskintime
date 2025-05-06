# PhosKinTime Data Preprocessing & Mapping

This workflow prepares and maps time-series data for kinase and transcription factor optimization models from raw
proteomics and transcriptomics datasets.

---

## Structure

```
phoskintime/
├── processing/
│   ├── cleanup.py     # Data cleaning and preparation
│   └── map.py         # Optimization result mapping and network table generation
├── raw/               # Input CSVs (CollecTRI, MS Gaussian, Rout Limma)
├── kinopt/data/       # Kinase model inputs
├── tfopt/data/        # TF model inputs
└── data/              # Network export for Cytoscape
```

---

## Scripts Overview

### `cleanup.py`

Performs the following steps:

1. **TF-mRNA Interaction Cleanup**
    - Filters complex interactions in CollecTRI
    - Keeps only TFs matching phospho-interactions in `input2.csv`

2. **Proteomics Data Transformation**
    - Transforms MS Gaussian predictions with `2^mean`
    - Formats phosphorylation sites, saves to `input1.csv`

3. **Error Propagation**
    - Computes std propagation:  
      `σ_y = 2^x * ln(2) * σ_x`
    - Saves to `input1_wstd.csv`

4. **Transcriptomics Cleanup**
    - Transforms Rout Limma values with `2^x`
    - Saves to `input3.csv`

5. **Gene Symbol Mapping**
    - Replaces Ensembl/Entrez IDs with gene symbols (using MyGeneInfo)

6. **File Management**
    - Moves cleaned files to `kinopt/data/` and `tfopt/data/`

### `map.py`

This script processes optimization results for transcription factors (TFs) and kinases, mapping their interactions with mRNA and phosphorylation sites. It generates Cytoscape-compatible edge and node tables for network visualization.

#### Key Features:
- **TF-mRNA Mapping**: Extracts non-zero optimization results and groups mRNA by associated TFs and their strengths.
- **Kinase-Phosphorylation Mapping**: Maps kinases to mRNA and phosphorylation sites based on optimization results.
- **Cytoscape Table Generation**: Creates edge and node tables for network visualization, including interaction types and strengths.
- **Kinetic Strength Integration**: Adds kinetic strength columns to mapping files for further analysis.

---

## Inputs

Place the following raw data in `processing/raw/`:

- `CollecTRI.csv`
- `MS_Gaussian_updated_09032023.csv`
- `Rout_LimmaTable.csv`
- `input2.csv` (phospho interactions)

---

## Outputs

| File              | Description                          |
|-------------------|--------------------------------------|
| `input1.csv`      | Phospho time series (KinOpt, TFOpt)  |
| `input1_wstd.csv` | Same as above + standard deviation   |
| `input2.csv`      | Phospho kinase-interaction metadata  |
| `input3.csv`      | mRNA time series (TFOpt)             |
| `input4.csv`      | Clean TF-mRNA interactions           |
| `mapping.csv`     | Mapped TF → mRNA with Kinase + Psite |
| `mapping_.csv`    | Cytoscape-compatible edge list       |
| `nodes.csv`       | Cytoscape node roles                 |

---

## Notes

- Complex TF interactions (e.g. `COMPLEX:TF1/TF2`) are excluded.
- Kinase-only proteins not appearing in CollecTRI (e.g. `PAK2`) are excluded from TF mapping.
- Unmappable GeneIDs are printed at runtime.