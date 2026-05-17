# PhosKinTime Data Preprocessing & Mapping

> **Note for contributors:** The canonical version of this file is `processing/README.md` at the
> repository root. This docs copy mirrors it. Keep both in sync when updating documentation.

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

## Input File Schemas

### `CollecTRI.csv`

TF–gene regulatory interaction database. `cleanup.py` uses these columns:

| Column | Type | Description |
|---|---|---|
| `source` | string | TF identifier (rows starting with `COMPLEX:` are excluded) |
| `source_genesymbol` | string | TF gene symbol (used as `Source` in output) |
| `target_genesymbol` | string | Target gene symbol (used as `Target` in output) |

Other columns are ignored. Only interactions where `Target` appears in `input2.csv`'s `GeneID`
column are retained.

**Common validation failures:**
- Rows where `source` starts with `COMPLEX:` — excluded automatically
- NaN or blank `source_genesymbol` / `target_genesymbol` — excluded automatically
- Duplicate rows — deduplicated automatically

---

### `MS_Gaussian_updated_09032023.csv`

Mass-spectrometry phosphoproteomics data with Gaussian-fitted log2 abundances.
`cleanup.py` uses these columns:

| Column | Type | Description |
|---|---|---|
| `GeneID` | string | Protein/gene identifier |
| `site` | string | Phosphorylation site (e.g. `S256`, `T308`) |
| `unit_time` | int | Time index (0–13, representing 14 time points) |
| `mean` | float | Log2 mean abundance at this time point |
| `std` | float | Log2 standard deviation (used for error propagation) |

The script pivots the data to wide format using `(GeneID, Psite)` as index and `unit_time` as
columns. After transformation:
- `mean` values are exponentiated: `2^mean`
- `std` is propagated: `σ_y = 2^x * ln(2) * σ_x`
- Only rows with `Psite` starting with `Y_`, `S_`, `T_`, or empty are retained
- Time columns are renamed `x1`–`x14`

**Minimal example row:**
```
GeneID,site,unit_time,mean,std
ABL2,S256,0,1.23,0.05
ABL2,S256,1,1.45,0.06
```

---

### `Rout_LimmaTable.csv`

RNA time-series derived from Rout Limma differential expression analysis.
`cleanup.py` uses these columns:

| Column | Type | Description |
|---|---|---|
| `GeneID` | string | Gene identifier |
| `Min4vsCtrl` | float | Log2 fold-change at 4 min vs control |
| `Min8vsCtrl` | float | Log2 fold-change at 8 min vs control |
| `Min15vsCtrl` | float | Log2 fold-change at 15 min vs control |
| `Min30vsCtrl` | float | Log2 fold-change at 30 min vs control |
| `Hr1vsCtrl` | float | Log2 fold-change at 1 hr vs control |
| `Hr2vsCtrl` | float | Log2 fold-change at 2 hr vs control |
| `Hr4vsCtrl` | float | Log2 fold-change at 4 hr vs control |
| `Hr8vsCtrl` | float | Log2 fold-change at 8 hr vs control |
| `Hr16vsCtrl` | float | Log2 fold-change at 16 hr vs control |

All 9 time-point columns are required. Values are exponentiated (`2^x`) to convert from log2 to
linear fold-change. Renamed to `x1`–`x9` in `input3.csv`.

**Common validation failures:**
- Missing time-point columns — script will raise `KeyError`
- Non-numeric values in time-point columns — will cause `TypeError` during `2^x` computation

---

### `input2.csv`

Kinase-substrate network / phospho-interaction metadata. Used both as raw input by `cleanup.py`
and as a standard input by `kinopt`.

| Column | Type | Description |
|---|---|---|
| `GeneID` | string | Substrate gene/protein identifier |
| `Psite` | string | Phosphorylation site identifier (e.g. `S256`) |
| `Kinase` | string | Kinase name(s) (may be a comma- or semicolon-separated set) |

Additional columns may be present and are preserved in outputs.

**Common validation failures:**
- Missing `GeneID` column — `cleanup.py` will raise `KeyError`
- Empty `GeneID` values — filtered out automatically

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