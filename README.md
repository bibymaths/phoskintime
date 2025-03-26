# PhosKinTime
ODE-based modeling and parameter estimation of phosphorylation.
 
[![CI/CD](https://github.com/bibymaths/phoskintime/actions/workflows/test.yml/badge.svg)](https://github.com/bibymaths/phoskintime/actions/workflows/test.yml)  
[![codecov](https://codecov.io/gh/bibymaths/phoskintime/branch/master/graph/badge.svg)](https://codecov.io/gh/bibymaths/phoskintime)

## Usage
```bash
usage: main.py [-h] [--A-bound A_BOUND] [--B-bound B_BOUND] [--C-bound C_BOUND] [--D-bound D_BOUND] [--Ssite-bound SSITE_BOUND] [--Dsite-bound DSITE_BOUND] [--fix-A FIX_A] [--fix-B FIX_B] [--fix-C FIX_C] [--fix-D FIX_D] [--fix-Ssite FIX_SSITE] [--fix-Dsite FIX_DSITE] [--fix-t FIX_T]
               [--bootstraps BOOTSTRAPS] [--profile-start PROFILE_START] [--profile-end PROFILE_END] [--profile-step PROFILE_STEP] [--input-excel INPUT_EXCEL]

PhosKinTime - ODE Parameter Estimation of Phosphorylation Events in Temporal Space

options:
  -h, --help            show this help message and exit
  --A-bound A_BOUND
  --B-bound B_BOUND
  --C-bound C_BOUND
  --D-bound D_BOUND
  --Ssite-bound SSITE_BOUND
  --Dsite-bound DSITE_BOUND
  --fix-A FIX_A
  --fix-B FIX_B
  --fix-C FIX_C
  --fix-D FIX_D
  --fix-Ssite FIX_SSITE
  --fix-Dsite FIX_DSITE
  --fix-t FIX_T         JSON string mapping time points to fixed param values
  --bootstraps BOOTSTRAPS
  --profile-start PROFILE_START
  --profile-end PROFILE_END
  --profile-step PROFILE_STEP
  --input-excel INPUT_EXCEL
                        Path to the input Excel file

```