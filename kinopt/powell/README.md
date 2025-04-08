# Powell Module

The **Powell** module is part of the **abopt** framework and serves as a bridge between Python and Julia for running specialized optimization routines. It leverages Julia's capabilities (via a dedicated Julia script) to perform optimization tasks and then integrates the results into the abopt post-processing workflow.

---

## Overview

The Powell module executes a Julia script (`powell.jl`) to perform optimization using a Powell-based (or similar) algorithm. It automatically configures the execution environment by determining an optimal number of Julia threads, runs the Julia optimization, and then triggers a series of post-optimization steps such as result copying, feasibility checks, file organization, and report generation.

---

## Directory Structure

```
abopt/
└── powell/
    ├── __main__.py        # Entry point for the Powell module. citeturn3file0
    ├── runpowell.py       # Contains the run_powell() function to execute the Julia script. citeturn3file1
    └── powell.jl          # (Julia script) Implements the core optimization routine.
```

---

## Features

- **Automatic Thread Configuration:**  
  The module dynamically determines the number of available CPU threads using the `lscpu` command, sets the `JULIA_NUM_THREADS` environment variable to half that number, and passes this setting to the Julia process for efficient parallel computation.

- **Julia Integration:**  
  Executes a Julia script (`powell.jl`) via a subprocess call. The real-time output from Julia is logged for monitoring purposes.

- **Post-Optimization Workflow:**  
  After the Julia optimization, the module:
  - Copies the results file to a designated ODE data directory.
  - Invokes post-optimization analysis routines from the Optimality module.
  - Organizes output files and generates a comprehensive report.

- **Logging and Reporting:**  
  Uses a custom logging configuration to provide detailed real-time feedback and final summary messages, ensuring users are informed about the location of the generated report and results.

---

## Requirements

- **Python 3.x:**  
  Required to run the main module and post-processing steps.

- **Julia:**  
  Must be installed and accessible via the command line to execute the `powell.jl` script.

- **System Utilities:**  
  The module relies on the `lscpu` command to determine the number of available CPU threads.

---

## Usage

1. **Ensure Julia is Installed:**  
   Verify that Julia is installed on your system and is available in your PATH.

2. **Run the Powell Module:**  
   From the command line, execute the module as follows:
   ```bash
   python -m kinopt.powell
   ```
   This command will:
   - Invoke the `run_powell()` function to run the Julia optimization script.
   - Copy the output file to the ODE data directory.
   - Trigger the post-optimization processing (feasibility analysis, file organization, and report generation).

3. **Monitor Logs:**  
   The module logs detailed output from the Julia script in real time and provides a final summary message with the location of the generated report.

---

## Dependencies

- **abopt.evol.config.constants:** Provides configuration constants like `OUT_FILE`, `ODE_DATA_DIR`, and `OUT_DIR`.
- **abopt.optimality.KKT:** Used for post-optimization analysis and result validation.
- **abopt.local.config.logconf:** Supplies the logging configuration.
- **Python Standard Libraries:** `subprocess`, `os`, and `shutil` for process management and file operations.

---

## Advanced Options

- **Thread Configuration:**  
  The module automatically computes the number of threads to use by halving the total number detected via `lscpu`. This ensures a balance between resource usage and performance.

- **Environment Customization:**  
  The environment variable `JULIA_NUM_THREADS` is set within the module, but it can be modified externally if needed.

---