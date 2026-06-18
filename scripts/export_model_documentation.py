#!/usr/bin/env python3
"""Generate Markdown, LaTeX, and optional PDF documentation for fitted models."""
from __future__ import annotations
import argparse, datetime as dt, json, shutil, subprocess, sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Reuse the exporter discovery without requiring package installation.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_model_to_sbml import FAMILIES, SOURCE_HINTS, SCHEMA_SOURCES, discover_model_outputs, ModelResult  # noqa: E402

TERMS = {
 "protwise": [("R_i(t)", "unitless mRNA observable"), ("P_i(t)", "unitless protein observable"), ("S_{i,j}(t)", "unitless phosphorylation-site observable"), ("A/B/C/D/E/Dp", "fitted kinetic coefficients preserved as exported")],
 "kinopt": [("P_i(t)", "observed/predicted phosphosite or protein target fold-change"), ("K_{k,s}(t)", "kinase phosphosite fold-change signal"), ("alpha", "target-to-kinase regulatory weight"), ("beta", "kinase-site activity coefficient")],
 "tfopt": [("E_g(t)", "gene expression fold-change"), ("TF_r(t)", "TF protein fold-change"), ("PSite_{r,s}(t)", "TF phosphosite fold-change"), ("alpha", "gene-to-TF regulatory weight"), ("beta", "TF protein/phosphosite contribution weight")],
 "networkmodel": [("R_i(t)", "unitless RNA state"), ("P_i(t)", "unitless protein state"), ("S_{i,j}(t)", "unitless phosphosite state"), ("A_i,B_i,C_i,D_i,E_i,Dp_i,tf_scale", "physical-space fitted parameters from networkmodel parameter unpacking")],
}
CONSTRAINTS = {
 "kinopt": "Alpha and beta parameters are optimized with bounds from the KinOpt optimizer; predictions are clipped at zero in the objective.",
 "tfopt": "Alpha values are bounded in [0,1] and constrained to sum to one per gene; beta vectors are bounded by configured lower/upper bounds and constrained to sum to one per TF.",
 "protwise": "Parameter names and bounds are selected by the configured ODE model; exported values are read from result files without refitting or rescaling.",
 "networkmodel": "Optimized raw parameters are unpacked through softplus into positive physical-space arrays; configured bounds apply in raw optimizer space.",
}

@dataclass
class ModelDocumentation:
    result: ModelResult
    source_files: list[str]
    notes: list[str]
    equation_source_files: list[Path]
    equation_source_functions: list[str]
    parameter_source_files: list[Path]
    schema_source_functions: list[str]


def read_source_summary(fam: str) -> tuple[list[str], list[str]]:
    files=[]; notes=[]
    for rel in SOURCE_HINTS.get(fam, []):
        p=Path(rel)
        if p.exists():
            files.append(rel)
            try:
                text=p.read_text(encoding="utf-8", errors="replace")
                hits=[line.strip() for line in text.splitlines() if any(tok in line.lower() for tok in ("def ", "constraint", "bounds", "objective", "estimated", "rhs", "unpack"))][:12]
                notes.extend(f"{rel}: {h}" for h in hits)
            except Exception as exc: notes.append(f"Could not inspect {rel}: {exc}")
        else: notes.append(f"Source hint missing: {rel}")
    return files, notes


def md_table(params: dict[str,float]) -> str:
    if not params: return "No numeric fitted parameters were detected.\n"
    rows=["| Parameter | Value |", "| --- | ---: |"]
    for k,v in sorted(params.items())[:1000]: rows.append(f"| `{k}` | `{v!r}` |")
    if len(params)>1000: rows.append(f"| … | {len(params)-1000} additional parameters omitted from display |")
    return "\n".join(rows)+"\n"


def render_markdown(docs: list[ModelDocumentation], results_dir: Path) -> str:
    now=dt.datetime.now(dt.timezone.utc).isoformat()
    lines=["# Fitted Model Documentation", "", f"Generated: `{now}`", f"Results directory: `{results_dir}`", "", "## Measurement-scale warning", "", "The fitted measurements are fold-change or otherwise unitless values. They do **not** imply absolute molecular concentrations. Exported variables and parameters should be interpreted as dimensionless mathematical/regulatory quantities unless independent calibration information is supplied.", "", "## Project schema sources", ""] + [f"- `{src}`" for src in SCHEMA_SOURCES] + [""]
    for d in docs:
        r=d.result; fam=r.model_family
        lines += [f"## {fam}", "", "### Source files inspected", ""]
        lines += [f"- `{s}`" for s in d.source_files] or ["- No source files were available for inspection."]
        lines += ["", "### Implementation grounding", "", "#### Source files inspected", ""] + [f"- `{s}`" for s in d.source_files]
        lines += ["", "#### Equation authority", ""] + [f"- `{fn}`" for fn in d.equation_source_functions or ["No family-specific equation authority detected"]]
        lines += ["", "#### Schema/export authority", ""] + [f"- `{fn}`" for fn in d.schema_source_functions]
        lines += ["", "#### Parameter source files", ""] + [f"- `{p}`" for p in d.parameter_source_files or [Path(x) for x in r.metadata.get("result_files", [])]]
        if r.structure_files:
            lines += ["", "#### Structure/metadata files used for model context", ""] + [f"- `{p}`" for p in r.structure_files]
        lines += ["", "### Result files used", ""] + [f"- `{p}`" for p in r.metadata.get("result_files", [str(r.source_path)])]
        lines += ["", "### Detected fitted-parameter schemas", ""] + [f"- `{schema}`" for schema in r.metadata.get("schemas", []) or ["No fitted-parameter schema detected"]]
        lines += ["", "### Mathematical equations", ""] + [f"- `${eq}$`" for eq in r.equations]
        lines += ["", "### Term explanations", ""] + [f"- `{t}`: {desc}" for t,desc in TERMS.get(fam, [])]
        lines += ["", "### Fitted parameter table", "", md_table(r.parameters)]
        lines += ["", "### Constraints used during fitting", "", CONSTRAINTS.get(fam, "No family-specific constraints could be reconstructed from inspected source files."), ""]
        lines += ["### Assumptions and limitations", "", "- Results are documented exactly as found; this script does not refit models or regenerate parameters.", "- Unitless/fold-change observables are represented as dimensionless and cannot be interpreted as absolute concentrations.", "- If a model is algebraic, SBML export uses the closest assignment-rule style representation; if dynamic, equations are documented as ODE right-hand-side relationships.", ""]
        if r.skipped_tables:
            lines += ["### Skipped non-parameter result tables", ""] + [f"- `{s.file}`" + (f" sheet `{s.sheet}`" if s.sheet else "") + f": {s.reason}" for s in r.skipped_tables[:50]] + [""]
        if r.warnings or d.notes:
            lines += ["### Notes about missing or unavailable information", ""] + [f"- {w}" for w in r.warnings + d.notes[:20]] + [""]
        lines += ["### Reproducibility metadata", "", f"- Parameter count: `{len(r.parameters)}`", f"- Documentation generated by: `scripts/export_model_documentation.py`", ""]
    return "\n".join(lines)


def esc(s: Any) -> str:
    return str(s).replace('\\','\\textbackslash{}').replace('&','\\&').replace('%','\\%').replace('$','\\$').replace('#','\\#').replace('_','\\_').replace('{','\\{').replace('}','\\}')


def render_latex(docs: list[ModelDocumentation], results_dir: Path) -> str:
    out=[r"\documentclass{article}", r"\usepackage[margin=1in]{geometry}", r"\usepackage{longtable}", r"\usepackage{amsmath}", r"\usepackage[T1]{fontenc}", r"\begin{document}", r"\title{Fitted Model Documentation}", r"\maketitle", r"\section*{Measurement-scale warning}", "The fitted measurements are fold-change or otherwise unitless values. They do not imply absolute molecular concentrations. Exported variables and parameters are dimensionless mathematical/regulatory quantities unless independent calibration is supplied.", r"\section*{Project schema sources}", r"\begin{itemize}"] + [f"\\item \\texttt{{{esc(src)}}}" for src in SCHEMA_SOURCES] + [r"\end{itemize}"]
    for d in docs:
        r=d.result; fam=r.model_family; out += [f"\\section*{{{esc(fam)}}}", r"\subsection*{Source files inspected}", "\\begin{itemize}"]
        out += [f"\\item \\texttt{{{esc(s)}}}" for s in d.source_files] or [r"\item None available"]
        out += [r"\end{itemize}", r"\subsection*{Result files used}", r"\begin{itemize}"] + [f"\\item \\texttt{{{esc(p)}}}" for p in r.metadata.get("result_files", [str(r.source_path)])] + [r"\end{itemize}"]
        out += [r"\subsection*{Detected fitted-parameter schemas}", r"\begin{itemize}"] + [f"\\item \\texttt{{{esc(schema)}}}" for schema in (r.metadata.get("schemas", []) or ["No fitted-parameter schema detected"])] + [r"\end{itemize}"]
        out += [r"\subsection*{Equations}"] + [f"\\[{esc(eq)}\\]" for eq in r.equations]
        out += [r"\subsection*{Terms}", r"\begin{itemize}"] + [f"\\item \\texttt{{{esc(t)}}}: {esc(desc)}" for t,desc in TERMS.get(fam, [])] + [r"\end{itemize}"]
        out += [r"\subsection*{Parameters}", r"\begin{longtable}{p{0.65\linewidth}r}", r"Parameter & Value\\ \hline"]
        for k,v in sorted(r.parameters.items())[:800]: out.append(f"\\texttt{{{esc(k)}}} & \\texttt{{{esc(v)}}}\\\\")
        out += [r"\end{longtable}", r"\subsection*{Skipped non-parameter result tables}", r"\begin{itemize}"] + [f"\\item \\texttt{{{esc(s.file)}}}" + (f" sheet \\texttt{{{esc(s.sheet)}}}" if s.sheet else "") + f": {esc(s.reason)}" for s in r.skipped_tables[:50]] + [r"\end{itemize}", r"\subsection*{Constraints and assumptions}", esc(CONSTRAINTS.get(fam, "No family-specific constraints reconstructed.")), " Unitless/fold-change observables are dimensionless; no absolute concentration units are implied."]
    out.append(r"\end{document}")
    return "\n".join(out)


def compile_pdf(tex_path: Path) -> Path | None:
    engine = shutil.which("tectonic") or shutil.which("pdflatex")
    if not engine:
        print("PDF generation skipped: no LaTeX engine found (install tectonic or pdflatex).")
        return None
    cmd = [engine, str(tex_path.name)] if Path(engine).name == "tectonic" else [engine, "-interaction=nonstopmode", str(tex_path.name)]
    subprocess.run(cmd, cwd=tex_path.parent, check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    pdf=tex_path.with_suffix(".pdf")
    if pdf.exists(): return pdf
    print("PDF generation attempted but no PDF was produced; inspect LaTeX logs in the output directory.")
    return None


def main(argv=None) -> int:
    ap=argparse.ArgumentParser(); ap.add_argument("--results-dir", required=True, type=Path); ap.add_argument("--output-dir", required=True, type=Path); ap.add_argument("--model-family", choices=FAMILIES); ap.add_argument("--run-id"); ap.add_argument("--overwrite", action="store_true"); ap.add_argument("--verbose", action="store_true"); ap.add_argument("--no-pdf", action="store_true")
    a=ap.parse_args(argv); a.output_dir.mkdir(parents=True, exist_ok=True)
    results=discover_model_outputs(a.results_dir, a.model_family, a.run_id, a.verbose); docs=[]; found={r.model_family for r in results if r.parameters}
    for fam in FAMILIES:
        if a.model_family and fam != a.model_family: continue
        if fam not in found: print(f"WARNING: {fam}: skipped, no fitted result files found")
    for r in results:
        sources, notes=read_source_summary(r.model_family)
        docs.append(ModelDocumentation(r, sources, notes, r.equation_source_files, r.equation_source_functions, r.parameter_source_files, r.schema_source_functions))
    md=a.output_dir/"model_documentation.md"; tex=a.output_dir/"model_documentation.tex"
    for path, text in [(md, render_markdown(docs, a.results_dir)), (tex, render_latex(docs, a.results_dir))]:
        if path.exists() and not a.overwrite: raise FileExistsError(f"{path} exists; use --overwrite")
        path.write_text(text, encoding="utf-8")
    pdf=None if a.no_pdf else compile_pdf(tex)
    print("Fixed implementation grounding:"); print("- inspected networkmodel/protwise/common/KinOpt/TFOpt equation and export functions"); print("- recorded schema and equation source files/functions"); print("- imported existing functions where safe, otherwise mirrored schemas from implementation")
    print("Fixed result parsing:"); print("- supported results_network_combinatorial layout"); print("- supported old_results/results_kinopt/kinopt/kinopt_results.xlsx"); print("- supported old_results/results_tfopt/tfopt/tfopt_results.xlsx"); print("- supported old_results/results_model/Distributive_results protwise layout"); print("- removed generic numeric parsing from *_results.* files"); print("- added workbook sheet-level filtering")
    print("Fixed model-family handling:"); print("- --model-family is now authoritative when paths are generic"); print("- valid alpha/beta/parameter tables are not silently dropped")
    print("Outputs:"); print("- Markdown, LaTeX, and PDF documentation written where possible")
    print("Discovered model families:"); [print(f"- {fam}: {'documented' if fam in found else 'skipped, no fitted result files found'}") for fam in FAMILIES if not a.model_family or fam==a.model_family]
    if "unknown" in found: print("- unknown: documented")
    print("Outputs:"); print(f"- {md}"); print(f"- {tex}"); print(f"- {pdf}" if pdf else "- PDF skipped or unavailable")
    return 0
if __name__ == "__main__": raise SystemExit(main())
