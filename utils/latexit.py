import os
import re
import pandas as pd

# Directory where Excel and PNG files are located
output_file = "_latex.tex"

# Regular expression for matching GeneSymbol_Something
sheet_pattern = re.compile(r"^[A-Z0-9]+_[a-zA-Z0-9]+$")


def generate_latex_table(df, sheet_name):
    caption = f"Table showing data for {sheet_name.replace('_', ' ')}."
    label = f"tab:{sheet_name.lower()}"

    latex_table = df.to_latex(index=False, escape=False)

    latex_code = (
        "\\begin{table}[H]\n"
        "\\centering\n"
        f"{latex_table}\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\end{table}\n"
    )
    return latex_code


def generate_latex_image(image_filename):
    image_name = os.path.basename(image_filename)
    image_path_in_latex = f"Figures/{image_name.split('_')[0]}/{image_name}"
    caption = f"Figure showing {os.path.splitext(image_name)[0].replace('_', ' ')}."
    label = f"fig:{os.path.splitext(image_name)[0].lower()}"

    latex_code = (
        "\\begin{figure}[H]\n"
        "\\centering\n"
        f"\\includegraphics[width=0.8\\textwidth]{{{image_path_in_latex}}}\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\end{figure}\n"
    )
    return latex_code


def main(input_dir):
    tables_latex = []
    figures_latex = []

    # Process Excel files
    for filename in os.listdir(input_dir):
        if filename.endswith(".xlsx") or filename.endswith(".xls"):
            filepath = os.path.join(input_dir, filename)
            excel_file = pd.ExcelFile(filepath)
            for sheet_name in excel_file.sheet_names:
                if sheet_pattern.match(sheet_name):
                    df = pd.read_excel(filepath, sheet_name=sheet_name)
                    latex_table = generate_latex_table(df, sheet_name)
                    tables_latex.append(latex_table)

    # Process PNG files
    for filename in os.listdir(input_dir):
        if filename.endswith(".png"):
            name_without_ext = os.path.splitext(filename)[0]
            if not sheet_pattern.match(name_without_ext):
                latex_img = generate_latex_image(filename)
                figures_latex.append(latex_img)

    # Write LaTeX output
    with open(os.path.join(input_dir, output_file), "w", encoding="utf-8") as f:
        if tables_latex:
            f.write("\\section{Tables}\n\n")
            for latex_chunk in tables_latex:
                f.write(latex_chunk)
                f.write("\n\n")

        if figures_latex:
            f.write("\\section{Figures}\n\n")
            for latex_chunk in figures_latex:
                f.write(latex_chunk)
                f.write("\n\n")