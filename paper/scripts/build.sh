#!/usr/bin/env bash
# Rebuild research_paper.docx and research_paper.tex (+ PDF check) from research_paper.md.
# Run from the paper/ folder. Needs: pandoc, python3 (matplotlib, python-docx), xelatex (optional).
set -euo pipefail
cd "$(dirname "$0")/.."

# 1) Word version: equations rendered as images, inline math as Unicode
python3 scripts/build_docx_md.py research_paper.md figures/equations _docx_tmp.md
pandoc _docx_tmp.md -o research_paper.docx --reference-doc=scripts/reference.docx --resource-path=.
python3 scripts/polish_docx.py research_paper.docx
rm -f _docx_tmp.md

# 2) LaTeX version (real math). Compile with XeLaTeX, or pdfLaTeX on Overleaf.
pandoc research_paper.md -s -o research_paper.tex \
  -V documentclass=article -V fontsize=11pt -V geometry:margin=2.2cm -V linestretch=1.1 \
  -V mainfont="TeX Gyre Termes" -V mathfont="DejaVu Math TeX Gyre" \
  -V colorlinks=true -V linkcolor=black -V urlcolor=blue \
  -V header-includes='\usepackage{float}\floatplacement{figure}{H}'
sed -i 's/^\\usepackage{lmodern}$/\\ifPDFTeX\\usepackage{lmodern}\\fi/' research_paper.tex
sed -i -E 's/\\caption\{Fig\. [0-9]+\. /\\caption{/' research_paper.tex

if command -v xelatex >/dev/null; then
  xelatex -interaction=nonstopmode research_paper.tex >/dev/null || true
  xelatex -interaction=nonstopmode research_paper.tex >/dev/null || true
  rm -f research_paper.aux research_paper.log research_paper.out
fi
echo "built research_paper.docx, research_paper.tex, research_paper.pdf"
