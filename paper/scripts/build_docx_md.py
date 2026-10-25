"""Make a Word-friendly copy of research_paper.md.

Display equations ($$...$$) are rendered to PNG images with matplotlib mathtext and inline
math ($...$) is converted to plain Unicode, so the DOCX looks the same in Word, LibreOffice
and Google Docs. The LaTeX version keeps the original math.
Usage: python build_docx_md.py research_paper.md figures/equations out.md
"""
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

src, eqdir, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
eqdir.mkdir(parents=True, exist_ok=True)
text = src.read_text()

# matplotlib mathtext lacks a few LaTeX commands; map them to supported equivalents
MT_FIX = [(r"\Big", ""), (r"\big", ""), (r"\mathbb{1}", r"\mathbf{1}"), (r"\,", r"\ "), (r"\text{", r"\mathrm{")]


def render(expr: str, path: Path) -> None:
    e = expr.strip()
    for a, b in MT_FIX:
        e = e.replace(a, b)
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, f"${e}$", fontsize=13, family="serif")
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.04, transparent=False, facecolor="white")
    plt.close(fig)


counter = 0


def display(m):
    global counter
    counter += 1
    p = eqdir / f"eq{counter}.png"
    render(m.group(1), p)
    return f"\n\n![]({p.as_posix()}){{height=0.42in}}\n\n"


text = re.sub(r"\$\$(.+?)\$\$", display, text, flags=re.S)

UNI = [
    (r"\mathbf{x}", "x"), (r"\mathbf{w}", "w"), (r"\varepsilon", "ε"), (r"\mathcal{N}", "N"), (r"\sim", "~"),
    (r"\phi_0", "φ₀"), (r"\phi_j", "φⱼ"), (r"\sum_j", "Σⱼ"), (r"\bar{S}", "S̄"), (r"\ge", "≥"), (r"\le", "≤"),
    (r"\in", "∈"), (r"\{", "{"), (r"\}", "}"), (r"\text{none}", "none"), (r"\text{DPD}", "DPD"),
    (r"\text{EOD}", "EOD"), (r"\text{Cost}", "Cost"), (r"\max", "max"), (r"\,", " "), ("0.05^2", "0.05²"),
]


def inline(m):
    s = m.group(1)
    for a, b in UNI:
        s = s.replace(a, b)
    s = re.sub(r"\s+", " ", s).strip()
    return f"*{s}*" if re.fullmatch(r"[A-Za-z]", s) else s


text = re.sub(r"(?<!\$)\$([^$\n]+?)\$(?!\$)", inline, text)
out.write_text(text)
print(f"rendered {counter} display equations -> {out}")
