# How to Scale Your Model: Solutions and Notes

My worked solutions for [How to Scale Your Model](https://jax-ml.github.io/scaling-book/).

I completed the full book by hand, then converted the written solutions into
LaTeX. This repository keeps both versions because the scans show the original
reasoning, while the typed writeups include corrections and cleaner derivations
from later verification.

## 1. Repository contents

- `handwritten/`: original scanned solutions.
- `latex/`: standalone LaTeX source files.
- `output/`: compiled PDFs generated from the LaTeX sources.
- Section 10: JAX implementations and TPU benchmarks hosted on Kaggle.
- `VERIFICATION.md`: assumptions, corrections, and verification notes.

## 2. Solutions

| Section | Handwritten | LaTeX | Compiled PDF | Coverage |
| --- | --- | --- | --- | --- |
| 1 | [Scan](sections/section_01/handwritten/section_01_handwritten.pdf) | [Source](sections/section_01/latex/section_01.tex) | [PDF](sections/section_01/output/section_01.pdf) | Exercises |
| 2 | [Scan](sections/section_02/handwritten/section_02_handwritten.pdf) | [Source](sections/section_02/latex/section_02.tex) | [PDF](sections/section_02/output/section_02.pdf) | Exercises |
| 3 | [Scan](sections/section_03/handwritten/section_03_handwritten.pdf) | [Source](sections/section_03/latex/section_03.tex) | [PDF](sections/section_03/output/section_03.pdf) | Exercises 1-10 |
| 4 | [Scan](sections/section_04/handwritten/section_04_handwritten.pdf) | [Source](sections/section_04/latex/section_04.tex) | [PDF](sections/section_04/output/section_04.pdf) | Exercises 1-8 |
| 5 | [Scan](sections/section_05/handwritten/section_05_handwritten.pdf) | [Source](sections/section_05/latex/section_05.tex) | [PDF](sections/section_05/output/section_05.pdf) | Worked Problems 1-3 |
| 6 | [Scan](sections/section_06/handwritten/section_06_handwritten.pdf) | [Source](sections/section_06/latex/section_06.tex) | [PDF](sections/section_06/output/section_06.pdf) | Worked Problems 1-2 |
| 7 | [Scan](sections/section_07/handwritten/section_07_handwritten.pdf) | [Source](sections/section_07/latex/section_07.tex) | [PDF](sections/section_07/output/section_07.pdf) | Worked Problems 1-5 |
| 8 | [Scan](sections/section_08/handwritten/section_08_handwritten.pdf) | [Source](sections/section_08/latex/section_08.tex) | [PDF](sections/section_08/output/section_08.pdf) | Worked Problems 1-3 |
| 9 | [Scan](sections/section_09/handwritten/section_09_q1_handwritten.pdf) | [Source](sections/section_09/latex/section_09.tex) | [PDF](sections/section_09/output/section_09.pdf) | Worked Problem 1 |
| 10 | See [JAX and TPU notebooks](sections/section_10/README.md) | Kaggle notebooks | Kaggle outputs | Worked Problems 1-4 |
| 11 | No exercises | No exercises | No exercises | Conclusions |
| 12 | [Scan](sections/section_12/handwritten/section_12_handwritten.pdf) | [Five quiz sources](sections/section_12/latex/) | [Five quiz PDFs](sections/section_12/output/) | GPU quizzes 1-5 |

Section 7 also includes a separate [cheatsheet source](sections/section_07/latex/section_07_cheatsheet.tex)
and [compiled cheatsheet](sections/section_07/output/section_07_cheatsheet.pdf).

## 3. Notes

These are unofficial solutions. Several typed answers correct arithmetic or
make assumptions explicit when the original handwritten work was ambiguous.
The exact changes are documented in `VERIFICATION.md`.

Section 10 is code-first, so its implementations remain in Kaggle notebooks
rather than LaTeX files.

## 4. Building the LaTeX

See [BUILDING.md](BUILDING.md) for local compilation commands.
