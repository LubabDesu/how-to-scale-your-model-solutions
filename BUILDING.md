# Building the LaTeX Solutions

Each `.tex` file is standalone. A TeX Live installation with `latexmk` is the
most direct option.

From a section's `latex/` directory:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -outdir=../output section_01.tex
```

Replace `section_01.tex` with the relevant source filename. Section 12 has one
source file per quiz.

To remove local build files:

```bash
latexmk -c
```
