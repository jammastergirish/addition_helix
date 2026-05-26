#!/bin/bash
# Build paper.pdf. Two xelatex passes so cross-refs resolve. xelatex
# (not pdflatex) is required so the fontspec-based unicode fonts render
# the 12 numeral scripts inline.
set -e
cd "$(dirname "$0")"
xelatex -interaction=nonstopmode paper.tex
xelatex -interaction=nonstopmode paper.tex
