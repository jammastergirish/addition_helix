#!/bin/bash
# Two passes so cross-references and the TOC resolve.
set -e
cd "$(dirname "$0")"
xelatex -interaction=nonstopmode paper.tex
xelatex -interaction=nonstopmode paper.tex
