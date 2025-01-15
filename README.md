# olas

[![pypi package version](https://img.shields.io/pypi/v/olas.svg)](https://pypi.python.org/pypi/olas)
[![conda-forge version](https://img.shields.io/conda/vn/conda-forge/olas.svg)](https://anaconda.org/conda-forge/olas)
[![python supported shield](https://img.shields.io/pypi/pyversions/olas.svg)](https://pypi.python.org/pypi/olas)

Library with wave tools. At the moment it only includes a prototype of ESTELA.

Documentation: <https://jorgeperezg.github.io/olas>

## Installation

Installation with pip (or similar tools) is straightforward:
```
pip install olas
```

It can also be installed with conda from conda-forge:
```
conda install -c conda-forge olas
```

## Basic usage
Calculate and plot ESTELA maps from the sample netcdf files.

Using the command line interface:
```sh
# estela -h to check additional options
estela "./tests/sample_files/test20180101T??.nc" 44 -4 --outdir "."
```

Or python functions:
```python
from olas.estela import calc, plot
estelas = calc("./tests/sample_files/test20180101T??.nc", 44, -4)
plot(estelas, outdir=".")
plot(estelas, outdir=".")  # gainloss=True for plotting GL maps
```
