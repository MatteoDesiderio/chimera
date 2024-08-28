# chimera

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Tests status][tests-badge]][tests-link]
[![Linting status][linting-badge]][linting-link]
[![Documentation status][documentation-badge]][documentation-link]
[![License][license-badge]](./LICENSE.md)

<!--
[![PyPI version][pypi-version]][pypi-link]
[![Conda-Forge][conda-badge]][conda-link]
[![PyPI platforms][pypi-platforms]][pypi-link]
-->

<!-- prettier-ignore-start -->
[tests-badge]:              https://github.com/MatteoDesiderio/chimera/actions/workflows/tests.yml/badge.svg
[tests-link]:               https://github.com/MatteoDesiderio/chimera/actions/workflows/tests.yml
[linting-badge]:            https://github.com/MatteoDesiderio/chimera/actions/workflows/linting.yml/badge.svg
[linting-link]:             https://github.com/MatteoDesiderio/chimera/actions/workflows/linting.yml
[documentation-badge]:      https://github.com/MatteoDesiderio/chimera/actions/workflows/docs.yml/badge.svg
[documentation-link]:       https://github.com/MatteoDesiderio/chimera/actions/workflows/docs.yml
[conda-badge]:              https://img.shields.io/conda/vn/conda-forge/chimera
[conda-link]:               https://github.com/conda-forge/chimera-feedstock
[pypi-link]:                https://pypi.org/project/chimera/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/chimera
[pypi-version]:             https://img.shields.io/pypi/v/chimera
[license-badge]:            https://img.shields.io/badge/License-MIT-yellow.svg
<!-- prettier-ignore-end -->

A package to translate StagYY output to seismic velocities.

This project is developed in collaboration with the
[Centre for Advanced Research Computing](https://ucl.ac.uk/arc), University
College London.

## About
This simple python package allows StagYY, StagPy, axiSEM, Perple_X talk to 
each other. 

It uses StagPy to load the Pressure (P), Temperature (P), 
Compositional (C) and Density fields ouptut by the geodynamic modeling code 
StagYY. Then, these fields are interpolated on the finer AxiSEM grid.
Then, it translates the P, T fields into thermoelastic properties for 
each composition (Adiabatic K, G moduli and Density). It does so based on 
Perple_X thermodynamic tables supplied for each composition (a modified version
of the phempg is used to import the tab file). 
Then, the thermoelastic properties are averaged via a Voigt-Reuss-Hill scheme to 
create a geodynamically self-consistent seismic velocity model that can be 
later fed into AxiSEM as lateral heterogeneities. 

### Project Team

Matteo Desiderio ([ucfbmde@ucl.ac.uk](mailto:ucfbmde@ucl.ac.uk))

<!-- TODO: how do we have an array of collaborators ? -->

### Research Software Engineering Contact

Centre for Advanced Research Computing, University College London
([arc.collaborations@ucl.ac.uk](mailto:arc.collaborations@ucl.ac.uk))

## Built With

<!-- TODO: can cookiecutter make a list of frameworks? -->

- [Framework 1](https://something.com)
- [Framework 2](https://something.com)
- [Framework 3](https://something.com)

## Getting Started

### Prerequisites

<!-- Any tools or versions of languages needed to run code. For example specific Python or Node versions. Minimum hardware requirements also go here. -->

`chimera` requires Python 3.10&ndash;3.12.

### Installation

<!-- How to build or install the application. -->

We recommend installing in a project specific virtual environment created using
a environment management tool such as
[Conda](https://docs.conda.io/projects/conda/en/stable/). To install the latest
development version of `chimera` using `pip` in the currently active
environment run

```sh
pip install git+https://github.com/MatteoDesiderio/chimera.git
```

Alternatively create a local clone of the repository with

```sh
git clone https://github.com/MatteoDesiderio/chimera.git
```

and then install in editable mode by running

```sh
pip install -e .
```

### Running Locally

How to run the application on your local system.

### Running Tests

<!-- How to run tests on your local system. -->

Tests can be run across all compatible Python versions in isolated environments
using [`tox`](https://tox.wiki/en/latest/) by running

```sh
tox
```

To run tests manually in a Python environment with `pytest` installed run

```sh
pytest tests
```

again from the root of the repository.

### Building Documentation

The MkDocs HTML documentation can be built locally by running

```sh
tox -e docs
```

from the root of the repository. The built documentation will be written to
`site`.

Alternatively to build and preview the documentation locally, in a Python
environment with the optional `docs` dependencies installed, run

```sh
mkdocs serve
```

## Roadmap

- [x] Initial Research
- [ ] Minimum viable product <-- You are Here
- [ ] Alpha Release
- [ ] Feature-Complete Release


### References
- Tackley, P.J. (2008) 'Modelling compressible mantle convection with large viscosity contrasts in a three-dimensional spherical shell using the yin-yang grid', Physics of the Earth and Planetary Interiors, 171(1), pp. 7–18. Available at: https://doi.org/10.1016/j.pepi.2008.08.005.
- Connolly, J.A.D. (2005) 'Computation of phase equilibria by linear programming: A tool for geodynamic modeling and its application to subduction zone decarbonation', Earth and Planetary Science Letters, 236(1–2), pp. 524–541. Available at: https://doi.org/10.1016/j.epsl.2005.04.033.
- Morison, A.; Ulvrova, M.; Labrosse S. ; B4rsh; theofatou; tfrass49 (2022) 'StagPython/StagPy': Zenodo. Available at: https://doi.org/10.5281/ZENODO.6388133.
- Nissen-Meyer, T.; van Driel, M.; Stähler, S. C.; Hosseini, K.; Hempel, S.; Auer, L.; Colombi, A. and Fournier, A. (2014) 'AxiSEM: broadband 3-D seismic wavefields in axisymmetric media', Solid Earth, 5(1), pp. 425–445. Available at: https://doi.org/10.5194/se-5-425-2014.
