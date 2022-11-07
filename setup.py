from setuptools import setup

setup(
name="chimera",
version="0.0.1",
description="""A simple python package that allows stagyy, stagpy, 
axisem, perplex talk to each other. It uses stagpy to load the 
Pressure (P), Temperature (P), Compositional (C) and Density fields 
output by the geodynamic modeling code StagYY. Then, these fields are 
interpolated on the finer AxiSEM grid. Then, it translates the 
P, T fields into thermoelastic properties for each composition 
(Adiabatic K, G moduli and Density). It does so based on PerpleX 
thermodynamic tables supplied for each composition 
(a modified version of the phempg is used to import the tab file). 
Then, the thermoelastic properties are averaged via a Voigt-Reuss 
scheme to create a geodynamically self-consistent seismic velocity model 
that can be later fed into AxiSEM as lateral heterogeneities.""",
url="https://github.com/MatteoDesiderio/chimera",
author="Matteo Desiderio",
author_email="ucfbmde@ucl.ac.uk",
license="",
packages=["interfaces"],
zip_safe=False
)