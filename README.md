# Chimera

A simple python package that allows stagyy, stagpy, axisem, perplex talk to 
each other. 

It uses stagpy to load the Pressure (P), Temperature (P), 
Compositional (C) and Density fields ouptut by the geodynamic modeling code 
StagYY. Then, these fields are interpolated on the finer AxiSEM grid.
Then, it translates the P, T fields into thermoelastic properties for 
each composition (Adiabatic K, G moduli and Density). It does so based on 
PerpleX thermodynamic tables supplied for each composition (a modified version
of the phempg is used to import the tab file). 
Then, the thermoelastic properties are averaged via a Voigt-Reuss scheme to 
create a geodynamically self-consistent seismic velocity model that can be 
later fed into AxiSEM as lateral heterogeneities. 

## Installation


```bash
echo HELLO WORLD
```

## Usage

```python

import xyz

print("hello")
```

## TO-DO
-Save Perplex tab file directly into project folder

-Make Perplex tab reading faster maybe via parallelization

-Make coords and fields loading from stagyy faster

-Make K, G, rho extraction from tab files (based on geodynamic model P, T) 
faster using kdtrees

-Give option to compute seismic velocities using densities obtained either via 
tab file reading or stagyy
(curently only perplex available)

-Convert coordinates of velocity models to radial coordinates

-Compute seismic velocity anomaly

-Export to .sph file readable by axisem

-Complete readme file

-Make a package

## References
- stagyy
- stagpy
- numbakdtree
- axisem
- perplex
- phempg

## Contributing

## License


