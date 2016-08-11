This repository contains an R package for multivariate, multiple regression extension of [ash](https://github.com/stephens999/ashr).

To install the `m2ashr` package,
```
devtools::install_github("gaow/m2ashr")
```
If you do not have `devtools` or have no internet access, you need to [obtain the source code](https://github.com/gaow/m2ashr/archive/master.zip), decompress the tarball and type `make` to install the package.

## Running paired factor analysis

The main function in the `m2ashr` is `m2ash`:
```
> ?m2ashr::m2ash
```
You can follow the `example` section of the documentation to run the program on an example data set.

## Troubleshoot

### Linear algebra support
If you get error message *Cannot find lapack / openblas* you need to install `LAPACK` and `OpenBLAS` libraries. On Debian Linux:
```
sudo apt-get install libopenblas-dev liblapack-dev
```
### OpenMP
`m2ashr` requires compilers with OpenMP support. On Debian Linux:
```
sudo apt-get install libgomp1
```
