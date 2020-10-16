# Adaptive Shape Approximation

* This code implements all the main concepts from the report "Adaptive Approximation of Shapes" for the simple domain integral shape functional: the first and second shape derivatives, quadratic approximation of the functional, low rank approximation of the quadratic functional and the adaptive algorithm based on subspace iteration. The code is based on the 2DParametricbem libray (url: https://gitlab.ethz.ch/ppanchal/2dparametricbem) which is used as a submodule.

* Lead developer: Piyush Panchal

## Building the code
* Create a "build" directory in the project directory. Use "cmake .." from the build directory.

## Compiling for different shape functionals
* It is possible to choose a different function for the domain integral shape functional
* The function is chosen by defining an environment variable "mm"
* Example: "make circle_pert_vels -e mm=1" or  
* The number "1" can be replaced by other numbers. See functionals.hpp for the options

## Perturbation fields
* From the build directory call "make circle_pert_vels -e mm=1" for a circular domain
* For a kite domain call "make kite_pert_vels -e mm=1"
* The output contains two files which include the basis vectors for the full space and the eigenvalues.

