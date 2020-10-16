# Adaptive Shape Approximation

* This code implements all the main concepts from the report "Adaptive Approximation of Shapes" for the simple domain integral shape functional: the first and second shape derivatives, quadratic approximation of the functional, low rank approximation of the quadratic functional and the adaptive algorithm based on subspace iteration. The code is based on the 2DParametricbem libray (url: https://gitlab.ethz.ch/ppanchal/2dparametricbem).

Requirements: Eigen3, cmake, GNU compiler

* Lead developer: Piyush Panchal

## Building the code
* Requires Eigen 3. If unavailable, use "sudo apt-get install -y libeigen3-dev".
* Get the submodule by "git submodule update --init --recursive".
* Create a "build" directory in the project directory. Use "cmake .." from the build directory.
* All the compiled executables lie in the folder build/examples

## Compiling for different shape functionals 
* It is possible to choose a different domain integral shape functional by providing an environment variable "mm"
* Example: "make circle_pert_vels -e mm=1" or  "make circle_random_samping -e mm=2" 
* Other numbers can be used instead of "1". See functionals.hpp for the options

## Eigenfunctions/Eigenvectors
* Concerned with obtaining the vectors/functions which span the low rank approximation subspace.
* From the build directory call "make circle_pert_vels -e mm=1" for a circular domain or "make kite_pert_vels -e mm=1" for a kite domain
* The output contains two files which includes all the basis functions/vectors and the eigenvalues.

## Approximation Errors
* Concerned with obtaining different approximation errors: Low rank approximation error(lrae), quadratic approximation error(qae) and full approximation error(fae).Errors are evaluated using a sampling approach. They are evaluated at all the basis vectors for the approximating subspace and random linear combinations of these vectors.
* From the build directory call "make circle_random_sampling -e mm=1" for a circular domain or "make kite_random_sampling -e mm=1" for a kite domain.
* The output from circle_random_sampling and kite_random_sampling includes three files which contain the fae, qae and lrae.

## Adaptive Algorithm
* Concerned with the performance of the adaptive algorithm
* From the build directory call "make circle_adaptive -e mm=1" for a circular domain
* For a kite domain call "make kite_adaptive -e mm=1"
* Outputs three files which contain the eigenvectors, eigenvalues and subspace iterations at each adaptive step.

