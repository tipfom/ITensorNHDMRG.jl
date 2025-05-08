# ITensorNHDMRG.jl

ITensorNHDMRG.jl is a library containing algorithms for non-hermitian density-matrix renormalization (DMRG) algorithms based on [ITensor.jl](https://github.com/ITensor/ITensors.jl) and [ITensorMPS.jl](https://github.com/ITensor/ITensors.jl).
The object is given an operator $A$ to solve for the right- and left-eigenvalues $|x\rangle$ and $|y \rangle$ such that for an eigenvalue $\lambda$ it holds that $A |x\rangle = \lambda |x\rangle$ and $A^\dag |y\rangle = \lambda^\ast |y\rangle$. 

## Installation instructions

Currently, this library requires a non-standard version of [KrylovKit.jl](https://github.com/Jutho/KrylovKit.jl/pull/124). 
This is an early work-in-progress development version and the implementation details are up to change.

## Features
Non-hermitian DMRG is currently supported for systems with and without quantum numbers.
Both one-sided Krylov iteration, i.e., the analog of solving $A |x \rangle = \lambda |x \rangle$ and $A^\dag |y\rangle = \lambda^\ast |y\rangle$ seperately, as well as two-sided Krylov iteration solving the combined problem $\langle y| A | x \rangle = \lambda \langle y|x\rangle$ [1].
As algorithm to compute the biorthogonal representation of the MPS, we include the experimental approach `pseudoeigen` as well as the `biorthoblock` algorithm [2].

## Examples

The current interface provided by `nhdmrg` is very similar to the `dmrg` function in ITensorMPS.jl.
See the example folder.
More elaborate examples will follow once the API is finished.

## References

For the two-sided Krylov solver we implemented 

[1] [Krylov-Schur-Type Restarts for the Two-Sided Arnoldi Method, Ian N. Zwaan and Michiel E. Hochstenbach](https://doi.org/10.1137/16M1078987)

in [KrylovKit.jl](https://github.com/Jutho/KrylovKit.jl).

The `biorthoblock` algorithm is introduced in 

[2] [Density-matrix renormalization group algorithm for non-Hermitian systems, Peigeng Zhong, Wei Pan, Haiqing Lin, Xiaoqun Wang, Shijie Hu](https://arxiv.org/abs/2401.15000).
