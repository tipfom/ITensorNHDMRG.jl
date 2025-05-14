# ITensorNHDMRG.jl

ITensorNHDMRG.jl is a library containing algorithms for non-hermitian density-matrix renormalization (DMRG) algorithms based on [ITensor.jl](https://github.com/ITensor/ITensors.jl) and [ITensorMPS.jl](https://github.com/ITensor/ITensors.jl).
The object is given an operator $A$ to solve for the right- and left-eigenvalues $|x\rangle$ and $|y \rangle$ such that for an eigenvalue $\lambda$ it holds that $A |x\rangle = \lambda |x\rangle$ and $A^\dagger |y\rangle = \lambda^\ast |y\rangle$. 

## Installation instructions

Currently, this library requires a non-standard version of [KrylovKit.jl](https://github.com/Jutho/KrylovKit.jl/pull/124). 
This is an early work-in-progress development version and the implementation details are up to change.

## Features
Non-hermitian DMRG is currently supported for systems with and without quantum numbers.
Both one-sided Krylov iteration, i.e., the analog of solving $A |x \rangle = \lambda |x \rangle$ and $\langle y | A = \lambda \langle y|$ seperately, as well as two-sided Krylov iteration solving the combined problem $\langle y| A | x \rangle = \lambda \langle y|x\rangle$ [1].
As algorithm to compute the biorthogonal representation of the MPS, we include the `biorthoblock` [2] as well as the `lrdensity` algorithm [3].

## Examples

The current interface provided by `nhdmrg` is very similar to the `dmrg` function in ITensorMPS.jl and works with the libraries datatypes `MPO` and `MPS`.
An exemplary application of `nhdmrg` is provided in the following snippet for a non-Hermitian SSH chain.

```julia
using ITensors, ITensorMPS, ITensorNHDMRG
let
  # Create 200 Fermionic indices
  N = 100
  sites = siteinds("Fermion", 2N; conserve_qns=true)

  # Input operator terms which define
  # a Hamiltonian matrix, and convert
  # these terms to an MPO tensor network
  # (here we make the 1D Heisenberg model)
  t2, tL, tR = 1.0, 0.9, 1.1

  os = OpSum()
  for j in 1:N
    ja = 2(j - 1) + 1 # index on the a subsystem
    jb = 2(j - 1) + 2 # index on the b subsystem
    os += tL, "Cdag", ja, "C", jb
    os += tR, "Cdag", jb, "C", ja
  end
  # intercell hopping 
  for l in 1:(N - 1)
    lb = 2(l - 1) + 2 # index on the b subsystem
    lna = 2(l - 1) + 3 # neighbor index on the a subsystem
    os += t2, "Cdag", lb, "C", lna
    os += t2, "Cdag", lna, "C", lb
  end
  H = MPO(os, sites)

  # Create an initial random matrix product state
  psi0 = random_mps(sites)

  # Plan to do 5 passes or 'sweeps' of DMRG,
  # setting maximum MPS internal dimensions
  # for each sweep and maximum truncation cutoff
  # used when adapting internal dimensions:
  nsweeps = 5
  maxdim = [10, 20, 100, 100, 200]
  cutoff = 1E-10

  # Run the DMRG algorithm, returning energy
  # (dominant eigenvalue) and optimized left- and right- MPS
  energy, psil, psir = dmrg(H, psi0, psi0; nsweeps, maxdim, cutoff)
  println("Final energy = $energy")

  nothing
end

# Output:

# [ Info: running eigenvalue alg twosided with biortho alg biorthoblock
# After sweep 1 energy=-125.79651182109944 + 6.38880869634365e-16im  maxlinkdim=4 maxerr=7.55E-17 time=49.470
# After sweep 2 energy=-126.5970440167191 - 3.8972929632962534e-14im  maxlinkdim=16 maxerr=9.91E-11 time=1.693
# After sweep 3 energy=-126.59394665389584 - 9.206390877479873e-15im  maxlinkdim=51 maxerr=4.73E-10 time=2.589
# After sweep 4 energy=-126.57878497462858 - 2.501508273447256e-14im  maxlinkdim=62 maxerr=7.06E-10 time=3.319
# After sweep 5 energy=-126.56273673482853 + 3.8017127307406415e-14im  maxlinkdim=77 maxerr=6.58E-10 time=3.669
# After sweep 6 energy=-126.54329764990256 - 1.068965867907386e-14im  maxlinkdim=85 maxerr=8.09E-10 time=3.496
# After sweep 7 energy=-126.53800759038874 - 1.1080314163419911e-13im  maxlinkdim=92 maxerr=9.02E-10 time=4.078
# After sweep 8 energy=-126.57589538975382 - 2.0158840602741168e-12im  maxlinkdim=100 maxerr=9.75E-10 time=4.629
# After sweep 9 energy=-126.58284049475749 - 1.927081258540446e-12im  maxlinkdim=102 maxerr=1.36E-09 time=4.954
# After sweep 10 energy=-126.59544739321743 - 3.919447735556215e-12im  maxlinkdim=134 maxerr=1.70E-09 time=5.830
# After sweep 11 energy=-126.60623266874386 + 9.717784098601642e-13im  maxlinkdim=136 maxerr=1.41E-09 time=6.505
# After sweep 12 energy=-126.61462521528067 - 6.019094520257317e-13im  maxlinkdim=147 maxerr=1.71E-09 time=7.155
# After sweep 13 energy=-126.61960248328036 - 1.2666311319648402e-13im  maxlinkdim=126 maxerr=1.95E-09 time=6.979
# After sweep 14 energy=-126.62180262371618 - 5.374185300278436e-14im  maxlinkdim=115 maxerr=2.04E-09 time=7.361
# After sweep 15 energy=-126.62302688426553 + 6.139309720728697e-14im  maxlinkdim=112 maxerr=1.72E-09 time=7.556
# Final energy = -126.62302688426553 + 6.139309720728697e-14im
```

The eigen routine may be chosen by supplying either `"onesided"` or `"twosided"` as the keyword `alg` for `nhdmrg`; the biorthogonalization routine may be chosen by supplying either `"biorthoblock"` or `"lrdensity"` as the `biorthoalg` keyword.
The system in Ref. [2] provided in the example folder.


## References

For the two-sided Krylov solver we implemented 

[1] [Krylov-Schur-Type Restarts for the Two-Sided Arnoldi Method, Ian N. Zwaan and Michiel E. Hochstenbach](https://doi.org/10.1137/16M1078987)

in [KrylovKit.jl](https://github.com/Jutho/KrylovKit.jl).

The `biorthoblock` algorithm is introduced in 

[2] [Density-matrix renormalization group algorithm for non-Hermitian systems, Peigeng Zhong, Wei Pan, Haiqing Lin, Xiaoqun Wang, Shijie Hu](https://arxiv.org/abs/2401.15000).

The `lrdensity` algorithm is introduced in 

[3] [Universal properties of dissipative Tomonaga-Luttinger liquids: Case study of a non-Hermitian XXZ spin chain, Kazuki Yamamoto, Masaya Nakagawa, Masaki Tezuka, Masahito Ueda, and Norio Kawakami](https://doi.org/10.1103/PhysRevB.105.205125).