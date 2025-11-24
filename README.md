# ITensorNHDMRG.jl

ITensorNHDMRG.jl is a library containing algorithms for non-hermitian density-matrix renormalization (DMRG) algorithms based on [ITensor.jl](https://github.com/ITensor/ITensors.jl) and [ITensorMPS.jl](https://github.com/ITensor/ITensors.jl).
The object is given an operator $A$ to solve for the right- and left-eigenvalues $|x\rangle$ and $|y \rangle$ such that for an eigenvalue $\lambda$ it holds that $A |x\rangle = \lambda |x\rangle$ and $A^\dagger |y\rangle = \lambda^\ast |y\rangle$. 

## Installation instructions

The ITensorNHDMRG package can be installed with the Julia package manager.
From the Julia REPL, type `]` to enter the Pkg REPL mode and run:

```
~ julia
```

```julia
julia> ]

pkg> add https://github.com/tipfom/ITensorNHDMRG.jl
```
## Features
Non-hermitian DMRG is currently supported for systems with and without quantum numbers.
Both one-sided Krylov iteration, i.e., the analog of solving $A |x \rangle = \lambda |x \rangle$ and $\langle y | A = \lambda \langle y|$ seperately, as well as two-sided Krylov iteration solving the combined problem $\langle y| A | x \rangle = \lambda \langle y|x\rangle$ [1].
As algorithm to compute the biorthogonal representation of the MPS, we include the `biorthoblock` [2] as well as the `fidelity` algorithm [3].

## Examples

The current interface provided by `nhdmrg` is very similar to the `dmrg` function in ITensorMPS.jl and works with the libraries datatypes `MPO` and `MPS`.
An exemplary application of `nhdmrg` is provided in the following snippet for a non-Hermitian SSH chain.

```julia
using ITensors, ITensorMPS, ITensorNHDMRG, Random
let
    # Create 200 Fermionic indices
    N = 50
    sites = siteinds("Fermion", 2N; conserve_qns=true)

    # Input operator terms which define
    # a Hamiltonian matrix, and convert
    # these terms to an MPO tensor network
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

    # make results reproducible 
    rng = Xoshiro(1234)

    # Create an initial random matrix product state
    # with half filling
    psi0 = random_mps(rng, sites, [ifelse(mod(i, 2) == 0, "Occ", "Emp") for i in 1:2N])

    # Plan to do 15 passes or 'sweeps' of DMRG,
    # setting maximum MPS internal dimensions
    # for each sweep and maximum truncation cutoff
    # used when adapting internal dimensions:
    nsweeps = 5
    maxdim = [20, 50, 100]
    cutoff = 1E-10
    noise = [1e-5, 1e-7, 0.0]

    # Run the DMRG algorithm, returning energy
    # (dominant eigenvalue) and optimized left- and right- MPS
    energy, psil, psir = nhdmrg(H, psi0, psi0; nsweeps, maxdim, cutoff, noise, alg="onesided")
    println("Final energy = $energy")

    return nothing
end

# Output:

# After sweep 1 energy=-62.968112390229436 + 0.0im  maxlinkdim=20 maxerr=8.22E-05 time=0.522
# After sweep 2 energy=-63.112380103700644 + 0.0im  maxlinkdim=50 maxerr=5.40E-07 time=1.464
# After sweep 3 energy=-63.14434243476322 + 0.0im  maxlinkdim=100 maxerr=5.96E-08 time=3.641
# After sweep 4 energy=-63.134169852574004 + 0.0im  maxlinkdim=100 maxerr=9.50E-08 time=3.531
# After sweep 5 energy=-63.13514581299604 + 0.0im  maxlinkdim=100 maxerr=4.64E-08 time=3.793
# Final energy = -63.13514581299604 + 0.0im
```

The eigen routine may be chosen by supplying either `"onesided"` or `"twosided"` as the keyword `alg` for `nhdmrg`; the biorthogonalization routine may be chosen by supplying either `"biorthoblock"` or `"fidelity"` as the `biorthoalg` keyword.
The system in Ref. [2] provided in the example folder.


## References

For the two-sided Krylov solver we implemented 

[1] [Krylov-Schur-Type Restarts for the Two-Sided Arnoldi Method, Ian N. Zwaan and Michiel E. Hochstenbach](https://doi.org/10.1137/16M1078987)

in [KrylovKit.jl](https://github.com/Jutho/KrylovKit.jl).

The `biorthoblock` algorithm is introduced in 

[2] [Density-matrix renormalization group algorithm for non-Hermitian systems, Peigeng Zhong, Wei Pan, Haiqing Lin, Xiaoqun Wang, Shijie Hu](https://doi.org/10.1103/5vnl-w9p4).

The `fidelity` algorithm is introduced in 

[3] [Universal properties of dissipative Tomonaga-Luttinger liquids: Case study of a non-Hermitian XXZ spin chain, Kazuki Yamamoto, Masaya Nakagawa, Masaki Tezuka, Masahito Ueda, and Norio Kawakami](https://doi.org/10.1103/PhysRevB.105.205125).