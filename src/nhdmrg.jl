using ITensors: @timeit_debug, @debug_check, scalartype, Printf, Algorithm, @Algorithm_str
using ITensorMPS: check_hascommoninds, ProjMPO, position!
using KrylovKit: bieigsolve, eigsolve, BiArnoldi
using Printf: @printf
using LinearAlgebra

function nhdmrg(H::MPO, psir0::MPS, psil0::MPS, sweeps::Sweeps; kwargs...)
    check_hascommoninds(siteinds, H, psir0)
    check_hascommoninds(siteinds, H, psil0')
    check_hascommoninds(siteinds, psir0, psil0)
    # Permute the indices to have a better memory layout
    # and minimize permutations
    H = permute(H, (linkind, siteinds, linkind))
    PH = ProjNHMPO(H)
    return nhdmrg(PH, psir0, psil0, sweeps; kwargs...)
end

function nhdmrg(
    H::MPO,
    Msl::Vector{MPS},
    Msr::Vector{MPS},
    psir0::MPS,
    psil0::MPS,
    sweeps::Sweeps;
    weight=true,
    kwargs...,
)
    check_hascommoninds(siteinds, H, psir0)
    check_hascommoninds(siteinds, H, psil0')
    check_hascommoninds(siteinds, psir0, psil0)
    # Permute the indices to have a better memory layout
    # and minimize permutations
    H = permute(H, (linkind, siteinds, linkind))
    PH = ProjNHMPO_MPS(H, Msl, Msr; weight)
    return nhdmrg(PH, psir0, psil0, sweeps; kwargs...)
end

function decomposerho(
    ::Algorithm"pseudoeigen",
    phil,
    phir,
    drho,
    lindsl,
    lindsr,
    targettags;
    ishermitian=false,
    kwargs...,
)
    # compute reduced density matrix and apply perturbation
    replaceinds!(phil, lindsl, lindsl')
    rho = phil * dag(phir)
    if !isnothing(drho)
        rho += drho
    end

    D, U, spec = eigen(rho, lindsl', dag(lindsr); ishermitian, kwargs...)

    if ishermitian
        U = noprime!(U)
        return U, U, spec
    else
        Ubar = pinv(U, lindsr)
        U = noprime!(U)
        Ubar = noprime!(Ubar)
        return U, Ubar, spec
    end
end

function decomposerho(
    ::Algorithm"biorthoblock", phil, phir, drho, lindsl, lindsr, targettags; kwargs...
)
    # compute reduced density matrix and apply perturbation
    replaceinds!(phil, lindsl, lindsl')
    rho = phil * dag(phir)
    if !isnothing(drho)
        rho += drho
    end

    B, Y, Ybar, spec = transform(rho, lindsl', dag(lindsr); kwargs...)
    noprime!(Y)
    noprime!(Ybar)
    Y = replacetags!(Y, tags(commonind(Y, B)), targettags)
    Ybar = replacetags!(Ybar, tags(commonind(Ybar, B)), targettags)
    return Y, dag(Ybar), spec
end

function decomposerho(
    ::Algorithm"lrdensity", phil, phir, drho, lindsl, lindsr, targettags; kwargs...
)
    # Phys. Rev. B 105, 205125 
    # https://doi.org/10.1103/PhysRevB.105.205125
    # compute reduced density matrix and apply perturbation
    phir2 = replaceinds(phir, lindsr, lindsr')
    phil2 = replaceinds(phil, lindsl, lindsl')

    rho = (phil2 * dag(phil) + phir2 * dag(phir)) / 2
    if !isnothing(drho)
        rho += drho
    end

    D, U, spec = eigen(rho, lindsl', dag(lindsl); ishermitian=true, kwargs...)
    U = noprime!(U)
    return U, U, spec
end

function nhreplacebond!(
    Ml::MPS,
    Mr::MPS,
    b::Int,
    phil::ITensor,
    phir::ITensor,
    alg;
    ortho=nothing,
    eigen_perturbation=nothing,
    # Decomposition kwargs
    mindim=nothing,
    maxdim=nothing,
    cutoff=nothing,
    kwargs...,
)
    ortho = NDTensors.replace_nothing(ortho, "left")

    if ortho != "left" && ortho != "right"
        error(
            "In replacebond!, got ortho = $ortho, only currently supports `left` and `right`.",
        )
    end

    leftindsl = if ortho == "left"
        commoninds(Ml[b], phil)
    else
        commoninds(Ml[b + 1], phil)
    end
    leftindsr = if ortho == "left"
        commoninds(Mr[b], phir)
    else
        commoninds(Mr[b + 1], phir)
    end

    U, Ubar, spec = decomposerho(
        Algorithm(alg),
        phil,
        phir,
        eigen_perturbation,
        leftindsl,
        leftindsr,
        tags(commonind(Ml[b], Ml[b + 1]));
        mindim,
        maxdim,
        cutoff,
        kwargs...,
    )

    # replaceinds!(phil, leftindsl', leftindsl)
    noprime!(phil)

    sD = sum(eigs(spec))
    normfactor = sqrt(abs(sD))

    for (M, phi, U, U2) in [(Ml, phil, U, Ubar), (Mr, phir, Ubar, U)]
        L, R = if ortho == "left"
            U, phi * dag(U2) / normfactor
        elseif ortho == "right"
            phi * dag(U2) / normfactor, U
        end
        M[b] = L
        M[b + 1] = R

        if ortho == "left"
            ITensorMPS.leftlim(M) == b - 1 &&
                ITensorMPS.setleftlim!(M, ITensorMPS.leftlim(M) + 1)
            ITensorMPS.rightlim(M) == b + 1 &&
                ITensorMPS.setrightlim!(M, ITensorMPS.rightlim(M) + 1)
        elseif ortho == "right"
            ITensorMPS.leftlim(M) == b &&
                ITensorMPS.setleftlim!(M, ITensorMPS.leftlim(M) - 1)
            ITensorMPS.rightlim(M) == b + 2 &&
                ITensorMPS.setrightlim!(M, ITensorMPS.rightlim(M) - 1)
        end
    end

    return spec
end

function eigproblemsolver!(
    ::Algorithm"twosided",
    PH,
    Θl,
    Θr;
    eigsolve_tol,
    eigsolve_krylovdim,
    eigsolve_maxiter,
    eigsolve_verbosity,
    eigsolve_which_eigenvalue,
)
    fA = x -> productr(PH, x)
    fAH = x -> productl(PH, x)

    vals, V, W, info = bieigsolve(
        (fA, fAH),
        Θr,
        Θl,
        1,
        eigsolve_which_eigenvalue,
        BiArnoldi(;
            tol=eigsolve_tol,
            krylovdim=eigsolve_krylovdim,
            maxiter=eigsolve_maxiter,
            verbosity=eigsolve_verbosity,
        ),
    )

    while length(vals) < 1
        # did not converge, retrying 
        eigsolve_maxiter = div(5eigsolve_maxiter, 3)
        eigsolve_krylovdim = div(5eigsolve_krylovdim, 3)
        @warn "Eigensolver did not converge, consider increasing the krylovdimension or iterations; now using eigsolve_krylovdim=$eigsolve_krylovdim and eigsolve_maxiter=$eigsolve_maxiter."

        vals, V, W, info = bieigsolve(
            (fA, fAH),
            Θr,
            Θl,
            1,
            eigsolve_which_eigenvalue,
            BiArnoldi(;
                tol=eigsolve_tol,
                krylovdim=eigsolve_krylovdim,
                maxiter=eigsolve_maxiter,
                verbosity=eigsolve_verbosity,
            ),
        )
    end

    return first(vals), first(V), first(W)
end

function eigproblemsolver!(
    ::Algorithm"onesided",
    PH,
    Θl,
    Θr;
    eigsolve_tol,
    eigsolve_krylovdim,
    eigsolve_maxiter,
    eigsolve_verbosity,
    eigsolve_which_eigenvalue,
)
    fA = x -> productr(PH, x)
    fAH = x -> productl(PH, x)

    valsH, vecsH = eigsolve(
        fAH,
        Θl,
        1,
        eigsolve_which_eigenvalue;
        ishermitian=false,
        tol=eigsolve_tol,
        krylovdim=eigsolve_krylovdim,
        maxiter=eigsolve_maxiter,
        verbosity=eigsolve_verbosity,
    )

    vals, vecs = eigsolve(
        fA,
        Θr,
        1,
        eigsolve_which_eigenvalue;
        ishermitian=false,
        tol=eigsolve_tol,
        krylovdim=eigsolve_krylovdim,
        maxiter=eigsolve_maxiter,
        verbosity=eigsolve_verbosity,
    )

    return first(vals), noprime(first(vecsH)), noprime(first(vecs))
end

function biorthogonalize!(psir, psil, alg; mindim=nothing, maxdim=10, cutoff=1e-12)
    @assert siteinds(psir) == siteinds(psil) "both MPS need to share the same basis"

    sites = siteinds(psir)

    noprime!(psir)
    noprime!(psil)
    prime!(psir, "Link")

    M = ITensor[]
    for i in firstindex(sites):lastindex(sites)
        Mi = ITensor(1)

        if length(M) > 0
            Mi = M[end] * delta(prime(dag(sites[i - 1])), sites[i - 1])
        end

        Mi *= prime(psir[i], sites[i])
        Mi *= dag(psil[i])
        push!(M, Mi)
    end

    noprime!(psir)

    for i in (lastindex(sites) - 1):-1:1
        phil = psil[i] * psil[i + 1]
        phir = setprime(psir[i], 1) * setprime(psir[i + 1], 1)

        if i > 1
            phir *= M[i - 1] * delta(prime(dag(sites[i - 1])), sites[i - 1])
        end

        phir = noprime(phir)

        nhreplacebond!(
            psil, psir, i, phil, phir, alg; ortho="right", mindim, maxdim, cutoff
        )
    end

    X = dag(psil[lastindex(sites)]) * prime(psir[lastindex(sites)], "Link")
    for i in (lastindex(sites) - 1):-1:1
        display(matrix(X))
        X *= dag(psil[i]) * prime(psir[i], "Link")
    end

    noprime!(psil)
    noprime!(psir)

    return psir, psil
end

"""
    nhdmrg(H::MPO, psir0::MPS, psil0::MPS, sweeps::Sweeps; kwargs...)

Use a non-Hermitian variant of the density matrix renormalization group 
(NH-DMRG) algorithm to optimize the matrix product states (MPS) 
such that they are the left- and right-eigenvector of lowest eigenvalue 
of a non-Hermitian matrix `H`, represented as a matrix product operator (MPO).

    nhdmrg(H::MPO, Msl::Vector{MPS}, Msr::Vector{MPS}, psir0::MPS, psil0::MPS, sweeps::Sweeps; weight=1.0, kwargs...)
    
Use a non-Hermitian variant of the density matrix renormalization group 
(NH-DMRG) algorithm to optimize the matrix product states (MPS) 
such that they are the left- and right-eigenvector of lowest eigenvalue 
of a non-Hermitian matrix `H`, subject to the constraint that the MPS
are orthogonal to each of the left- and right-MPS provided in the Vector 
`Msl` and `Msr`. The orthogonality constraint is approximately enforced by
adding to `H` terms of the form `w|Mr1><Ml1| + w|Mr2><Ml2| + ...` where 
`Mls=[Ml1, Ml2, ...]`, `Mrs=[Mr1, Mr2, ...]` and `w` is the "weight" parameter, 
which can be adjusted through the optional `weight` keyword argument.

!!! note
    `nhdmrg` will report the energy of the operator
    `H + w|Mr1><Ml1| + w|Mr2><Ml2| + ...`, not the operator `H`.
    If you want the expectation value of the MPS eigenstate
    with respect to just `H`, you can compute it yourself with
    an observer or after DMRG is run with `inner(psil', H, psir)`.

The MPS `psil0` and `psir0` is used to initialize the MPS to be optimized.

The number of sweeps and accuracy parameters can be passed through a 
`Sweeps` object.

The NH-DMRG algorithm allows choosing both the eigenvalue solver by passing 
the `alg` keyword and the truncation algorithm by passing the `biorthoalg`
keyword. The eigenvalue algorithm `alg` may be `onesided`, such that the for each 
block the eigenvalue problem ``A |x> = λ |x>`` and ``A† |y> = λ* |y>`` are 
solved, or `twosided`, such that ``<y| A |x> = λ <y|x>`` are solved simultaneously
using a two-sided Krylov approach [1]. The truncation algorithm `biorthoalg` currently 
supports the biorthogonal block method `biorthoblock` [2] and the left-right density 
matrix method `lrdensity` [3].

[1] https://doi.org/10.1137/16M1078987
[2] https://doi.org/10.48550/arXiv.2401.15000
[3] https://doi.org/10.1103/PhysRevB.105.205125

Returns:

  - `energy::Number` - eigenvalue of the optimized MPS
  - `psil::MPS` - optimized left MPS
  - `psir::MPS` - optimized right MPS

Optional keyword arguments:

  - `alg::String = "twosided"` - local eigenvalue algorithm, either `"onesided"` or `"twosided"`
  - `biorthoalg::String = "biorthoblock"` - orthogonalization algorithm, eiter `"biorthoblock"` 
     or `"lrdensity"`
  - `isbiortho::Bool=false` - if `true` the input MPS are not biorthogonalized before starting 
     the sweeps 
  - `eigsolve_krylovdim::Int = 3` - maximum dimension of Krylov space used to
     locally solve the eigenvalue problem. Try setting to a higher value if
     convergence is slow or the Hamiltonian is close to a critical point. [^krylovkit]
  - `eigsolve_tol::Number = 1e-14` - Krylov eigensolver tolerance. [^krylovkit]
  - `eigsolve_maxiter::Int = 1` - number of times the Krylov subspace can be
     rebuilt. [^krylovkit]
  - `eigsolve_verbosity::Int = 0` - verbosity level of the Krylov solver.
     Warning: enabling this will lead to a lot of outputs to the terminal. [^krylovkit]
  - `outputlevel::Int = 1` - larger outputlevel values make DMRG print more
     information and 0 means no output.
  - `observer` - object implementing the [Observer](@ref observer) interface
     which can perform measurements and stop DMRG early.
  - `biorthokwargs` - additional kwargs for the biorthogonalization routine, 
     see `decomposerho`.
"""
function nhdmrg(
    PH::Union{ProjNHMPO,ProjNHMPO_MPS,ProjNHMPS},
    psir0::MPS,
    psil0::MPS,
    sweeps::Sweeps;
    alg="twosided",
    biorthoalg="biorthoblock",
    isbiortho::Bool=false,
    observer=NoObserver(),
    outputlevel=1,
    # eigsolve kwargs
    eigsolve_tol=1e-14,
    eigsolve_krylovdim=3,
    eigsolve_maxiter=1,
    eigsolve_verbosity=0,
    eigsolve_which_eigenvalue=:SR,
    biorthokwargs...,
)
    if length(psir0) == 1
        error(
            "`dmrg` currently does not support system sizes of 1. You can diagonalize the MPO tensor directly with tools like `LinearAlgebra.eigen`, `KrylovKit.eigsolve`, etc.",
        )
    end

    @info "running eigenvalue alg $alg with biortho alg $biorthoalg"

    @debug_check begin
        # Debug level checks
        # Enable with ITensors.enable_debug_checks()
        checkflux(psir0)
        checkflux(PH)
    end

    psir = deepcopy(psir0)
    psil = deepcopy(psil0)
    N = length(psir0)

    if !isbiortho
        psir, psil = biorthogonalize!(
            psir,
            psil,
            biorthoalg;
            maxdim=maxdim(sweeps, 1),
            mindim=mindim(sweeps, 1),
            cutoff=cutoff(sweeps, 1),
            biorthokwargs...,
        )
    end

    PH = ITensorMPS.position!(PH, psil, psir, 1)
    energy = 0.0

    for sw in 1:nsweep(sweeps)
        sw_time = @elapsed begin
            maxtruncerr = 0.0

            for (b, ha) in sweepnext(N)
                @debug_check begin
                    checkflux(psir)
                    checkflux(PH)
                end

                PH = ITensorMPS.position!(PH, psil, psir, b)

                @debug_check begin
                    checkflux(psir)
                    checkflux(PH)
                end

                Θr = psir[b] * psir[b + 1]
                Θl = psil[b] * psil[b + 1]

                energy, v, w = eigproblemsolver!(
                    Algorithm(alg),
                    PH,
                    Θl,
                    Θr;
                    eigsolve_tol,
                    eigsolve_krylovdim,
                    eigsolve_maxiter,
                    eigsolve_verbosity,
                    eigsolve_which_eigenvalue,
                )

                ortho = ha == 1 ? "left" : "right"

                drho = nothing
                if noise(sweeps, sw) > 0
                    # Use noise term when determining new MPS basis.
                    # This is used to preserve the element type of the MPS.
                    elt = real(scalartype(psir))
                    drho = elt(noise(sweeps, sw)) * noiseterm(PH, v, w, ortho)
                end

                spec = nhreplacebond!(
                    psil,
                    psir,
                    b,
                    v,
                    w,
                    biorthoalg;
                    ortho,
                    eigen_perturbation=drho,
                    maxdim=maxdim(sweeps, sw),
                    mindim=mindim(sweeps, sw),
                    cutoff=cutoff(sweeps, sw),
                    biorthokwargs...,
                )

                maxtruncerr = max(maxtruncerr, spec.truncerr)

                @debug_check begin
                    checkflux(psir)
                    checkflux(PH)
                end

                if outputlevel >= 2
                    @printf(
                        "Sweep %d, half %d, bond (%d,%d) energy=%s\n",
                        sw,
                        ha,
                        b,
                        b + 1,
                        energy
                    )
                    @printf(
                        "  Truncated using cutoff=%.1E maxdim=%d mindim=%d\n",
                        cutoff(sweeps, sw),
                        maxdim(sweeps, sw),
                        mindim(sweeps, sw)
                    )
                    flush(stdout)
                end
            end
        end
        if outputlevel >= 1
            @printf(
                "After sweep %d energy=%s  maxlinkdim=%d maxerr=%.2E time=%.3f\n",
                sw,
                energy,
                maxlinkdim(psil),
                maxtruncerr,
                sw_time
            )
            flush(stdout)
        end
        isdone = checkdone!(observer; energy, psir, sweep=sw, outputlevel)
        isdone && break
    end
    return (energy, psil, psir)
end
