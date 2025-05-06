using ITensors: @timeit_debug, @debug_check, scalartype, Printf, Algorithm, @Algorithm_str
using ITensorMPS: check_hascommoninds, ProjMPO, position!
using KrylovKit: bieigsolve, eigsolve, BiArnoldi
using Printf: @printf
using LinearAlgebra

function nhdmrg(H::MPO, psi0::MPS, sweeps::Sweeps; kwargs...)
    check_hascommoninds(siteinds, H, psi0)
    check_hascommoninds(siteinds, H, psi0')
    # Permute the indices to have a better memory layout
    # and minimize permutations
    H = permute(H, (linkind, siteinds, linkind))
    PH = ProjNHMPO(H)
    return nhdmrg(PH, psi0, getbiorthogonalmps!(psi0), sweeps; kwargs...)
end

function nhdmrg(
    H::MPO,
    Msl::Vector{MPS},
    Msr::Vector{MPS},
    psi0::MPS,
    sweeps::Sweeps;
    weight=true,
    kwargs...,
)
    check_hascommoninds(siteinds, H, psi0)
    check_hascommoninds(siteinds, H, psi0')
    # Permute the indices to have a better memory layout
    # and minimize permutations
    H = permute(H, (linkind, siteinds, linkind))
    PH = ProjNHMPO_MPS(H, Msl, Msr; weight)
    return nhdmrg(PH, psi0, getbiorthogonalmps!(psi0), sweeps; kwargs...)
end

function invertitensor(A::ITensor, finv, Linds...)
    Lis = commoninds(A, ITensors.indices(Linds...))
    Ris = uniqueinds(A, Lis)

    Cr = combiner(Ris...)
    Cl = combiner(Lis...)

    A = A * Cr * Cl

    Minv = nothing

    return if hasqns(A)
        Minv = deepcopy(A)
        for b in eachnzblock(A)
            Amat = matrix(A[b])
            Amatinv = finv(Amat)

            Minv[b] .= adjoint(Amatinv)
        end
        Minv * dag(Cr) * dag(Cl)
    else
        M = matrix(A, combinedind(Cr), combinedind(Cl))
        Minv = itensor(adjoint(finv(M)), combinedind(Cr), combinedind(Cl); tol=1e-8)
        Minv * dag(Cr) * dag(Cl)
    end
end

# function invertitensor(A::ITensor, finv, Linds...)
#     U, S, V = svd(A, Linds...; cutoff=1e-8)
#     Sinv = S
#     storage(Sinv) .= 1 ./ storage(Sinv)

#     Ainv = dag(V) * dag(Sinv) * dag(U)

#     return dag(Ainv)
# end

function LinearAlgebra.inv(A::ITensor, Linds...)
    return invertitensor(A, inv, Linds...)
end

function LinearAlgebra.pinv(A::ITensor, Linds...; kwargs...)
    return invertitensor(A, x -> pinv(x; kwargs...), Linds...)
end

function LinearAlgebra.inv(A::ITensor)
    Ris = filterinds(A; plev=0)
    Lis = Ris'
    return LinearAlgebra.inv(A, Lis)
end

function LinearAlgebra.pinv(A::ITensor; kwargs...)
    Ris = filterinds(A; plev=0)
    Lis = Ris'
    return LinearAlgebra.pinv(A, Lis; kwargs...)
end

function getbiorthogonalmps!(ψ)
    orthogonalize!(ψ, 1)

    ψo = deepcopy(ψ)
    ψo[1] = pinv(ψo[1], uniqueinds(ψo[1], ψo[2]))

    return ψo / inner(ψo, ψ)
end

function nhreplacebond!(
    Ml::MPS,
    Mr::MPS,
    b::Int,
    phil::ITensor,
    phir::ITensor;
    ortho=nothing,
    eigen_perturbation=nothing,
    # Decomposition kwargs
    mindim=nothing,
    maxdim=nothing,
    cutoff=nothing,
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
    replaceinds!(phil, leftindsl, leftindsl')

    # compute reduced density matrix and apply perturbation
    rho = phil * dag(phir)
    if !isnothing(eigen_perturbation)
        rho += eigen_perturbation
    end

    indsl = commoninds(rho, phil)
    indsr = commoninds(rho, phir)
    D, U, spec = eigen(rho, indsl, indsr; mindim, maxdim, cutoff, ishermitian=true)

    U = noprime!(U)
    U = replacetags!(U, tags(commonind(U, D)), tags(commonind(Ml[b], Ml[b + 1])))
    phil = noprime!(phil)

    sD = sum(D)

    normfactor = sqrt(abs(sD))

    for (M, phi) in [(Ml, phil), (Mr, phir)]
        L, R = if ortho == "left"
            U, phi * dag(U) / (sign(sD) * normfactor)
        elseif ortho == "right"
            phi * dag(U) / normfactor, U
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
    Θr,
    leftinds;
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
    ::Algorithm"pseudoonesided",
    PH,
    Θl,
    Θr,
    leftinds;
    eigsolve_tol,
    eigsolve_krylovdim,
    eigsolve_maxiter,
    eigsolve_verbosity,
    eigsolve_which_eigenvalue,
)
    fA = x -> productr(PH, x)
    fAH = x -> productl(PH, x)

    vals, vecs = eigsolve(
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

    v = noprime(first(vecs))
   
    w = if isnothing(leftinds)
        v
    else
        pinv(v, leftinds)
    end
    @assert length(inds(w)) == length(commoninds(inds(v), inds(w))) == length(inds(w)) "Index mismatch between $(inds(v)) and $(inds(w))"

    return first(vals), noprime(v), noprime(w)
end

function eigproblemsolver!(
    ::Algorithm"onesided",
    PH,
    Θl,
    Θr,
    leftinds;
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

# current options for alg are "twosided" "onesided" and "pseudoonesided"
function nhdmrg(
    PH,
    psir0::MPS,
    psil0::MPS,
    sweeps::Sweeps;
    alg="twosided",
    observer=NoObserver(),
    outputlevel=1,
    # eigsolve kwargs
    eigsolve_tol=1e-14,
    eigsolve_krylovdim=3,
    eigsolve_maxiter=3,
    eigsolve_verbosity=0,
    eigsolve_which_eigenvalue=:SR,
)
    if length(psir0) == 1
        error(
            "`dmrg` currently does not support system sizes of 1. You can diagonalize the MPO tensor directly with tools like `LinearAlgebra.eigen`, `KrylovKit.eigsolve`, etc.",
        )
    end

    @debug_check begin
        # Debug level checks
        # Enable with ITensors.enable_debug_checks()
        checkflux(psir0)
        checkflux(PH)
    end

    psir = deepcopy(psir0)
    psil = deepcopy(psil0)
    N = length(psir0)
    @assert isortho(psir) && orthocenter(psir) == 1

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
                    Θr,
                    linkind(psir, b + ifelse(ha==1, 1, -1));
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
                    w;
                    ortho,
                    eigen_perturbation=drho,
                    maxdim=maxdim(sweeps, sw),
                    mindim=1,
                    cutoff=cutoff(sweeps, sw),
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
                "After sweep %d energy=%s  maxlinkdim=(l: %d, r: %d) maxerr=%.2E time=%.3f\n",
                sw,
                energy,
                maxlinkdim(psil),
                maxlinkdim(psir),
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
