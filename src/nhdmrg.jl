using ITensors: @timeit_debug, @debug_check, scalartype, Printf
using ITensorMPS: check_hascommoninds, ProjMPO, position!
using KrylovKit: bieigsolve, BiArnoldi
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

function getbiorthogonalmps!(ψ)
    orthogonalize!(ψ, 1)

    ψo = deepcopy(ψ)
    rinds = uniqueinds(ψo[1], ψo[2])
    ltags = tags(commonind(ψo[1], ψo[2]))
    U, S, V, spec = svd(ψo[1], rinds; lefttags=ltags)
    Sinv = deepcopy(S)
    storage(Sinv) .= 1 ./ storage(Sinv)

    ψo[1] = U * S * V

    return ψo
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
    
    normfactor = sqrt(sum(D))
   
    for (M, phi) in [(Ml, phil), (Mr, phir)]    
        L, R = if ortho == "left"
            U, phi * dag(U) / normfactor
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

    spec
end

function nhdmrg(
    PH,
    psir0::MPS,
    psil0::MPS,
    sweeps::Sweeps;
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

                Θ = psir[b] * psir[b + 1]
                barΘ = psil[b] * psil[b + 1]

                fA = x -> productr(PH, x)
                fAH = x -> productl(PH, x)

                vals, V, W, info = bieigsolve(
                    (fA, fAH),
                    Θ,
                    barΘ,
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
                        Θ,
                        barΘ,
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

                ortho = ha == 1 ? "left" : "right"

                drho = nothing
                if noise(sweeps, sw) > 0
                    # Use noise term when determining new MPS basis.
                    # This is used to preserve the element type of the MPS.
                    elt = real(scalartype(psir))
                    drho = elt(noise(sweeps, sw)) * noiseterm(PH, V[1], W[1], ortho)
                end

                energy = vals[1]

                spec = nhreplacebond!(
                    psil,
                    psir,
                    b,
                    V[1],
                    W[1];
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
