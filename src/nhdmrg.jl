using ITensors: @timeit_debug, @debug_check, scalartype, Printf
using ITensorMPS: check_hascommoninds, ProjMPO, position!
using KrylovKit: geneigsolve, eigsolve, bieigsolve, BiArnoldi
using Printf: @printf
using LinearAlgebra

include("projnhmpo.jl")

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
    # Decomposition kwargs
    mindim=nothing,
    maxdim=nothing,
    cutoff=nothing,
)
    ortho = NDTensors.replace_nothing(ortho, "left")

    @assert ortho == "left" || ortho == "right"

    leftindsl = if ortho == "left"
        commoninds(Ml[b], phil)
    else
        commoninds(Ml[b + 1], phil)
    end
    replaceinds!(phil, leftindsl, leftindsl')
    # @show inds(phil)

    leftindsr = if ortho == "left"
        commoninds(Mr[b], phir)
    else
        commoninds(Mr[b + 1], phir)
    end
    replaceinds!(phir, leftindsr, leftindsr'')
    # @show inds(phir)

    rho = phil * dag(phir)
    # @show inds(block)
    # @show norm(rho)
    # @show rho

    indsl = commoninds(rho, phil)
    indsr = commoninds(rho, phir)

    # rhomat = Array(rho, indsl..., indsr...)
    # @show reshape(rhomat, prod(dim(i) for i in indsl), prod(dim(i) for i in indsr))

    # @show indsl
    D, U, spec = eigen(rho, indsl, indsr; mindim, maxdim, cutoff, ishermitian=true)
    # @show sum(D), size(D)
    # @show D
    # @show U
    # @show inds(U)
    U = noprime!(U)
    U = replacetags!(U, tags(commonind(U, D)), tags(commonind(Ml[b], Ml[b + 1])))
    phil = noprime(phil)
    phir = noprime(phir)
    # @show inds(U)

    normfactor = sqrt(sum(D))

    Ll, Rl = if ortho == "left"
        U, phil * dag(U) / normfactor
    elseif ortho == "right"
        phil * dag(U) / normfactor, U
    end
    # @show inds(Ml[b])
    Ml[b] = Ll
    # @show inds(Ml[b])
    # @show inds(Ml[b+1])
    Ml[b + 1] = Rl
    # @show inds(Ml[b+1])

    Lr, Rr = if ortho == "left"
        U, phir * dag(U) / normfactor
    elseif ortho == "right"
        phir * dag(U) / normfactor, U
    end
    # @show inds(Mr[b])
    Mr[b] = Lr
    # @show inds(Mr[b])
    # @show inds(Mr[b+1])
    Mr[b + 1] = Rr
    # @show inds(Mr[b+1])

    for M in [Ml, Mr]
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
        else
            error(
                "In replacebond!, got ortho = $ortho, only currently supports `left` and `right`.",
            )
        end
    end

    return spec
end

function nhdmrg(
    PH,
    psir0::MPS,
    psil0::MPS,
    sweeps::Sweeps;
    which_decomp=nothing,
    svd_alg=nothing,
    observer=NoObserver(),
    outputlevel=1,
    # eigsolve kwargs
    eigsolve_tol=1e-14,
    eigsolve_krylovdim=5,
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
                # @info "LEFT START $b, bonddim: $(maxlinkdim(psil)), "
                # @show [id(i) for i in linkinds(psir)]
                # @show [id(i) for i in linkinds(psil)]
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
                    :SR,
                    BiArnoldi(;
                        tol=eigsolve_tol,
                        krylovdim=eigsolve_krylovdim,
                        maxiter=eigsolve_maxiter,
                        verbosity=eigsolve_verbosity,
                    ),
                )

                while length(vals) < 1
                    # did not converge, retrying 
                    Θp = Θ + 1e-3 * random_itensor(inds(Θ))
                    barΘp = barΘ + 1e-3 * random_itensor(inds(barΘ))

                    vals, V, W, info = bieigsolve(
                        (fA, fAH),
                        Θp,
                        barΘp,
                        1,
                        :SR,
                        BiArnoldi(;
                            tol=eigsolve_tol,
                            krylovdim=eigsolve_krylovdim,
                            maxiter=eigsolve_maxiter,
                            verbosity=eigsolve_verbosity,
                        ),
                    )
                end

                ortho = ha == 1 ? "left" : "right"

                energy = vals[1]

                nhreplacebond!(
                    psil,
                    psir,
                    b,
                    V[1],
                    W[1];
                    ortho,
                    maxdim=maxdim(sweeps, sw),
                    mindim=1,
                    cutoff=cutoff(sweeps, sw),
                )

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
                    # @printf(
                    #     "  R Trunc. err=%.2E, bond dimension %d\n", specr.truncerr, dim(linkind(psir, b))
                    # )
                    # @printf(
                    #     "  L Trunc. err=%.2E, bond dimension %d\n", specl.truncerr, dim(linkind(psil, b))
                    # )
                    flush(stdout)
                end

                sweep_is_done = (b == 1 && ha == 2)
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
        # isdone = checkdone!(observer; energy, psir, sweep=sw, outputlevel)
        # isdone && break
    end
    return (energy, psil, psir)
end
