function nhfactorize(
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

function nhfactorize(
    ::Algorithm"biorthoblock", phil, phir, drho, lindsl, lindsr, targettags; kwargs...
)
    # compute reduced density matrix and apply perturbation
    replaceinds!(phil, lindsl, lindsl')
    rho = phil * dag(phir)
    if !isnothing(drho)
        !hassameinds(rho, drho) && error("Noise term has wrong indices")
        rho += drho
    end

    B, Y, Ybar, spec = biorthoblocktransform(rho, lindsl', dag(lindsr); kwargs...)
    noprime!(Y)
    noprime!(Ybar)
    noprime!(phir)
    Y = replacetags!(Y, tags(commonind(Y, B)), targettags)
    Ybar = replacetags!(Ybar, tags(commonind(Ybar, B)), targettags)
    return Y, dag(Ybar), spec
end

function nhfactorize(
    ::Algorithm"lrdensity", phil, phir, drho, lindsl, lindsr, targettags; kwargs...
)
    !hassameinds(phil, phir) && error("Left and right states need to share the same indices")

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
    alg,
    idright=nothing;
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
        commoninds(Ml[b+1], phil)
    end
    leftindsr = if ortho == "left"
        commoninds(Mr[b], phir)
    else
        commoninds(Mr[b+1], phir)
    end

    phirdeco = copy(phir)
    if !isnothing(idright)
        phirdeco = apply(idright, phirdeco)
        setprime!(phirdeco, 0, commonind(phirdeco, idright))
    end

    U, Ubar, spec = nhfactorize(
        Algorithm(alg),
        phil,
        phirdeco,
        eigen_perturbation,
        leftindsl,
        leftindsr,
        tags(commonind(Ml[b], Ml[b+1]));
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
        M[b+1] = R

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

function biorthogonalize!(psil, psir, alg; mindim=nothing, maxdim=10, cutoff=nothing, kwargs...)
    @assert siteinds(psir) == siteinds(psil) "both MPS need to share the same basis"

    @assert inner(psil, psir) >= sqrt(cutoff) "The initial vectors are almost orthogonal, overlap is $(inner(psil, psir))"

    sites = siteinds(psir)

    noprime!(psir)
    noprime!(psil)
    prime!(psir, "Link")

    M = ITensor[]
    for i in firstindex(sites):lastindex(sites)
        Mi = ITensor(1)

        if length(M) > 0
            Mi = M[end] * delta(prime(dag(sites[i-1])), sites[i-1])
        end

        Mi *= prime(psir[i], sites[i])
        Mi *= dag(psil[i])
        push!(M, Mi)
    end

    noprime!(psir)

    for i in (lastindex(sites)-1):-1:1
        phil = psil[i] * psil[i+1]
        phir = setprime(psir[i], 1) * setprime(psir[i+1], 1)

        idright = nothing
        if i > 1
            idright = M[i-1] * delta(prime(dag(sites[i-1])), sites[i-1])
            swapprime!(idright, 0 => 1)
        end

        phir = noprime(phir)

        nhreplacebond!(
            psil, psir, i, phil, phir, alg, idright; ortho="right", mindim, maxdim, cutoff, kwargs...
        )
    end

    noprime!(psil)
    noprime!(psir)

    return psir, psil
end
