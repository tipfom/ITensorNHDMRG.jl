using ITensors
import ITensorMPS: MPS

mutable struct ProjNHMPS
    projL::ITensorMPS.ProjMPS
    projR::ITensorMPS.ProjMPS
    cached_site_range::AbstractRange
    envL::ITensor
    envR::ITensor
end
function ProjNHMPS(Ml::MPS, Mr::MPS)
    return ProjNHMPS(
        ITensorMPS.ProjMPS(Ml),
        ITensorMPS.ProjMPS(Mr),
        -1:-1,
        emptyITensor(),
        emptyITensor(),
    )
end

function copy(P::ProjNHMPS)
    return ProjNHMPS(copy(P.projL), copy(P.projR), P.cached_site_range, copy(P.envL), copy(P.envR))
end

ITensorMPS.nsite(P::ProjNHMPS) = ITensorMPS.nsite(P.projL)

ITensorMPS.site_range(P::ProjNHMPS) = ITensorMPS.site_range(P.projL)

function set_nsite!(P::ProjNHMPS, nsite)
    set_nsite!(P.projL, nsite)
    set_nsite!(P.projR, nsite)
    return P
end

Base.length(P::ProjNHMPS) = length(P.projL)

function getLR(P::ITensorMPS.ProjMPS)
    L = dag(prime(P.M[P.lpos + 1], "Link"))
    !isnothing(lproj(P)) && (L *= lproj(P))

    R = dag(prime(P.M[P.rpos - 1], "Link"))
    !isnothing(rproj(P)) && (R *= rproj(P))

    return L, R
end

function getenvironments(P::ProjNHMPS)
    ITensorMPS.site_range(P) == P.cached_site_range && return P.envL, P.envR

    Lr, Rr = getLR(P.projR)
    Ll, Rl = getLR(P.projL)

    P.envR = Lr * Rr
    P.envL = Ll * Rl

    P.cached_site_range = ITensorMPS.site_range(P)

    return P.envR, P.envL
end

function productl(P::ProjNHMPS, vl::ITensor)::ITensor
    if nsite(P) != 2
        error("Only two-site ProjMPS currently supported")
    end

    envR, envL = getenvironments(P)

    pv = scalar(envR * vl)

    Mv = pv * dag(envL)

    return noprime(Mv)
end

function productr(P::ProjNHMPS, vr::ITensor)::ITensor
    if nsite(P) != 2
        error("Only two-site ProjMPS currently supported")
    end

    envR, envL = getenvironments(P)

    pv = scalar(envL * vr)

    Mv = pv * dag(envR)

    return noprime(Mv)
end

function ITensorMPS.position!(P::ProjNHMPS, psil::MPS, psir::MPS, pos::Int)
    ITensorMPS.position!(P.projR, psil, pos)
    ITensorMPS.position!(P.projL, psir, pos)

    return P
end

function ITensorMPS.checkflux(P::ProjNHMPS)
    checkflux(P.projL)
    checkflux(P.projR)
    return nothing
end