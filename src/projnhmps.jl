using ITensors
import ITensorMPS: MPS

mutable struct ProjNHMPS
    lpos::Int
    rpos::Int
    nsite::Int
    Ml::MPS
    Mr::MPS
    LRl::Vector{ITensor}
    LRr::Vector{ITensor}
end
function ProjNHMPS(Ml::MPS, Mr::MPS)
    return ProjNHMPS(
        0,
        length(Ml) + 1,
        2,
        Ml,
        Mr,
        Vector{ITensor}(undef, length(Ml)),
        Vector{ITensor}(undef, length(Mr)),
    )
end

function copy(P::ProjNHMPS)
    return ProjNHMPS(
        P.lpos, P.rpos, P.nsite, copy(P.Ml), copy(P.Mr), copy(P.LRl), copy(P.LRr)
    )
end

nsite(P::ProjNHMPS) = P.nsite

# The range of center sites
# TODO: Use the `AbstractProjMPO` version.
site_range(P::ProjNHMPS) = (P.lpos + 1):(P.rpos - 1)

function set_nsite!(P::ProjNHMPS, nsite)
    P.nsite = nsite
    return P
end

Base.length(P::ProjNHMPS) = length(P.Ml)

function lprojl(P::ProjNHMPS)
    (P.lpos <= 0) && return nothing
    return P.LRl[P.lpos]
end

function rprojl(P::ProjNHMPS)
    (P.rpos >= length(P) + 1) && return nothing
    return P.LRl[P.rpos]
end

function lprojr(P::ProjNHMPS)
    (P.lpos <= 0) && return nothing
    return P.LRl[P.lpos]
end

function rprojr(P::ProjNHMPS)
    (P.rpos >= length(P) + 1) && return nothing
    return P.LRl[P.rpos]
end

function productl(P::ProjNHMPS, vl::ITensor)::ITensor
    if nsite(P) != 2
        error("Only two-site ProjMPS currently supported")
    end

    Lpmr = dag(prime(P.Mr[P.lpos + 1], "Link"))
    !isnothing(lprojr(P)) && (Lpmr *= lprojr(P))

    Rpmr = dag(prime(P.Mr[P.rpos - 1], "Link"))
    !isnothing(rprojr(P)) && (Rpmr *= rprojr(P))

    Lpml = dag(prime(P.Mr[P.lpos + 1], "Link"))
    !isnothing(lprojr(P)) && (Lpml *= lprojr(P))

    Rpml = dag(prime(P.Mr[P.rpos - 1], "Link"))
    !isnothing(rprojr(P)) && (Rpml *= rprojr(P))

    pmr = Lpmr * Rpmr
    pml = Lpml * Rpml

    pv = scalar(pmr * vl)

    Mv = pv * dag(pml)

    return noprime(Mv)
end

function productr(P::ProjNHMPS, vr::ITensor)::ITensor
    if nsite(P) != 2
        error("Only two-site ProjMPS currently supported")
    end

    Lpmr = dag(prime(P.Mr[P.lpos + 1], "Link"))
    !isnothing(lprojr(P)) && (Lpmr *= lprojr(P))

    Rpmr = dag(prime(P.Mr[P.rpos - 1], "Link"))
    !isnothing(rprojr(P)) && (Rpmr *= rprojr(P))

    Lpml = dag(prime(P.Mr[P.lpos + 1], "Link"))
    !isnothing(lprojr(P)) && (Lpml *= lprojr(P))

    Rpml = dag(prime(P.Mr[P.rpos - 1], "Link"))
    !isnothing(rprojr(P)) && (Rpml *= rprojr(P))

    pmr = Lpmr * Rpmr
    pml = Lpml * Rpml

    pv = scalar(pml * vr)

    Mv = pv * dag(pmr)

    return noprime(Mv)
end

function ITensorMPS.makeL!(P::ProjNHMPS, psil::MPS, psir::MPS, k::Int)
    while P.lpos < k
        ll = P.lpos
        if ll <= 0
            P.LRl[1] = psir[1] * dag(prime(P.Ml[1], "Link"))
            P.LRr[1] = psil[1] * dag(prime(P.Mr[1], "Link"))
            P.lpos = 1
        else
            P.LRl[ll + 1] = P.LRl[ll] * psir[ll + 1] * dag(prime(P.Ml[ll + 1], "Link"))
            P.LRr[ll + 1] = P.LRr[ll] * psil[ll + 1] * dag(prime(P.Mr[ll + 1], "Link"))
            P.lpos += 1
        end
    end
end

function ITensorMPS.makeR!(P::ProjNHMPS, psil::MPS, psir::MPS, k::Int)
    N = length(P.Mr)
    while P.rpos > k
        rl = P.rpos
        if rl >= N + 1
            P.LRl[N] = psir[N] * dag(prime(P.Ml[N], "Link"))
            P.LRr[N] = psil[N] * dag(prime(P.Mr[N], "Link"))
            P.rpos = N
        else
            P.LRl[rl - 1] = P.LRl[rl] * psir[rl - 1] * dag(prime(P.Ml[rl - 1], "Link"))
            P.LRr[rl - 1] = P.LRr[rl] * psil[rl - 1] * dag(prime(P.Mr[rl - 1], "Link"))
            P.rpos -= 1
        end
    end
end

function ITensorMPS.position!(P::ProjNHMPS, psil::MPS, psir::MPS, pos::Int)
    ITensorMPS.makeL!(P, psil, psir, pos - 1)
    ITensorMPS.makeR!(P, psil, psir, pos + nsite(P))

    #These next two lines are needed
    #when moving lproj and rproj backward
    P.lpos = pos - 1
    P.rpos = pos + nsite(P)
    return P
end

function ITensorMPS.checkflux(P::ProjNHMPS)
    checkflux(P.Ml)
    checkflux(P.Mr)
    foreach(eachindex(P.LRl)) do i
        isassigned(P.LRl, i) && checkflux(P.LRl[i])
    end
    foreach(eachindex(P.LRr)) do i
        isassigned(P.LRr, i) && checkflux(P.LRr[i])
    end
    return nothing
end