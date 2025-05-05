using ITensors

mutable struct ProjNHMPO_MPS
    PH::ProjNHMPO
    pm::Vector{ProjNHMPS}
    weight::Float64
end

copy(P::ProjNHMPO_MPS) = ProjNHMPO_MPS(copy(P.PH), copy.(P.pm), P.weight)

function ProjNHMPO_MPS(H::MPO, mpsvl::Vector{MPS}, mpsvr::Vector{MPS}; weight=1.0)
    @assert length(mpsvl) == length(mpsvr)
    return ProjNHMPO_MPS(
        ProjNHMPO(H), [ProjNHMPS(mpsvl[i], mpsvr[i]) for i in eachindex(mpsvl)], weight
    )
end

ITensorMPS.nsite(P::ProjNHMPO_MPS) = ITensorMPS.nsite(P.PH)

function ITensorMPS.set_nsite!(Ps::ProjNHMPO_MPS, nsite)
    ITensorMPS.set_nsite!(Ps.PH, nsite)
    for P in Ps.pm
        ITensorMPS.set_nsite!(P, nsite)
    end
    return Ps
end

Base.length(P::ProjNHMPO_MPS) = length(P.PH)

function ITensorMPS.site_range(P::ProjNHMPO_MPS)
    r = ITensorMPS.site_range(P.PH)
    @assert all(m -> ITensorMPS.site_range(m) == r, P.pm)
    return r
end

function productl(P::ProjNHMPO_MPS, vl::ITensor)::ITensor
    Pv = productl(P.PH, vl)
    for p in P.pm
        Pv += P.weight * productl(p, vl)
    end
    return Pv
end

function productr(P::ProjNHMPO_MPS, vr::ITensor)::ITensor
    Pv = productr(P.PH, vr)
    for p in P.pm
        Pv += P.weight * productr(p, vr)
    end
    return Pv
end

function Base.eltype(P::ProjNHMPO_MPS)
    elT = eltype(P.PH)
    for p in P.pm
        elT = promote_type(elT, eltype(p))
    end
    return elT
end

Base.size(P::ProjNHMPO_MPS) = size(P.H)

function ITensorMPS.position!(P::ProjNHMPO_MPS, psil::MPS,psir::MPS, pos::Int)
    ITensorMPS.position!(P.PH, psil, psir, pos)
    for p in P.pm
        ITensorMPS.position!(p, psil, psir, pos)
    end
    return P
end

ITensorMPS.noiseterm(P::ProjNHMPO_MPS, thetal::ITensor, thetar::ITensor, ortho::String) = ITensorMPS.noiseterm(P.PH, thetal, thetar, ortho)

function ITensorMPS.checkflux(P::ProjNHMPO_MPS)
    checkflux(P.PH)
    foreach(checkflux, P.pm)
    return nothing
end