include("abstractprojnhmpo.jl")

mutable struct ProjNHMPO <: AbstractProjNHMPO
    lpos::Int
    rpos::Int
    nsite::Int
    H::MPO
    LR::Vector{ITensor}
end
ProjNHMPO(H::MPO) = ProjNHMPO(0, length(H) + 1, 2, H, Vector{ITensor}(undef, length(H)))

copy(P::ProjNHMPO) = ProjNHMPO(P.lpos, P.rpos, P.nsite, copy(P.H), copy(P.LR))

function set_nsite!(P::ProjNHMPO, nsite)
    P.nsite = nsite
    return P
end