using ITensors, ITensorMPS

mutable struct ProjNHMPO <: ITensorMPS.AbstractProjMPO
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

productl(P::ProjNHMPO, v::ITensor) = product(P, v)
productr(P::ProjNHMPO, v::ITensor) = product(P, v)

nsite(P::ProjNHMPO) = P.nsite

function ITensorMPS._makeL!(
    P::ProjNHMPO, psil::MPS, psir::MPS, k::Int
)::Union{ITensor,Nothing}
    # Save the last `L` that is made to help with caching
    # for DiskProjMPO
    ll = P.lpos
    if ll ≥ k
        # Special case when nothing has to be done.
        # Still need to change the position if lproj is
        # being moved backward.
        P.lpos = k
        return nothing
    end
    # Make sure ll is at least 0 for the generic logic below
    ll = max(ll, 0)
    L = lproj(P)
    while ll < k
        L = L * psir[ll + 1] * P.H[ll + 1] * dag(prime(psil[ll + 1]))
        P.LR[ll + 1] = L
        ll += 1
    end
    # Needed when moving lproj backward.
    P.lpos = k
    return L
end

function ITensorMPS.makeL!(P::ProjNHMPO, psil::MPS, psir::MPS, k::Int)
    ITensorMPS._makeL!(P, psil, psir, k)
    return P
end

function ITensorMPS._makeR!(
    P::ProjNHMPO, psil::MPS, psir::MPS, k::Int
)::Union{ITensor,Nothing}
    # Save the last `R` that is made to help with caching
    # for DiskProjMPO
    rl = P.rpos
    if rl ≤ k
        # Special case when nothing has to be done.
        # Still need to change the position if rproj is
        # being moved backward.
        P.rpos = k
        return nothing
    end
    N = length(P.H)
    # Make sure rl is no bigger than `N + 1` for the generic logic below
    rl = min(rl, N + 1)
    R = rproj(P)
    while rl > k
        R = R * psir[rl - 1] * P.H[rl - 1] * dag(prime(psil[rl - 1]))
        P.LR[rl - 1] = R
        rl -= 1
    end
    P.rpos = k
    return R
end

function ITensorMPS.makeR!(P::ProjNHMPO, psil::MPS, psir::MPS, k::Int)
    ITensorMPS._makeR!(P, psil, psir, k)
    return P
end

function ITensorMPS.position!(P::ProjNHMPO, psil::MPS, psir::MPS, pos::Int)
    ITensorMPS.makeL!(P, psil, psir, pos - 1)
    ITensorMPS.makeR!(P, psil, psir, pos + nsite(P))
    return P
end
