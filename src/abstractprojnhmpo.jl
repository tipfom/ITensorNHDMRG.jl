using ITensors, ITensorMPS
import ITensors.OneITensor

abstract type AbstractProjNHMPO end

copy(::AbstractProjNHMPO) = error("Not implemented")

nsite(P::AbstractProjNHMPO) = P.nsite

set_nsite!(::AbstractProjNHMPO, nsite) = error("Not implemented")

# The range of center sites
site_range(P::AbstractProjNHMPO) = (P.lpos + 1):(P.rpos - 1)

Base.length(P::AbstractProjNHMPO) = length(P.H)

function ITensorMPS.lproj(P::AbstractProjNHMPO)::Union{ITensor,OneITensor}
    (P.lpos <= 0) && return OneITensor()
    return P.LR[P.lpos]
end

function ITensorMPS.rproj(P::AbstractProjNHMPO)::Union{ITensor,OneITensor}
    (P.rpos >= length(P) + 1) && return OneITensor()
    return P.LR[P.rpos]
end

function ITensors.contract(P::AbstractProjNHMPO, v::ITensor)::ITensor
    itensor_map = Union{ITensor,OneITensor}[lproj(P)]
    append!(itensor_map, P.H[site_range(P)])
    push!(itensor_map, rproj(P))

    # Reverse the contraction order of the map if
    # the first tensor is a scalar (for example we
    # are at the left edge of the system)
    if dim(first(itensor_map)) == 1
        reverse!(itensor_map)
    end

    # for it in itensor_map
    #     @show inds(it)
    # end

    # Apply the map
    Hv = v
    for it in itensor_map
        Hv *= it
    end
    return Hv
end

function ITensorMPS.product(P::AbstractProjNHMPO, v::ITensor)::ITensor
    Pv = contract(P, v)
    if order(Pv) != order(v)
        error(
            string(
                "The order of the ProjMPO-ITensor product P*v is not equal to the order of the ITensor v, ",
                "this is probably due to an index mismatch.\nCommon reasons for this error: \n",
                "(1) You are trying to multiply the ProjMPO with the $(nsite(P))-site wave-function at the wrong position.\n",
                "(2) `orthogonalize!` was called, changing the MPS without updating the ProjMPO.\n\n",
                "P*v inds: $(inds(Pv)) \n\n",
                "v inds: $(inds(v))",
            ),
        )
    end
    return noprime(Pv)
end

(P::AbstractProjNHMPO)(v::ITensor) = product(P, v)

productl(P::AbstractProjNHMPO, v::ITensor) = product(P, v)
productr(P::AbstractProjNHMPO, v::ITensor) = product(P, v)

function Base.eltype(P::AbstractProjNHMPO)::Type
    ElType = eltype(lproj(P))
    for j in site_range(P)
        ElType = promote_type(ElType, eltype(P.H[j]))
    end
    return promote_type(ElType, eltype(rproj(P)))
end

function Base.size(P::AbstractProjNHMPO)::Tuple{Int,Int}
    d = 1
    for i in inds(lproj(P))
        plev(i) > 0 && (d *= dim(i))
    end
    for j in site_range(P)
        for i in inds(P.H[j])
            plev(i) > 0 && (d *= dim(i))
        end
    end
    for i in inds(rproj(P))
        plev(i) > 0 && (d *= dim(i))
    end
    return (d, d)
end

function ITensorMPS._makeL!(
    P::AbstractProjNHMPO, psil::MPS, psir::MPS, k::Int
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

function ITensorMPS.makeL!(P::AbstractProjNHMPO, psil::MPS, psir::MPS, k::Int)
    ITensorMPS._makeL!(P, psil, psir, k)
    return P
end

function ITensorMPS._makeR!(
    P::AbstractProjNHMPO, psil::MPS, psir::MPS, k::Int
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

function ITensorMPS.makeR!(P::AbstractProjNHMPO, psil::MPS, psir::MPS, k::Int)
    ITensorMPS._makeR!(P, psil, psir, k)
    return P
end

function ITensorMPS.position!(P::AbstractProjNHMPO, psil::MPS, psir::MPS, pos::Int)
    ITensorMPS.makeL!(P, psil, psir, pos - 1)
    ITensorMPS.makeR!(P, psil, psir, pos + nsite(P))
    return P
end

function ITensorMPS.noiseterm(P::AbstractProjNHMPO, phi::ITensor, ortho::String)::ITensor
    if nsite(P) != 2
        error("noise term only defined for 2-site ProjMPO")
    end

    site_range_P = site_range(P)
    if ortho == "left"
        AL = P.H[first(site_range_P)]
        AL = lproj(P) * AL
        nt = AL * phi
    elseif ortho == "right"
        AR = P.H[last(site_range_P)]
        AR = AR * rproj(P)
        nt = phi * AR
    else
        error("In noiseterm, got ortho = $ortho, only supports `left` and `right`")
    end
    nt = nt * dag(noprime(nt))

    return nt
end

function ITensorMPS.checkflux(P::AbstractProjNHMPO)
    ITensorMPS.checkflux(P.H)
    for n in length(P.LR)
        if isassigned(P.LR, n)
            ITensorMPS.checkflux(P.LR[n])
        end
    end
    return nothing
end