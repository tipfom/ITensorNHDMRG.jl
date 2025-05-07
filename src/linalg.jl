function transform(M::ITensor, leftinds, rightinds; kwargs...)
    CL = combiner(leftinds...)
    CR = combiner(rightinds...)

    cL = combinedind(CL)
    cR = combinedind(CR)

    M = M * CL * CR

    Mtensor = NDTensors.Tensor(M)
    if inds(Mtensor) != (cL, cR)
        Mtensor = permute(Mtensor, cL, cR)
    end

    Bi, Yi, Ybari = transform(Mtensor; kwargs...)

    Yi = Yi * dag(CL)
    Ybari = Ybari * dag(CR)

    return Bi, Yi, Ybari
end

# for b in eachnzblock(T)
#     all(==(b[1]), b) || error("Eigen currently only supports block diagonal matrices.")
# end

function transform(
    M::ITensors.Tensor{ElT,2,<:ITensors.Dense}; kwargs...
) where {ElT<:Union{Real,Complex}}
    # transforms the matrix M according to the procedure outlined in App. C in 2401.15000
    B, Ys, Ybars = transform(matrix(M); kwargs...)

    link = Index(size(B, 1))

    Yi = itensor(Ys, inds(M)[1], link)
    Ybari = itensor(Ybars, inds(M)[2], link')
    Bi = itensor(B, link, link')

    return Bi, Yi, Ybari
end

function transform(M::Matrix{ElT}; keep) where {ElT<:Union{Real,Complex}}
    # transforms the matrix M according to the procedure outlined in App. C in 2401.15000
    F = schur(M)
    # We have that M = F.vectors * F.Schur * F.vectors'

    # permute the Schur decomposition such that (0 ... keep) are in the beginning and the discarded 
    # part of M is at (keep+1 ... end)
    vals = F.values
    keep >= length(vals) && return F.Schur, F.vectors, F.vectors
    
    valsperm = sortperm(vals; by=abs2, rev=true)
    
    if eltype(F) <: Real  && !iszero(F.Schur[keep + 1, keep])
        @info "Increasing $keep because of 2x2 block in the Schur decomposition"
        keep += 1
    end

    select = zeros(Bool, size(vals))
    for i in firstindex(valsperm):(firstindex(valsperm) + keep - 1)
        select[valsperm[i]] = true
    end

    @assert sum(select) == keep

    F = ordschur!(F, select)

    @views A = F.Schur[1:keep, 1:keep]
    @views B = F.Schur[(keep + 1):end, 1:keep]
    @views C = F.Schur[(keep + 1):end, (keep + 1):end]
    @views D = F.Schur[1:keep, (keep + 1):end]

    @assert all(iszero, B) "matrix B contains non-zero elements $B"

    Y = copy(F.vectors)
    @views Ys = Y[:, 1:keep]
    @views Yd = Y[:, (keep + 1):end]
    Ybar = copy(F.vectors)
    @views Ybars = Ybar[:, 1:keep]
    @views Ybard = Ybar[:, (keep + 1):end]

    # solve the Sylvester equation AX - XC = D -> A X + X (-C) + (-D) = 0
    X = sylvester(A, -C, -D)

    # apply the transformation
    mul!(Ybars, Ybard, Adjoint(X), 1.0, 1.0)
    mul!(Yd, Ys, X, -1.0, 1.0)

    # K = zero(M)
    # K[1:keep, 1:keep] .= A
    # K[keep+1:end, keep+1:end] .= C

    # return K, Y, Ybar
    return A, Ys, Ybars
end
