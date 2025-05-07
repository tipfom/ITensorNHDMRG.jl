function transform(M::ITensor, leftinds, rightinds; kwargs...)
    CL = combiner(leftinds...)
    CR = combiner(rightinds...)

    cL = combinedind(CL)
    cR = combinedind(CR)

    M = M * CL * CR
    if inds(M) != (cL, cR)
        M = permute(M, cL, cR)
    end
    Mtensor = NDTensors.Tensor(M)

    Bi, Yi, Ybari = transform(Mtensor; kwargs...)

    Yi = Yi * dag(CL)
    Ybari = Ybari * dag(CR)

    return Bi, Yi, Ybari
end

function transform(
    M::ITensors.Tensor{ElT,2,<:ITensors.Dense}; kwargs...
) where {ElT<:Union{Real,Complex}}
    # transforms the matrix M according to the procedure outlined in App. C in 2401.15000
    lB, lY, lYbar = transform(matrix(M); kwargs...)
    @show B = first(lB)
    @show Y = first(lY)
    @show Ybar = first(lYbar)

    link = Index(size(B, 1))

    Yi = itensor(Y, inds(M)[1], link)
    Ybari = itensor(Ybar, inds(M)[2], link')
    Bi = itensor(B, link, link')

    return Bi, Yi, Ybari
end

function transform(
    M::ITensors.Tensor{ElT,2,<:ITensors.BlockSparse}; kwargs...
) where {ElT<:Union{Real,Complex}}
    # transforms the matrix M according to the procedure outlined in App. C in 2401.15000

    for b in eachnzblock(M)
        all(==(b[1]), b) ||
            error("The transformation currently only supports block diagonal matrices.")
    end

    Ms = [collect(M[b]) for b in eachnzblock(M)]

    # transform all the blocks
    lB, lY, lYbar = transform(Ms...; kwargs...)

    # find all the blocks for which the truncation reduced the size to zero
    dropblocks = Int64[]
    for i in 1:nnzblocks(M)
        if size(lB[i], 1) == 0
            # the block has shrunk to zero 
            push!(dropblocks, i)
        end
    end

    # Get the list of blocks of M that are not dropped
    nzblocksM = nzblocks(M)
    deleteat!(nzblocksM, dropblocks)
    deleteat!(lB, dropblocks)
    deleteat!(lY, dropblocks)
    deleteat!(lYbar, dropblocks)

    # The number of blocks of T remaining
    nnzblocksM = nnzblocks(M) - length(dropblocks)

    # reconstruct ITensor
    i1, i2 = inds(M)
    l = sim(i1)

    lkeepblocks = Int[b[1] for b in nzblocksM]
    ldropblocks = setdiff(1:nblocks(l), lkeepblocks)
    deleteat!(l, ldropblocks)

    # l may have too many blocks
    (nblocks(l) > nnzblocksM) && error("New index l in transform has too many blocks")

    # Truncation may have changed some block sizes
    for n in 1:nnzblocksM
        ITensors.setblockdim!(l, minimum(size(lB[n])), n)
    end

    r = dag(sim(l))

    indsB = (dag(l), r)
    indsY = (i1, l)
    indsYbar = (i2, r)

    nzblocksB = Vector{Block{2}}(undef, nnzblocksM)
    nzblocksY = Vector{Block{2}}(undef, nnzblocksM)
    nzblocksYbar = Vector{Block{2}}(undef, nnzblocksM)
    for n in 1:nnzblocksM
        blockM = nzblocksM[n]

        blockB = (n, n)
        nzblocksB[n] = blockB

        blockY = (blockM[1], n)
        nzblocksY[n] = blockY
        nzblocksYbar[n] = blockY
    end

    ElB = eltype(first(lB))
    ElY = eltype(first(lY))

    B = ITensors.BlockSparseTensor(
        ITensors.set_eltype(ITensors.unwrap_array_type(M), ElB), undef, nzblocksB, indsB
    )
    Y = ITensors.BlockSparseTensor(
        ITensors.set_eltype(ITensors.unwrap_array_type(M), ElY), undef, nzblocksY, indsY
    )
    Ybar = ITensors.BlockSparseTensor(
        ITensors.set_eltype(ITensors.unwrap_array_type(M), ElY),
        undef,
        nzblocksYbar,
        indsYbar,
    )

    # copy the data 
    for n in 1:nnzblocksM
        blockB = nzblocksB[n]
        copyto!(ITensors.blockview(B, blockB), lB[n])

        blockY = nzblocksY[n]
        copyto!(ITensors.blockview(Y, blockY), lY[n])
        copyto!(ITensors.blockview(Ybar, blockY), lYbar[n])
    end

    return itensor(B), itensor(Y), itensor(Ybar)
end

function transform(Ms::Matrix{ElT}...; keep) where {ElT<:Union{Real,Complex}}
    # transforms the matrix M according to the procedure outlined in App. C in 2401.15000
    cumdims = cumsum([size(M, 1) for M in Ms])
    @show cumdims

    Fs = Schur[]
    sizehint!(Fs, length(Ms))

    vals = zeros(complex(ElT), cumdims[end])
    for (i, M) in enumerate(Ms)
        F = schur(M)
        push!(Fs, F)
        # We have that M = F.vectors * F.Schur * F.vectors'

        vals[(i == firstindex(Ms) ? 1 : cumdims[i - 1]):cumdims[i]] = F.values
    end

    keep >= length(vals) &&
        return [F.Schur for F in Fs], [F.vectors for F in Fs], [F.vectors for F in Fs]

    # permute the Schur decomposition such that (0 ... keep) are in the beginning and the discarded 
    # part of M is at (keep+1 ... end)
    valsperm = sortperm(vals; by=abs2, rev=true)
    select = zeros(Bool, size(vals))
    for i in firstindex(valsperm):(firstindex(valsperm) + keep - 1)
        select[valsperm[i]] = true
    end

    @assert sum(select) == keep

    lB = Matrix[]
    lY = Matrix[]
    lYbar = Matrix[]

    for (i, F) in enumerate(Fs)
        selecti = select[(i == firstindex(Ms) ? 1 : cumdims[i - 1]):cumdims[i]]
        F = ordschur!(F, selecti)
        keepi = sum(selecti)

        if eltype(F) <: Real && !iszero(F.Schur[keepi + 1, keepi])
            @info "Increasing keep in block $i because of 2x2 block in the Schur decomposition"
            keepi += 1
        end

        @views A = F.Schur[1:keepi, 1:keepi]
        @views B = F.Schur[(keepi + 1):end, 1:keepi]
        @views C = F.Schur[(keepi + 1):end, (keepi + 1):end]
        @views D = F.Schur[1:keepi, (keepi + 1):end]

        @assert all(iszero, B) "matrix B contains non-zero elements $B"

        Y = copy(F.vectors)
        @views Ys = Y[:, 1:keepi]
        @views Yd = Y[:, (keepi + 1):end]
        Ybar = copy(F.vectors)
        @views Ybars = Ybar[:, 1:keepi]
        @views Ybard = Ybar[:, (keepi + 1):end]

        # solve the Sylvester equation AX - XC = D -> A X + X (-C) + (-D) = 0
        X = sylvester(A, -C, -D)

        # apply the transformation
        mul!(Ybars, Ybard, Adjoint(X), 1.0, 1.0)
        mul!(Yd, Ys, X, -1.0, 1.0)

        # K = zero(M)
        # K[1:keep, 1:keep] .= A
        # K[keep+1:end, keep+1:end] .= C

        # K, Y, Ybar
        push!(lB, A)
        push!(lY, Ys)
        push!(lYbar, Ybars)
    end

    return lB, lY, lYbar
end
