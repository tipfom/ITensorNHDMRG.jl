using ITensors, LinearAlgebra, ITensorNHDMRG

function main()
    l1, l2 = Index(QN("Sz", -1) => 3, QN("Sz", 1) => 1; tags="l1", dir=ITensors.In),
    Index(QN("Sz", 2) => 1, QN("Sz", 1) => 1; tags="l2", dir=ITensors.Out)
    r1, r2, r3 = Index(QN("Sz", -2) => 1, QN("Sz", 1) => 3; tags="r1", dir=ITensors.Out),
    Index(QN("Sz", 2) => 1, QN("Sz", 1) => 3; tags="r2", dir=ITensors.In),
    Index(QN("Sz", -2) => 1, QN("Sz", 1) => 3; tags="r3", dir=ITensors.In)
    A = random_itensor(ComplexF64, l1, l2, r1, r2, r3)
    Ainv = pinv(deepcopy(A), l1, r2, r3)
    @show inds(A)
    @show inds(Ainv)
    c1 = combiner(l2, r1)
    A *= c1
    Ainv *= c1

    return dag(prime(Ainv, combinedind(c1))) * A
end

function main2()
    @show i = Index(20)
    @show j = Index(20)
    A = random_itensor(ComplexF64, j', i)
    Ainv = pinv(deepcopy(A), j')

    M = dag(replaceind(Ainv, j'=>j'')) * A
    @show inds(M)
    return M
end

# function test(A)
#     F = svd(A)
#     F.U * diagm(1 ./ F.S) * F.Vt
# end

display(matrix(main()))
# @show (main())
display(matrix(main2()))