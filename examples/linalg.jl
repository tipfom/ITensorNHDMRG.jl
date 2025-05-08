using ITensors, ITensorMPS
using LinearAlgebra

include("../src/linalg.jl")

# i = Index(QN(0) => 3, QN(1)=>5)
i = Index(QN(1) => 5, QN(0) => 3)
# i = Index(QN(0) => 10)
# i = Index(10)
M = random_itensor(ComplexF64, dag(i)', i)

maxdim = 4

B, Y, Ybar, spec = transform(M, [dag(i)'], [i]; maxdim, mindim=0, cutoff=0)
Mt = Y * B * Ybar

K = Y * replaceind(Ybar, i => i')
# K = Ybar * noprime(Y, commonind(Y, B))
@show inds(K)
display(matrix(K))

F = eigen(matrix(M))
v = copy(F.values)
vperm = sortperm(v; by=abs2, rev=true)
for i in (firstindex(vperm) + dim(commonind(B, Y))):lastindex(vperm)
    v[vperm[i]] = zero(v[vperm[i]])
end
Me = F.vectors * Diagonal(v) * inv(F.vectors)

display(matrix(Mt))
display(Me)

@show norm(matrix(Mt) - Me)