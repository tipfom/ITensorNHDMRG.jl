using ITensors, ITensorMPS
using LinearAlgebra

include("../src/linalg.jl")

# i = Index(QN(0) => 3, QN(1)=>5)
i = Index(QN(1) => 5, QN(0) => 3)
# i = Index(QN(0) => 10)
# i = Index(10)
M = random_itensor(ComplexF64, dag(i)', i)

keep = 3

B, Y, Ybar = transform(M, [dag(i)'], [i]; keep)
Mt = Y * B * Ybar

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