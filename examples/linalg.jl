using ITensors, ITensorMPS, ITensorNHDMRG
using LinearAlgebra

i = Index(5)
M = random_itensor(i, i')

B, Y, Ybar = ITensorNHDMRG.transform(M, [i], [i']; keep=3)
Mt = Y * B * dag(Ybar)

display(matrix(M))
display(matrix(Mt))

F = eigen(matrix(M))
v = copy(F.values)
vperm = sortperm(v; by=abs2)
v[vperm[1]] = 0
v[vperm[2]] = 0
Me = F.vectors * Diagonal(v) * inv(F.vectors)
display(Me)

@show norm(M - Mt) / norm(M)