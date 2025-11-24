using ITensors, ITensorMPS
using ITensorNHDMRG

function hamiltonian(sites; tL, tR, V, t2, u, offset=nothing, scale=one(tL))
    @assert length(sites) % 2 == 0

    H = OpSum()

    N = length(sites) ÷ 2

    for l in 1:N
        la = 2(l - 1) + 1
        lb = 2(l - 1) + 2
        H += tL, "Cdag", la, "C", lb
        H += tR, "Cdag", lb, "C", la
        H += V, "N", la, "N", lb
    end

    for l in 1:(N - 1)
        lb = 2(l - 1) + 2
        lna = 2(l - 1) + 3
        H += t2, "Cdag", lb, "C", lna
        H += t2, "Cdag", lna, "C", lb
        H += V, "N", lb, "N", lna
    end

    if !iszero(u)
        for l in 1:N
            la = 2(l - 1) + 1
            lb = 2(l - 1) + 2
            H += sqrt(2) * exp(-1im * π / 4) * u, "N", la
            H += -sqrt(2) * exp(-1im * π / 4) * u, "N", lb
        end
    end

    if !isnothing(offset) && !iszero(offset)
        for l in 1:N
            H += -offset / N, "Id", l
        end
    end

    return MPO(H, sites) * scale
end

function gap(
    N;
    alg,
    biorthoalg,
    t1=1.2,
    γ=0.1,
    V=7.0,
    t2=1.0,
    u=0.0,
    nexcitedstates=1,
    weight=20.0,
    offset=nothing,
    scale=one(t1)
)
    sites = siteinds("Fermion", 2N; conserve_qns=true)
    tL = t1 - γ
    tR = t1 + γ
    # half filling

    @info "Starting constructing the Hamiltonian"
    H = hamiltonian(sites; tL, tR, V, t2, u, offset, scale)

    nsweeps = 5
    maxdim = 300
    cutoff = [
        1e-5,
        1e-9,
        1e-10,
        1e-11,
        1e-12,
    ]
    noise = [
        1e-2,
        1e-5,
        0.0,
    ]

    ψhf = [ifelse(mod(i, 2) == 0, "Occ", "Emp") for i in 1:(2N)]
    @assert count(ψhf .== "Occ") == count(ψhf .== "Emp")

    @info "Searching for the eigenvalues using DMRG with ishermitian=false"

    sweeps = Sweeps(nsweeps; maxdim, cutoff, noise)

    initial_guess = random_mps(sites, ψhf; linkdims=10)
    Edmrg, psi = dmrg(H, initial_guess, sweeps; ishermitian=false)
    @info "Found energy $Edmrg using dmrg"

    sweeps = Sweeps(nsweeps; maxdim=300, cutoff, noise)

    @info "Searching for the eigenvalues using NHDMRG"
    _, ψl0, ψr0= nhdmrg(
        H,
        initial_guess,
        initial_guess,
        sweeps;
        alg,
        biorthoalg,
        outputlevel=1,
        eigsolve_krylovdim=30,
        eigsolve_maxiter=3,
    )
    E0 = inner(ψl0', H, ψr0) / inner(ψl0, ψr0)
    @info "Found groundstate with energy $E0"

    Ψr = [ψr0]
    Ψl = [ψl0]
    Er = [E0]
    for i in 1:nexcitedstates
        initial_guess = random_mps(sites, ψhf; linkdims=5)
        Eri, ψli, ψri = nhdmrg(
            H,
            Ψl,
            Ψr,
            initial_guess,
            initial_guess,
            sweeps;
            weight,
            alg,
            biorthoalg,
            eigsolve_krylovdim=30,
            eigsolve_maxiter=3,
        )

        Ei = inner(ψli', H, ψri) / inner(ψli, ψri)
        push!(Er, Ei)
        push!(Ψr, ψri)
        push!(Ψl, ψli)
        @info "Found excited state #$i at energy $Ei ($Eri)"
    end

    overlaps = zeros(ComplexF64, length(Ψr), length(Ψl))
    for i in eachindex(Ψr)
        for j in eachindex(Ψl)
            overlaps[i, j] = inner(Ψr[i], Ψl[j])
        end
    end

    E = real.(Er)

    sort!(E)

    return E, overlaps
end

function main()
    # number of unit cells
    N = 10

    # either onesided or twosided
    alg = "onesided"
    
    # biorthogonalization routine, either `biorthoblock` or `fidelity`
    biorthoalg = "fidelity"
    
    # weight to enforce the biorthogonality constraint w.r.t. eigenstates already found
    weight = 20.0

    E, overlaps = gap(N; alg, biorthoalg, weight)

    display(overlaps)

    return nothing
end

main()