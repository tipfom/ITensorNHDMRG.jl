using ITensors, ITensorMPS
# using GLMakie
# using Revise
using ITensorNHDMRG: nhdmrg
using ArgParse
using HDF5

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "-N"
        help = "system size"
        arg_type = Int
        default = 10
        "--alg"
        help = "local eigenvalue algorithm, either `onesided` or `twosided`"
        arg_type = String
        default = "twosided"
        "--biorthoalg"
        help = "biorthogonalization routine, either `biorthoblock` or `lrdensity`"
        arg_type = String
        default = "biorthoblock"
        "--weight"
        help = "weight to enforce the biorthogonality constraint w.r.t. eigenstates already found"
        arg_type = Float64
        default = 20.0
        "--filename"
        help = "file to export results to"
        arg_type = String
        default = "export.hdf5"
    end

    return parse_args(s)
end

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

    if !isnothing(offset)
        for l in 1:N
            H += -offset / N, "Id", l
        end
    end

    return MPO(H, sites) * scale
end

function gap(
    N;
    t1=1.2,
    γ=0.1,
    V=7.0,
    t2=1.0,
    u=0.0,
    alg="twosided",
    biorthoalg="lrdensity",
    nexcitedstates=1,
    weight=20.0,
)
    sites = siteinds("Fermion", 2N; conserve_qns=true)
    tL = t1 - γ
    tR = t1 + γ
    # half filling

    @info "starting constructing the Hamiltonian"
    H = hamiltonian(sites; tL, tR, V, t2, u)

    nsweeps = 100
    maxdim = 100
    cutoff = [
        fill(1e-5, 6)...,
        fill(1e-7, 6)...,
        fill(1e-9, 6)...,
        fill(1e-10, 6)...,
        fill(1e-11, 4)...,
        1e-12,
    ]
    noise = [
        fill(1e-3, 20)...,
        fill(1e-5, 30)...,
        fill(1e-7, 6)...,
        fill(1e-8, 6)...,
        fill(1e-9, 2)...,
        0.0,
    ]

    ψhf = [ifelse(mod(i, 2) == 0, "Occ", "Emp") for i in 1:(2N)]
    @assert count(ψhf .== "Occ") == count(ψhf .== "Emp")

    @info "searching for the eigenvalues"

    sweeps = Sweeps(nsweeps; maxdim, cutoff, noise)

    initial_guess = random_mps(sites, ψhf; linkdims=5)
    _, ψr0, ψl0 = nhdmrg(
        H,
        initial_guess + random_mps(sites, ψhf; linkdims=5),
        initial_guess + random_mps(sites, ψhf; linkdims=5),
        sweeps;
        alg,
        biorthoalg,
    )
    E0 = inner(ψl0', H, ψr0) / inner(ψl0, ψr0)
    @info "Found groundstate with energy $E0"

    Ψr = [ψr0]
    Ψl = [ψl0]
    Er = [E0]
    for i in 1:nexcitedstates
        initial_guess = random_mps(sites, ψhf; linkdims=5)
        Eri, ψri, ψli = nhdmrg(
            H,
            Ψl,
            Ψr,
            initial_guess + random_mps(sites, ψhf; linkdims=5),
            initial_guess + random_mps(sites, ψhf; linkdims=5),
            sweeps;
            weight,
            alg,
            biorthoalg,
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
    args = parse_commandline()
    @show args

    N = args["N"]
    alg = args["alg"]
    biorthoalg = args["biorthoalg"]
    weight = args["weight"]

    E, overlaps = gap(N; alg, biorthoalg, weight)

    h5open(args["filename"], "w") do f
        write(f, "E", E)
        write(f, "overlaps", overlaps)

        for (k, v) in args
            write(f, "params/$k", v)
        end
    end

    return nothing
end

main()