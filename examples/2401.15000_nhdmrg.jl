using ITensors, ITensorMPS
using GLMakie
using ITensorNHDMRG: nhdmrg

function hamiltonian(sites; tL, tR, V, t2, u)
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

    for l in 1:N-1
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

    MPO(H, sites)
end

function gap(N; t1=1.2, γ=0.1, V=7.0, t2=1.0, u=0.0, alg="twosided")
    sites = siteinds("Fermion", 2N; conserve_qns=true)
    tL = t1 - γ
    tR = t1 + γ
    # half filling

    @info "starting constructing the Hamiltonian"
    H = hamiltonian(sites; tL, tR, V, t2, u)

    weight = 200.0
    nsweeps = 100
    maxdim = 100
    cutoff = [fill(1e-5, 6)..., fill(1e-7, 6)..., fill(1e-9, 6)..., fill(1e-10, 6)..., fill(1e-11, 4)..., 1e-12]
    noise = [fill(1e-3, 20)..., fill(1e-5, 30)..., fill(1e-7, 6)..., fill(1e-8, 6)..., fill(1e-9, 2)..., 0.0]

    refinementweight = 20000.0
    refinementnsweeps = 20
    refinementmaxdim = 100
    refinementcutoff = [1e-13]
    refinementnoise = [0.0]


    ψhf = [ifelse(mod(i, 2) == 0, "Occ", "Emp") for i in 1:2N]
    @assert count(ψhf .== "Occ") == count(ψhf .== "Emp")

    @info "searching for the eigenvalues"

    nexcitedstates = 1

    sweeps = Sweeps(nsweeps; maxdim, cutoff, noise)
    refinementsweeps = Sweeps(refinementnsweeps; maxdim=refinementmaxdim, cutoff=refinementcutoff, noise=refinementnoise)

    initial_guess = random_mps(sites, ψhf; linkdims=5)
    Er0, ψr0, ψl0 = nhdmrg(H, initial_guess, initial_guess, sweeps; alg)
    @info "Found groundstate with energy $Er0"
    @show E0 = ITensorMPS.inner(ψl0', H, ψr0) / ITensorMPS.inner(ψl0, ψr0)
    @show ITensorMPS.inner(ψl0, ψr0)

    Ψr = [ψr0]
    Ψl = [ψl0]
    Er = [Er0]
    for i in 1:nexcitedstates
        initial_guess = random_mps(sites, ψhf; linkdims=5)
        Eri, ψri, ψli = nhdmrg(H, Ψl, Ψr, initial_guess, initial_guess, sweeps; weight, alg)
        @info "refining the biorthogonality"
        Eri, ψri, ψli = nhdmrg(H, Ψl, Ψr, ψri, ψli, refinementsweeps; weight=refinementweight, alg, isbiortho=true)
        Ei = ITensorMPS.inner(ψli', H, ψri) / ITensorMPS.inner(ψli, ψri)
        push!(Er, Ei)
        @show inner(ψl0, ψri)
        @show inner(ψr0, ψli)
        push!(Ψr, ψri)
        push!(Ψl, ψli)
        @info "Found excited state #$i at energy $Eri, $Ei"
    end

    E = real.(Er)

    sort!(E)

    E[2] - E[1]
end

function main()
    fig = Figure()
    ax = Axis(fig[1, 1], ylabel="gap Δ", xlabel="cos(π/(N+2)) - cos(2π/(N+2))")

    x_refV7 = [0.0002686757215619695, 0.0003641765704584041, 0.0004507640067911715, 0.0005691850594227505, 0.0008887945670628184, 0.0015776740237691002,]
    y_refV7 = [0.0016913319238900635, 0.00226215644820296, 0.0028329809725158566, 0.003572938689217759, 0.0054968287526427064, 0.009661733615221988,]

    x_refV14 = [0.0002686757215619695, 0.00036290322580645164, 0.000567911714770798, 0.0008862478777589135,]
    y_refV14 = [0.0009090909090909092, 0.001247357293868922, 0.0019661733615221988, 0.003086680761099366,]

    c7 = :red
    c14 = :blue

    y_refdmrg = [0.1524727667218535 0.07890564602254813 0.03213118993691921; 0.10051067204251751 0.048929772932453375 0.018927706144852507]
    N_refdmrg = [20, 30, 50]

    f(x) = cos(π / (x + 2)) - cos(2π / (x + 2))

    append!(x_refV7, reverse(f.(N_refdmrg)))
    append!(y_refV7, reverse(y_refdmrg[1, :]))
    
    append!(x_refV14, reverse(f.(N_refdmrg)))
    append!(y_refV14, reverse(y_refdmrg[2, :]))
    
    plot!(ax, x_refV7, y_refV7, label="Ref. V = 7.0", color=(c7, 0.5))
    lines!(ax, x_refV7, y_refV7, color=(c7, 0.5))
    plot!(ax, x_refV14, y_refV14, label="Ref. V = 14.0", color=(c14, 0.5))
    lines!(ax, x_refV14, y_refV14, color=(c14, 0.5))


    Ns = [20, 100, 200]
    Ns = [20, 50, 100, 150, 200]
    Ns = [150]
    Vs = [(c7, 7.0), (c14, 14.0)]
    Vs = [(c7, 7.0)]
    methods = [("twosided", :star5), ("onesided", :xcross)]
    methods = [("twosided", :star5)]

    for (alg, marker) in methods 
        @info "running method $alg"

        Δ1s = zeros(length(Vs), length(Ns))
        Δ2s = zeros(length(Vs), length(Ns))
        for i in CartesianIndices(Δ1s)
            V = Vs[i[1]][2]
            N = Ns[i[2]]
    
            Δ1s[i] = gap(N; V, alg)
        end
    
        @show Δ1s
        @show Δ2s
    
        for (i, x) in enumerate(Vs)
            c, V = x
            plot!(ax, f.(Ns), Δ1s[i, :]; label=ifelse(i == 1, "$alg", nothing), color=c, marker)
        end
    end


    axislegend(ax; position=:lt)
    display(fig)
end

main()