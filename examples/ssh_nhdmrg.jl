using ITensors, ITensorMPS, ITensorNHDMRG
let
    # Create 200 Fermionic indices
    N = 100
    sites = siteinds("Fermion", 2N; conserve_qns=true)

    # Input operator terms which define
    # a Hamiltonian matrix, and convert
    # these terms to an MPO tensor network
    # (here we make the 1D Heisenberg model)
    t2, tL, tR = 1.0, 0.9, 1.1

    os = OpSum()
    for j in 1:N
        ja = 2(j - 1) + 1 # index on the a subsystem
        jb = 2(j - 1) + 2 # index on the b subsystem
        os += tL, "Cdag", ja, "C", jb
        os += tR, "Cdag", jb, "C", ja
    end
    # intercell hopping 
    for l in 1:(N - 1)
        lb = 2(l - 1) + 2 # index on the b subsystem
        lna = 2(l - 1) + 3 # neighbor index on the a subsystem
        os += t2, "Cdag", lb, "C", lna
        os += t2, "Cdag", lna, "C", lb
    end
    H = MPO(os, sites)

    # Create an initial random matrix product state
    # with half filling
    psi0 = random_mps(sites, [ifelse(mod(i, 2) == 0, "Occ", "Emp") for i in 1:2N])

    # Plan to do 5 passes or 'sweeps' of DMRG,
    # setting maximum MPS internal dimensions
    # for each sweep and maximum truncation cutoff
    # used when adapting internal dimensions:
    nsweeps = 15
    maxdim = [10, 20, 100, 100, 200]
    cutoff = 1E-10

    # Run the DMRG algorithm, returning energy
    # (dominant eigenvalue) and optimized left- and right- MPS
    energy, psil, psir = nhdmrg(H, psi0, psi0; nsweeps, maxdim, cutoff)
    println("Final energy = $energy")

    nothing
end