function nhtdvpoperator(H, Ml, Mr, weight)
    @assert length(Ml) == length(Mr)

    if length(Ml) == 0
        return ProjNHMPO(H; keepadj=false)
    end

    return ProjNHMPO_MPS(H, Ml, Mr; weight)
end