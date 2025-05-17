using ITensors, LinearAlgebra
import ITensorNHDMRG: transform

@testset "Test Reconstruct QN Transformation $(elt)" for elt in (Float32, Float64, ComplexF32, ComplexF64)
    i = Index(QN(1) => 5, QN(0) => 3)
    M = random_itensor(elt, dag(i)', i)

    @testset "Maxdim $(maxdim)" for maxdim in 1:8
        B, Y, Ybar, spec = transform(M, [dag(i)'], [i]; maxdim, mindim=0, cutoff=0, biorthonormalize=false, unitarize=false)
        Mt = Y * B * Ybar
    
        F = eigen(matrix(M))
        v = copy(F.values)
        vperm = sortperm(v; by=abs2, rev=true)
        for i in (firstindex(vperm) + dim(commonind(B, Y))):lastindex(vperm)
            v[vperm[i]] = zero(v[vperm[i]])
        end
        Me = F.vectors * Diagonal(v) * inv(F.vectors)
    
        @show elt, maxdim
        @show norm(matrix(Mt) - Me)
        @test isapprox(matrix(Mt), Me; atol=sqrt(eps(real(elt))), rtol=sqrt(eps(real(elt))))
    end
end

@testset "Test Reconstruct Transformation $(elt)" for elt in (Float32, Float64, ComplexF32, ComplexF64)
    i = Index(10)
    M = random_itensor(elt, dag(i)', i)

    @testset "Maxdim $(maxdim)" for maxdim in 1:8
        B, Y, Ybar, spec = transform(M, [dag(i)'], [i]; maxdim, mindim=0, cutoff=0, biorthonormalize=false, unitarize=false)
        Mt = Y * B * Ybar
    
        F = eigen(matrix(M))
        v = copy(F.values)
        vperm = sortperm(v; by=abs2, rev=true)
        for i in (firstindex(vperm) + dim(commonind(B, Y))):lastindex(vperm)
            v[vperm[i]] = zero(v[vperm[i]])
        end
        Me = F.vectors * Diagonal(v) * inv(F.vectors)
    
        @show elt, maxdim
        @show norm(matrix(Mt) - Me)
        @test isapprox(matrix(Mt), Me; atol=sqrt(eps(real(elt))), rtol=sqrt(eps(real(elt))))
    end
end


@testset "Test Biorthogonality $(elt)" for elt in (Float32, Float64, ComplexF32, ComplexF64)
    i = Index(QN(1) => 5, QN(0) => 3)
    M = random_itensor(elt, dag(i)', i)

    @testset "Maxdim $maxdim" for maxdim in 1:8
        @testset "Biorthonormalize $biorthonormalize" for biorthonormalize in [true, false]
            @testset "Unitarize $unitarize" for unitarize in [true, false]            
                B, Y, Ybar, spec = transform(M, [dag(i)'], [i]; maxdim, mindim=0, cutoff=0, biorthonormalize, unitarize)
            
                K = Y * replaceind(Ybar, i => i')
                @test matrix(K) ≈ I
            end
        end
    end
end