using KrylovKit: bieigsolve, eigsolve, BiArnoldi, linsolve
using LinearAlgebra, KrylovKit
using VectorInterface: VectorInterface

function nhproblemsolver!(
    ::Algorithm"twosided",
    PH,
    Θl,
    Θr;
    eigsolve_tol,
    eigsolve_krylovdim,
    eigsolve_maxiter,
    eigsolve_verbosity,
    eigsolve_which_eigenvalue,
    max_krylovdim=200,
)
    fA = x -> product(PH, x)
    fAH = x -> adjointproduct(PH, x)
    f = (fA, fAH)
    
    vals, V, W, info = bieigsolve(
        f,
        Θr,
        Θl,
        1,
        eigsolve_which_eigenvalue,
        BiArnoldi(;
            tol=eigsolve_tol,
            krylovdim=eigsolve_krylovdim,
            maxiter=eigsolve_maxiter,
            verbosity=eigsolve_verbosity,
        ),
    )

    while length(vals) < 1
        # did not converge, retrying 
        eigsolve_maxiter = max(eigsolve_maxiter + 1, div(5eigsolve_maxiter, 3))
        eigsolve_krylovdim = max(eigsolve_krylovdim + 1, div(5eigsolve_krylovdim, 3))

        if eigsolve_krylovdim > max_krylovdim
            error("Did not converge")
            return eigproblemsolver!(
                Algorithm("onesided"), #
                PH,
                Θl,
                Θr;
                eigsolve_tol,
                eigsolve_krylovdim,
                eigsolve_maxiter,
                eigsolve_verbosity,
                eigsolve_which_eigenvalue,
            )
        end

        @warn "Eigensolver did not converge, consider increasing the krylovdimension or iterations; now using eigsolve_krylovdim=$eigsolve_krylovdim and eigsolve_maxiter=$eigsolve_maxiter."

        vals, V, W, info = bieigsolve(
            f,
            Θr,
            Θl,
            1,
            eigsolve_which_eigenvalue,
            BiArnoldi(;
                tol=eigsolve_tol,
                krypsil0lovdim=eigsolve_krylovdim,
                maxiter=eigsolve_maxiter,
                verbosity=eigsolve_verbosity,
            ),
        )
    end

    return first(vals), first(W), first(V)
end

function nhproblemsolver!(
    ::Algorithm"onesided",
    PH,
    Θl,
    Θr;
    eigsolve_tol,
    eigsolve_krylovdim,
    eigsolve_maxiter,
    eigsolve_verbosity,
    eigsolve_which_eigenvalue,
)
    fA = x -> product(PH, x)
    fAH = x -> adjointproduct(PH, x)

    valsH, vecsH = eigsolve(
        fAH,
        Θl,
        1,
        eigsolve_which_eigenvalue;
        ishermitian=false,
        tol=eigsolve_tol,
        krylovdim=eigsolve_krylovdim,
        maxiter=eigsolve_maxiter,
        verbosity=eigsolve_verbosity,
    )

    vals, vecs = eigsolve(
        fA,
        Θr,
        1,
        eigsolve_which_eigenvalue;
        ishermitian=false,
        tol=eigsolve_tol,
        krylovdim=eigsolve_krylovdim,
        maxiter=eigsolve_maxiter,
        verbosity=eigsolve_verbosity,
    )

    return first(vals), noprime(first(vecsH)), noprime(first(vecs))
end

function selecteigenvalue(f::Function, x)
    return f(x)
end

function selecteigenvalue(f::Number, x)
    return f
end

function selecteigenvalue(f, x)
    return x
end

function iteraterayleighquotient(
    PH,
    left,
    right;
    eigsolve_tol,
    eigsolve_krylovdim,
    eigsolve_maxiter,
    eigsolve_verbosity,
    eigsolve_which_eigenvalue,
)
    ρ = inner(left, product(PH, right)) / inner(left, right)

    λt = selecteigenvalue(eigsolve_which_eigenvalue, ρ)

    nr, info = linsolve(
        x -> product(PH, x),
        right,
        -λt;
        tol=eigsolve_tol,
        krylovdim=eigsolve_krylovdim,
        maxiter=eigsolve_maxiter,
        verbosity=eigsolve_verbosity,
    )
    nl, info = linsolve(
        x -> adjointproduct(PH, x),
        left,
        -conj(λt);
        tol=eigsolve_tol,
        krylovdim=eigsolve_krylovdim,
        maxiter=eigsolve_maxiter,
        verbosity=eigsolve_verbosity,
    )

    return nl, nr
end

function nhproblemsolver!(
    ::Algorithm"twosidedinverse",
    PH,
    Θl,
    Θr;
    eigsolve_tol,
    eigsolve_krylovdim,
    eigsolve_maxiter,
    eigsolve_verbosity,
    eigsolve_which_eigenvalue,
    eigsolve_refinements=2,
)
    # this function uses the two-sided inverse iteration and two-sided rayleigh quotient 
    # to anchor to the closest eigenvalue, see: https://doi.org/10.1002/nla.1945

    # by supplying a function to eigsolve_which_eigenvalue, the selected eigenvalue may be altered

    for _ in 1:eigsolve_refinements
        Θl, Θr = iteraterayleighquotient(
            PH,
            Θl,
            Θr;
            eigsolve_tol,
            eigsolve_krylovdim,
            eigsolve_maxiter,
            eigsolve_verbosity,
            eigsolve_which_eigenvalue,
        )

        normalize!(Θl)
        normalize!(Θr)
        ov = complex(inner(Θl, Θr))
        Θl /= conj(sqrt(ov))
        Θr /= sqrt(ov)
    end

    fx = inner(Θl, product(PH, Θr)) / inner(Θl, Θr)

    return fx, Θl, Θr
end

struct DoubleWoble
    x::ITensor
    ydag::ITensor
end

function VectorInterface.add(a::DoubleWoble, b::DoubleWoble)
    return DoubleWoble(add(a.x, b.x), add(a.ydag, b.ydag))
end
function VectorInterface.add!(a::DoubleWoble, b::DoubleWoble)
    VectorInterface.add!(a.x, b.x)
    VectorInterface.add!(a.ydag, b.ydag)
    return a
end
function VectorInterface.add!!(a::DoubleWoble, b::DoubleWoble)
    VectorInterface.add!(a.x, b.x)
    VectorInterface.add!(a.ydag, b.ydag)
    return a
end

function VectorInterface.add(a::DoubleWoble, b::DoubleWoble, α::Number)
    return DoubleWoble(VectorInterface.add(a.x, b.x, α), VectorInterface.add(a.ydag, b.ydag, α))
end
function VectorInterface.add!(a::DoubleWoble, b::DoubleWoble, α::Number)
    VectorInterface.add!(a.x, b.x, α)
    VectorInterface.add!(a.ydag, b.ydag, α)
    return a
end
function VectorInterface.add!!(a::DoubleWoble, b::DoubleWoble, α::Number)
    VectorInterface.add!(a.x, b.x, α)
    VectorInterface.add!(a.ydag, b.ydag, α)
    return a
end

function VectorInterface.add(a::DoubleWoble, b::DoubleWoble, α::Number, β::Number)
    return DoubleWoble(VectorInterface.add(a.x, b.x, α, β), VectorInterface.add(a.ydag, b.ydag, α, β))
end
function VectorInterface.add!(a::DoubleWoble, b::DoubleWoble, α::Number, β::Number)
    VectorInterface.add!(a.x, b.x, α, β)
    VectorInterface.add!(a.ydag, b.ydag, α, β)
    return a
end
function VectorInterface.add!!(a::DoubleWoble, b::DoubleWoble, α::Number, β::Number)
    VectorInterface.add!(a.x, b.x, α, β)
    VectorInterface.add!(a.ydag, b.ydag, α, β)
    return a
end

function VectorInterface.zerovector(a::DoubleWoble, type::Type{<:Number})
    return DoubleWoble(VectorInterface.zerovector(a.x, type), VectorInterface.zerovector(a.ydag, type))
end
function VectorInterface.zerovector!(a::DoubleWoble)
    VectorInterface.zerovector!(a.x)
    VectorInterface.zerovector!(a.ydag)
    return a
end
function VectorInterface.zerovector!!(a::DoubleWoble, type::Type{<:Number})
    VectorInterface.zerovector!!(a.x, type)
    VectorInterface.zerovector!!(a.ydag, type)
    return a
end

function VectorInterface.scalartype(a::DoubleWoble)
  return ITensors.scalartype(a.x)
end

VectorInterface.inner(a::DoubleWoble, b::DoubleWoble) = VectorInterface.inner(a.x, b.x) + VectorInterface.inner(a.ydag, b.ydag)
function VectorInterface.scale(a::DoubleWoble, x::Number)
    return DoubleWoble(VectorInterface.scale(a.x, x), VectorInterface.scale(a.ydag, x))
end
function VectorInterface.scale!(a::DoubleWoble, x::Number)
    VectorInterface.scale!(a.x, x)
    VectorInterface.scale!(a.ydag, x)
    return a
end
VectorInterface.scale!!(a::DoubleWoble, x::Number) = VectorInterface.scale!(a::DoubleWoble, x::Number)
function VectorInterface.scale!!(a::DoubleWoble, b::DoubleWoble, x::Number) 
    VectorInterface.scale!!(a.x, b.x, x)
    VectorInterface.scale!!(a.ydag, b.ydag, x)
    return a
end
LinearAlgebra.norm(x::DoubleWoble) = sqrt(norm(x.x)^2 + norm(x.ydag)^2)

# function nhproblemsolver!(
#     ::Algorithm"stabilized",
#     PHt,
#     Θl,
#     Θr;
#     eigsolve_tol,
#     eigsolve_krylovdim,
#     eigsolve_maxiter,
#     eigsolve_verbosity,
#     eigsolve_which_eigenvalue,
#     α = 0.01,
#     η = 20000,
#     PPPP
# )
#     PH, factor = PHt
#     PAHA, PAAH, PAl, PAr = PPPP
    
#     λ = 0.22400036958233754
#     # λ = 0.6248545073721898

#     # X = A - λ * I 
    
#     # Q = Adjoint(X) * X
#     #   Q = (Adjoint(A) - conj(λ)) * (A - λ) = Adjoint(A) * A - conj(λ) * A - λ * Adjoint(A) + |λ|^2
#     #   QT = AT * conj(A) - conj(λ) * AT - λ * conj(A) + |λ|^2
#     # Q̃ = X * Adjoint(X)
#     #   Q̃ = (A - λ) * (Adjoint(A) - conj(λ)) = A * Adjoint(A) - conj(λ) * A - λ * Adjoint(A) + |λ|^2
#     #   Q̃T = conj(A) * AT - conj(λ) * AT - λ * conj(A) + |λ|^2

#     # Need: AT*x = conj(A^†*conj(x)) 

#     # QT = α * transpose(Q)
#     # Q̃T = α * transpose(Q̃)

    

#     A(x, OP) = product(OP, x)
#     Adag(x, OP) = adjointproduct(OP, x)
#     AT(x, OP) = conj(Adag(conj(x), OP))
#     ATdag(x, OP) = conj(A(conj(x), OP))

#     XT(x, OP) = -conj(λ) * AT(x, OP) - λ * ATdag(x, OP) + abs2(λ) * x

#     QT(x) = α * (conj(product(PAHA, conj(x))) + XT(x, PAr))
#     Q̃T(x) = α * (conj(product(PAAH, conj(x))) + XT(x, PAl))

#     QTdag(x) = QT(x)
#     Q̃Tdag(x) = Q̃T(x)

#     # @show inds(Θl)
#     # @show inds(Θr)

#     # @show inner(Θl, A(Θr)) / inner(Θl, Θr)

#     # normalize!(Θl)
#     # normalize!(Θr)
#     # @show norm(Θl)
#     # @show norm(Θr)
#     # @show norm(A(Θr) - λ * Θr)
#     # @show norm(AT(Θr) - λ * Θr)
#     # # @show norm(A(Θl) - λ * Θl)
#     # @show norm(AT(Θl) - λ * Θl)
#     # @show norm(2Q̃T(Θl) + A(Θr, PH))
#     # @show norm(AT(Θl, PH) + 2QT(Θr))
#     # @show norm(AT(Θl) + 2Q̃T(Θr))
#     # @show norm(2QT(Θl) + A(Θr))

#     vals, lvecs, rvecs, info = svdsolve(DoubleWoble(Θl, Θr), 1, :SR; krylovdim=25, maxiter=2, tol=1e-4) do x, flag
#         if flag === Val(true)
#             # y = compute action of adjoint map on x
#             l, r = x.x, x.ydag
#             y = DoubleWoble(
#                 factor * AT(l, PH) + 2QT(r),
#                 2Q̃T(l) + factor * A(r, PH)
#                 )
#                 #     nΘl = Θl
#                 # else
#                 #     nΘr = Θr
#                 # end
#                 # y = AT(l, PH) + 2QT(r)
#             else
#                 # y = compute action of linear map on x
#                 l, r = x.x, x.ydag
#                 y = DoubleWoble(
#                     factor * ATdag(l, PH) + 2Q̃Tdag(r),
#                     2QTdag(l) + factor * Adag(r, PH)
#                 )
#             # y = DoubleWoble(
#             #     ATdag(x, PH),
#             #     2QTdag(x)
#             # )
#         end
#         return y
#     end

#     Θl, Θr = lvecs[1].x, lvecs[1].ydag

#     return factor * inner(Θl, A(Θr, PH)), Θl, Θr
# end


struct EmbeddedITensor
    x::ITensor
    L::ITensor 
    R::ITensor
    isdag::Bool
end

function VectorInterface.add(a::EmbeddedITensor, b::EmbeddedITensor)
    @assert hassameinds(a.x, b.x)
    return EmbeddedITensor(add(a.x, b.x), a.L, a.R, a.isdag)
end
function VectorInterface.add!(a::EmbeddedITensor, b::EmbeddedITensor)
    @assert hassameinds(a.x, b.x)
    VectorInterface.add!(a.x, b.x)
    return a
end
function VectorInterface.add!!(a::EmbeddedITensor, b::EmbeddedITensor)
    @assert hassameinds(a.x, b.x)
    VectorInterface.add!(a.x, b.x)
    return a
end

function VectorInterface.add(a::EmbeddedITensor, b::EmbeddedITensor, α::Number)
    @assert hassameinds(a.x, b.x)
    return EmbeddedITensor(VectorInterface.add(a.x, b.x, α), a.L, a.R, a.isdag)
end
function VectorInterface.add!(a::EmbeddedITensor, b::EmbeddedITensor, α::Number)
    @assert hassameinds(a.x, b.x)
    VectorInterface.add!(a.x, b.x, α)
    return a
end
function VectorInterface.add!!(a::EmbeddedITensor, b::EmbeddedITensor, α::Number)
    @assert hassameinds(a.x, b.x)
    VectorInterface.add!(a.x, b.x, α)
    return a
end

function VectorInterface.add(a::EmbeddedITensor, b::EmbeddedITensor, α::Number, β::Number)
    @assert hassameinds(a.x, b.x)
    return EmbeddedITensor(VectorInterface.add(a.x, b.x, α, β), a.L, a.R, a.isdag)
end
function VectorInterface.add!(a::EmbeddedITensor, b::DoubleWoble, α::Number, β::Number)
    VectorInterface.add!(a.x, b.x, α, β)
    return a
end
function VectorInterface.add!!(a::EmbeddedITensor, b::DoubleWoble, α::Number, β::Number)
    VectorInterface.add!(a.x, b.x, α, β)
    return a
end

function VectorInterface.zerovector(a::EmbeddedITensor, type::Type{<:Number})
    return EmbeddedITensor(VectorInterface.zerovector(a.x, type), a.L, a.R, a.isdag)
end
function VectorInterface.zerovector!(a::EmbeddedITensor)
    VectorInterface.zerovector!(a.x)
    return a
end
function VectorInterface.zerovector!!(a::EmbeddedITensor, type::Type{<:Number})
    VectorInterface.zerovector!!(a.x, type)
    return a
end

function VectorInterface.scalartype(a::EmbeddedITensor)
  return ITensors.scalartype(a.x)
end

function VectorInterface.inner(a::EmbeddedITensor, b::EmbeddedITensor) 
    m, n = if a.isdag 
        dag(a.x), prime(b.x, "Link")
    else
        prime(a.x, "Link"), dag(b.x)
    end

    O = a.L * m 
    O *= n 
    O *= a.R 
    return O[]
end 

function VectorInterface.scale(a::EmbeddedITensor, x::Number)
    return EmbeddedITensor(VectorInterface.scale(a.x, x), a.L, a.R, a.isdag)
end
function VectorInterface.scale!(a::EmbeddedITensor, x::Number)
    VectorInterface.scale!(a.x, x)
    return a
end
VectorInterface.scale!!(a::EmbeddedITensor, x::Number) = VectorInterface.scale!(a, x)
function VectorInterface.scale!!(a::EmbeddedITensor, b::EmbeddedITensor, x::Number) 
    VectorInterface.scale!!(a.x, b.x, x)
    return a
end
function LinearAlgebra.norm(x::EmbeddedITensor) 
    return norm(x.x)
end 

function nhproblemsolver!(
    ::Algorithm"stabilized",
    PHt,
    Θl,
    Θr;
    eigsolve_tol,
    eigsolve_krylovdim,
    eigsolve_maxiter,
    eigsolve_verbosity,
    eigsolve_which_eigenvalue,
    L, R,
    α = 0.01,
    η = 20000,
    PPPP
)
    PH, factor = PHt
    PAHA, PAAH, PAl, PAr = PPPP

    # X = A - λ * I 
    
    # Q = Adjoint(X) * X
    #   Q = (Adjoint(A) - conj(λ)) * (A - λ) = Adjoint(A) * A - conj(λ) * A - λ * Adjoint(A) + |λ|^2
    #   QT = AT * conj(A) - conj(λ) * AT - λ * conj(A) + |λ|^2
    # Q̃ = X * Adjoint(X)
    #   Q̃ = (A - λ) * (Adjoint(A) - conj(λ)) = A * Adjoint(A) - conj(λ) * A - λ * Adjoint(A) + |λ|^2
    #   Q̃T = conj(A) * AT - conj(λ) * AT - λ * conj(A) + |λ|^2

    # Need: AT*x = conj(A^†*conj(x)) 

    # QT = α * transpose(Q)
    # Q̃T = α * transpose(Q̃)

    A(x, OP) = product(OP, x)
    Adag(x, OP) = adjointproduct(OP, x)
    AT(x, OP) = conj(Adag(conj(x), OP))
    ATdag(x, OP) = conj(A(conj(x), OP))

    @info "Hello there, General"

    fA = a::ITensor -> product(PH, a)::ITensor
    fAH = a::ITensor -> adjointproduct(PH, a)::ITensor
    f = (fA, fAH)

    vals, (V, W), info = bieigsolve(
        f,
        Θr,
        Θl,
        1,
        eigsolve_which_eigenvalue,
        BiArnoldi(;
            tol=1e-10,
            krylovdim=5,
            maxiter=2,
            verbosity=eigsolve_verbosity,
        ),
    )

    @show λ = first(vals)
    Θl = first(W)
    Θr = first(V)

    X(x, OP) = -conj(λ) * A(x, OP) - λ * Adag(x, OP) + abs2(λ) * x

    Q(x) = product(PAHA, x) + X(x, PAr)
    Q̃(x) = product(PAAH, x) + X(x, PAl)

    Qdag(x) = Q(x)
    Q̃dag(x) = Q̃(x)

    valsl, lvecsl, infol = eigsolve(Q̃, Θl, 1, :SR; krylovdim=5, maxiter=2, tol=1e-10, verbosity=-1, ishermitian=true)
    valsr, lvecsr, infor = eigsolve(Q, Θr, 1, :SR; krylovdim=5, maxiter=2, tol=1e-10, verbosity=-1, ishermitian=true)

    # valsl, lvecsl, rvecsl, infol = svdsolve((Q̃dag, Q̃), Θl, 1, :SR; krylovdim=40, maxiter=2, tol=1e-10, verbosity=-1)
    # valsr, lvecsr, rvecsr, infor = svdsolve((Qdag, Q), Θr, 1, :SR; krylovdim=40, maxiter=2, tol=1e-10, verbosity=-1)
    
    # @show valsl[1], valsr[1]
    
    Θl, Θr = lvecsl[1], lvecsr[1]
    
    # @show inner(Θl, A(Θr, PH)) / (dag(Θl) * L * prime(Θr, "Link") * R)[]

    return inner(Θl, A(Θr, PH)), Θl, Θr
end

# function nhproblemsolver!(
#     ::Algorithm"stabilized",
#     PH,
#     Θl,
#     Θr;
#     eigsolve_tol,
#     eigsolve_krylovdim,
#     eigsolve_maxiter,
#     eigsolve_verbosity,
#     eigsolve_which_eigenvalue,
#     α = 0.01,
#     η = 20000,
#     PPPP
# )
#     PAHA, PAAH, PAl, PAr = PPPP
    
#     λ = 0.6248545073721898

#     # X = A - λ * I 
    
#     # Q = Adjoint(X) * X
#     #   Q = (Adjoint(A) - conj(λ)) * (A - λ) = Adjoint(A) * A - conj(λ) * A - λ * Adjoint(A) + |λ|^2
#     #   QT = AT * conj(A) - conj(λ) * AT - λ * conj(A) + |λ|^2
#     # Q̃ = X * Adjoint(X)
#     #   Q̃ = (A - λ) * (Adjoint(A) - conj(λ)) = A * Adjoint(A) - conj(λ) * A - λ * Adjoint(A) + |λ|^2
#     #   Q̃T = conj(A) * AT - conj(λ) * AT - λ * conj(A) + |λ|^2

#     # Need: AT*x = conj(A^†*conj(x)) 

#     # QT = α * transpose(Q)
#     # Q̃T = α * transpose(Q̃)

#     A(x, OP) = product(OP, x)
#     Adag(x, OP) = adjointproduct(OP, x)
#     AT(x, OP) = conj(Adag(conj(x), OP))
#     ATdag(x, OP) = conj(A(conj(x), OP))

#     XT(x, OP) = -conj(λ) * AT(x, OP) - λ * ATdag(x, OP) + abs2(λ) * x
#     X(x, OP) = -conj(λ) * A(x, OP) - λ * Adag(x, OP) + abs2(λ) * x

#     QT(x) = α * (conj(product(PAHA, conj(x))) + XT(x, PAr))
#     Q̃T(x) = α * (conj(product(PAAH, conj(x))) + XT(x, PAl))
#     Q̃(x) = α * (product(PAAH, x) + X(x, PAl))
    
#     QTdag(x) = QT(x)
#     Q̃Tdag(x) = Q̃T(x)

#     # @show inner(Θl, A(Θr)) / inner(Θl, Θr)

#     κ = 1000
#     β = 1000


#     # normalize!(Θl)
#     # normalize!(Θr)
#     # @show norm(Θl)
#     # @show norm(Θr)
#     # @show norm(A(Θr) - λ * Θr)
#     # @show norm(AT(Θr) - λ * Θr)
#     # # @show norm(A(Θl) - λ * Θl)
#     # @show norm(AT(Θl) - λ * Θl)
#     # @show norm(2Q̃T(Θl) + A(Θr, PH))
#     # @show norm(AT(Θl, PH) + 2QT(Θr))
#     # @show norm(AT(Θl) + 2Q̃T(Θr))
#     # @show norm(2QT(Θl) + A(Θr))

#     function freal(x)
#         l, r = x.x, x.ydag 

#         return DoubleWoble(
#             2Q̃(l) + A(r, PH),
#             AT(l, PH) + 2QT(r),
#         )
#     end

#     function fimag(x)
#         l, r = x.x, x.ydag 

#         return DoubleWoble(
#             2Q̃(l) + A(r, PH),
#             -AT(l, PH) + 2QT(r),
#         )
#     end

    
#     valsr, vecsr, infor = eigsolve(
#         freal, DoubleWoble(real(Θl), real(Θr)), 1, EigSorter(x -> abs(x - λ); rev = false),; krylovdim=20, maxiter=5
#     )
#     @show valsr[1]
#     if !iszero(norm(imag(Θr))) && !iszero(norm(imag(Θl)))
#         valsi, vecsi, infoi = eigsolve(
#             fimag, DoubleWoble(imag(Θl), imag(Θr)), 1, :SR; krylovdim=20, maxiter=5
#         )  
        
#         Θl, Θr = vecsr[1].x + 1im*vecsi[1].x, vecsr[1].ydag + 1im * vecsr[1].imag
#     else
#         Θl, Θr = vecsr[1].x, vecsr[1].ydag
#     end

#     return inner(Θl, A(Θr, PH)), Θl, Θr
# end

# function nhproblemsolver!(
#     ::Algorithm"stabilized",
#     PH,
#     Θl,
#     Θr;
#     eigsolve_tol,
#     eigsolve_krylovdim,
#     eigsolve_maxiter,
#     eigsolve_verbosity,
#     eigsolve_which_eigenvalue,
#     α = 0.01,
#     η = 20000,
#     PPPP
# )
#     PAHA, PAAH = PPPP
    
#     ψ = Θr
#     ϕ = Θl

#     λ = 0.6248545073721882

#     # X = A - λ * I 
    
#     # Q = Adjoint(X) * X
#     #   Q = (Adjoint(A) - conj(λ)) * (A - λ) = Adjoint(A) * A - conj(λ) * A - λ * Adjoint(A) + |λ|^2
#     #   QT = AT * conj(A) - conj(λ) * AT - λ * conj(A) + |λ|^2
#     # Q̃ = X * Adjoint(X)
#     #   Q̃ = (A - λ) * (Adjoint(A) - conj(λ)) = A * Adjoint(A) - conj(λ) * A - λ * Adjoint(A) + |λ|^2
#     #   Q̃T = conj(A) * AT - conj(λ) * AT - λ * conj(A) + |λ|^2

#     # Need: AT*x = conj(A^†*conj(x)) 

#     # QT = α * transpose(Q)
#     # Q̃T = α * transpose(Q̃)

#     A(x) = product(PH, x)
#     Adag(x) = adjointproduct(PH, x)
#     AT(x) = conj(Adag(conj(x)))
#     ATdag(x) = conj(A(conj(x)))

#     XT(x) = -conj(λ) * AT(x) - λ * ATdag(x) + abs2(λ) * x

#     QT(x) = 2α * (conj(product(PAHA, conj(x))) + XT(x))
#     Q̃T(x) = 2α * (conj(product(PAAH, conj(x))) + XT(x))

#     QTdag(x) = QT(x)
#     Q̃Tdag(x) = Q̃T(x)

#     κ = 1000
#     β = 1000

#         function lossgrad(ψϕ)
#             r, l = ψϕ

#             # r1 = AT * φ + 2 * QT * ψ
#             r1 = AT(l) + QT(r)

#             # r2 = A * conj(ψ) + 2 * Q̃T * φ
#             r2 = A(r) + Q̃T(l)

#             o = 1 - inner(l, r)
#             d = real(inner(l, l) - inner(r, r))

#             loss = norm(r1)^2 + norm(r2)^2 #+
#                 # κ * o * conj(o) + 
#                 # β * d * conj(d)
#                 # κ * (1 - norm(r)^2)^2 + κ * (1 - norm(l)^2)^2

#             grad_r = 2 * QTdag(r1) + 2 * Adag(r2) #- 2κ * conj(o) * r - 4β * d * l
#             # grad_ψ = 2 * QTdag(r1) + 2 * Adag(r2) - 4 * κ * (1 - norm(r)^2) * r
#             # grad_l = 2 * ATdag(r1) + 2 * Q̃Tdag(r2) - 4 * κ * (1 - norm(l)^2) * l
#             grad_l = 2 * ATdag(r1) + 2 * Q̃Tdag(r2) #- 2κ * o * l + 4β * d * r

#             return loss, (grad_r, grad_l)
#         end


#     r, fr, gx, numfg, normgradhistory = optimize(lossgrad, (ψ, ϕ), LBFGS(5; gradtol=1e-5, maxiter=1000))

#     ψ, ϕ = r

#     # # normalize!(ψ)
#     # # normalize!(ϕ)

#     # # r1 = AT * φ + 2 * QT * ψ
#     # r1 = AT(ϕ) + QT(ψ)

#     # # r2 = A * conj(ψ) + 2 * Q̃T * φ
#     # r2 = A(conj(ψ)) + Q̃T(ϕ)

#     # # loss = norm(r1)^2 + norm(r2)^2 +
#     # #     κ * (1 - norm(ψ)^2)^2 + κ * (1 - norm(φ)^2)^2

#     # grad_ψ = 2 * QTdag(r1) + 2 * AT(conj(r2)) #- 4 * κ * (1 - norm(ψ)^2) * ψ
#     # grad_φ = 2 * ATdag(r1) + 2 * Q̃Tdag(r2) #- 4 * κ * (1 - norm(φ)^2) * φ 

#     # ψ -= stepsize * grad_ψ
#     # ϕ -= stepsize * grad_φ

#     return inner(ϕ, A(ψ)) / inner(ϕ, ψ), ϕ, ψ
# end