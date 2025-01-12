using Plots, ProgressMeter
using LinearAlgebra, Einsum, PaddedViews, BlockDiagonals
using ApproxFun, FastGaussQuadrature
import ClassicalOrthogonalPolynomials as cl
#using ClassicalOrthogonalPolynomials: HermiteWeight
#using ApproxFun.ApproxFunOrthogonalPolynomials: weight

m = 3 # number of conserved v components
r = 5 # rank
m_x, m_v = 128, 16 # points in x and v 

τ = 1e-2    # time step
t_start = 0.
t_end = 10.
t_grid = t_start:τ:t_end

# must be centered around 0 for now
xlims = (-π, π)
#vlims = (-6, 6)

# set up quadrature

x_grid, x_weights = [-1:2/m_x:1-2/m_x;], ones(m_x)/m_x
v_grid, v_weights = gausshermite(m_v)

x_grid .= (xlims[2]-xlims[1])/2 .* x_grid
#v_grid .= (vlims[2]-vlims[1])/2 .* v_grid# .+ (vlims[2]+vlims[1])/2

x_weights .*= (xlims[2]-xlims[1])
#v_weights .*= (vlims[2]-vlims[1])/2

x_gram = Diagonal(x_weights)
sqrt_x_gram = sqrt(x_gram)

v_gram = Diagonal(v_weights)
sqrt_v_gram = sqrt(v_gram)

# bad 
sqrt_v_gram.diag[sqrt_v_gram.diag .< 10*eps()] .= 10*eps()

function gram_schmidt(f, sqrt_gram)
    QR = qr(sqrt_gram * f)
    Q = inv(sqrt_gram) * Matrix(QR.Q)
    R = QR.R
    E = Diagonal(sign.(diag(R)))   # want +1's on diagonal of R
    rmul!(Q, E); lmul!(E, R)
    return Q, R
end

x_domain = PeriodicSegment(xlims...)
fourierspace = Fourier(x_domain)
chebspace = Chebyshev(Segment(xlims...))

hermitespace = Hermite()

# initial conditions

hermweight = cl.HermiteWeight()   # == x -> exp(-x^2)
f0v = hermweight[v_grid]

cfourier = cl.Fourier()
X = cfourier[x_grid, 1:r]
#X[:, 3] .*= -1
#X, _ = gram_schmidt(X, sqrt_x_gram)    # normalize
X[:,1] ./= sqrt(2π)
X[:,2:end] ./= sqrt(π)


chermite = cl.Hermite()
V = chermite[v_grid, 1:r]
#V, _ = gram_schmidt(V, sqrt_v_gram)    # normalize
V ./= sqrt.( (2 .^ (0:r-1)) .* factorial.(0:r-1) .* sqrt(π) )'


α = 1e-1
S = zeros(r, r)
S[1, 1] = sqrt(2π) / sqrt(sqrt(π))
S[3, 1] = - α * sqrt(π) / sqrt(sqrt(π))

for k in 1:r
    S[k, k] += 1e-12    # for invertibility
end

∫dx(f) = vec(sum(f .* x_weights, dims=1))
∫dv(f) = vec(sum(f .* v_weights', dims=2))

f = X * S * V' .* f0v'

@assert ∫dx(∫dv(X * S * V')) ≈ [2π]   # e⁻ mass
@assert f ≈ @. (1 - α * cos(x_grid)) * exp(-(v_grid^2)') / sqrt(π)

heatmap(v_grid, x_grid, f, title="e⁻ density, time = 0")


# rhs functions

function ∇ₓ(f)  # 1-dimensional circle domain
    ∂ = Derivative(fourierspace)

    ∇ₓf = similar(f)
    #= @threads =# for (k, fk) in enumerate(eachcol(f))
        fit = Fun(fourierspace, transform(fourierspace, fk))
        ∂fit = ∂ * fit
        #= @simd =# for j in 1:size(f,1)
            ∇ₓf[j, k] = ∂fit( x_grid[j] )
        end
    end

    return ∇ₓf
end

function ∇ᵥ(f)  # 1-dimensional real domain
    ∂ = Derivative(hermitespace)
    
    ∇ᵥf = similar(f)
    #= @threads =# for (j, fj) in enumerate(eachrow(f))
        fit = Fun(hermitespace, transform(hermitespace, fj))
        ∂fit = ∂ * fit
        #= @simd =# for k in 1:size(f,2)
            ∇ᵥf[j, k] = ∂fit( v_grid[k] )
        end
    end

    return ∇ᵥf
end

function E(X, S, V)
    ∫fdv = ∫dv(X * S * V')
    m = ∫dx(∫fdv) / (2π)

    h = Fun(
        fourierspace, 
        transform(fourierspace, m .- ∫fdv)
    )

    ∇ = Derivative(fourierspace)
    Δ = ∇^2#Laplacian(fr)
    #B = periodic(dom, 0)
    
    ϕ = -Δ \ h
    coefficients(ϕ)[1] = 0
    #ϕ = \([B; Δ], [0.; h]; tolerance=1e-5)
    #ϕ = Δ \ h
    return (-∇*ϕ).(x_grid)
    #minusgradϕ = -∇*ϕ
    #plot(x_grid, [1 .- ∫dv(f), ϕ.(x_grid), minusgradϕ.(x_grid)])
end

# 1 x dimension,  1 v dimension
#RHS(f) = E(f) .* ∇ᵥ(f)  -  v_grid' .* ∇ₓ(f)

function RHS_unscaled(X, S, V)
    #f = X * S * V' .* f0v'
    ∇ₓX, ∇ᵥV, Ef = ∇ₓ(X), ∇ᵥ(V'), E(X, S, V)

    @einsum term1[x,v] := 2 * v_grid[v]^2 * X[x,i] * S[i,j] * V[v,j]
    @einsum term2[x,v] := X[x,i] * S[i,j] * v_grid[v] * ∇ᵥV[j,v]
    @einsum term3[x,v] := Ef[x] * ∇ₓX[x,i] * S[i,j] * V[v,j]

    return - term1 + term2 - term3
end

RHS(X, S, V) = RHS_unscaled(X, S, V) .* f0v'

# step algorithm starts here
function step(X, S, V, τ)
        
    f = X * S * V' .* f0v'

    U = @view V[:, 1:m]
    W = @view V[:, m+1:r]

    ∂ₜf = RHS_unscaled(X, S, V)

    K = X * S
    @einsum ∂ₜK[x,k] := v_weights[v] * V[v,k] * ∂ₜf[x,v]
    K = K  +  τ * ∂ₜK

    @einsum b1[k,v] := x_weights[x] * X[x,k] * ∂ₜf[x,v] * f0v[v]
    @einsum b2[k,l] := x_weights[x] * v_weights[v] * X[x,k] * V[v,l] * ∂ₜf[x,v]
    b3 = @view S[:, m+1:r]

    L = b3'b3 * W'
    ∂ₜL = b3' * ( b1 ./ f0v'  -  b2*V' )
    L = L  +  τ * ∂ₜL

    X̃ = [X;; ∇ₓ(X);; K]
    X̃, _ = gram_schmidt(X̃, sqrt_x_gram)

    Ṽ = [U;; L';; W]
    Ṽ, _ = gram_schmidt(Ṽ, sqrt_v_gram)

    W̃ = @view Ṽ[:, m+1:end]

    @einsum M[k,l] := x_weights[x] * X[x,k] * X̃[x,l]
    @einsum N[k,l] := v_weights[v] * V[v,k] * Ṽ[v,l]

    S̃ = M' * S * N
    f̂ = X̃ * S̃ * Ṽ' .* f0v'

    ∂ₜf̂ = RHS_unscaled(X̃, S̃, Ṽ)

    @einsum ∂ₜS̃[k,l] := x_weights[x] * v_weights[v] * X̃[x,k] * Ṽ[v,l] * ∂ₜf̂[x,v]
    S̃ = S̃  +  τ * ∂ₜS̃

    K̃ = X̃ * S̃

    K̃cons = @view K̃[:, 1:m]
    K̃rem = @view K̃[:, m+1:end]

    Xcons, Scons = gram_schmidt(K̃cons, sqrt_x_gram)
    X̃rem, S̃rem = gram_schmidt(K̃rem, sqrt_x_gram)

    svdSrem = svd(S̃rem)
    Û = svdSrem.U[:, 1:r-m]
    Ŝ = Diagonal(svdSrem.S[1:r-m])
    Ŵ = svdSrem.Vt[1:r-m, :]'

    Srem = Ŝ

    W = W̃ * Ŵ
    V = [U;; W]     # V update step

    Xrem = X̃rem * Û
    X̂ = [Xcons;; Xrem]

    X, R = gram_schmidt(X̂, sqrt_x_gram)    # X update step

    S = R * BlockDiagonal([Scons, Srem])    # S update step

    #f = X * S * V' .* f0v'      # f update step

    return X, S, V
end


# time evolution

mass(X, S, V) = ∫dx(∫dv(X * S * V'))
momentum(X, S, V) = ∫dx(∫dv(X * S * V' .* v_grid'))
energy(X, S, V) = ∫dx( ∫dv( X * S * V' .* (v_grid .^ 2)' )  +  E(X, S, V) .^ 2 ) / 2

mass_evolution = [mass(X, S, V)...]
momentum_evolution = [momentum(X, S, V)...]
energy_evolution = [energy(X, S, V)...]

@showprogress for t in t_grid[2:end]

    global X, S, V, f

    X, S, V = step(X, S, V, τ)
    #f = X * S * V' .* f0v'

    push!(mass_evolution, mass(X, S, V)...)
    push!(momentum_evolution, momentum(X, S, V)...)
    push!(energy_evolution, energy(X, S, V)...)

    if all(f .> -20eps())
    else
        @error "negative density"
    end

    if t in 1:t_end
        f = X * S * V' .* f0v'
        display(heatmap(v_grid, x_grid, f, title="e⁻ density, time = $t"))
    end

end
