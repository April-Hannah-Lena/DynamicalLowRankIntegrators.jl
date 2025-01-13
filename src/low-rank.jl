using Plots, ProgressMeter
using LinearAlgebra, Einsum, PaddedViews, BlockDiagonals
using ApproxFun, FastGaussQuadrature
import ClassicalOrthogonalPolynomials as cl
#using ClassicalOrthogonalPolynomials: HermiteWeight
#using ApproxFun.ApproxFunOrthogonalPolynomials: weight

m = 3 # number of conserved v components
r = 5 # rank
m_x, m_v = 256, 64 # points in x and v 

τ = 1e-2    # time step
t_start = 0.
t_end = 5.
t_grid = t_start:τ:t_end

# must be centered around 0 for now
xlims = (-π, π)
vlims = (-3, 3)

# set up quadrature

x_grid, x_weights = [-1:2/m_x:1-2/m_x;], ones(m_x)/m_x
v_grid, v_weights = gausslegendre(m_v)

x_grid .= (xlims[2]-xlims[1])/2 .* x_grid# .+ (xlims[2]+xlims[1])/2
v_grid .= (vlims[2]-vlims[1])/2 .* v_grid# .+ (vlims[2]+vlims[1])/2

x_weights .*= (xlims[2]-xlims[1])
v_weights .*= (vlims[2]-vlims[1])/2

@assert sum(x_weights) ≈ xlims[2] - xlims[1]
@assert sum(v_weights) ≈ vlims[2] - vlims[1]

f0v = @. exp(-v_grid^2)

x_gram = Diagonal(x_weights)
sqrt_x_gram = sqrt(x_gram)

v_gram = Diagonal(f0v .* v_weights)
sqrt_v_gram = sqrt(v_gram)

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

v_domain = Segment(vlims...)
legendrespace = Legendre(v_domain)

n_v = m_v ÷ 2  -  1
legendre_vandermonde = zeros(m_v, n_v)
for k in axes(legendre_vandermonde, 2)
    legendre_vandermonde[:, k] .= Fun(legendrespace, [zeros(k-1);1]).(v_grid)
    #legendre_vandermonde[:, k] ./= sqrt( 12 * 2 / (2*(k-1) + 1) )
end

# initial conditions

cfourier = cl.Fourier()
X = cfourier[x_grid, 1:r]
X[:, 3] .*= -1
X, _ = gram_schmidt(X, sqrt_x_gram)

chermite = cl.Hermite()
V = chermite[v_grid, 1:r]
V, _ = gram_schmidt(V, sqrt_v_gram)

α = 1e-1
S = zeros(r, r)
S[1, 1] = 1 / ( sqrt(π) * maximum(X[:, 1]) * maximum(V[:, 1]) )
S[3, 1] = α / ( sqrt(π) * maximum(X[:, 3]) * maximum(V[:, 1]) )

@assert X * S * V' .* f0v' ≈ @. (1 - α * cos(x_grid)) * f0v' / sqrt(π)

for k in 1:r
    S[k, k] += 1e-12    # for invertibility
end

∫dx(f) = vec(sum(f .* x_weights, dims=1))
∫dv(f) = vec(sum(f .* v_weights', dims=2))

S ./= 2π   # normalized e⁻ density

f = X * S * V' .* f0v'

#@assert ∫dx(∫dv(f)) ≈ [1]

function plot_density(f; title="e⁻ density", t=0)
    heatmap(
        x_grid, v_grid, f', 
        title="$title, time = $t", 
        xlabel="x",
        ylabel="v",
        xticks=([-π, -π/2, 0, π/2, π], ["-π", "-π/2", "0", "π/2", "π"])
    )
end

plot_density(f, t=0)


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
    ∂ = Derivative(legendrespace)
    
    ∇ᵥf = similar(f)
    #= @threads =# for (j, fj) in enumerate(eachrow(f))
        fit = Fun(legendrespace, legendre_vandermonde \ fj)
        ∂fit = ∂ * fit
        #= @simd =# for k in 1:size(f,2)
            ∇ᵥf[j, k] = ∂fit( v_grid[k] )
        end
    end

    return ∇ᵥf
end

function E(f)
    h = Fun(fourierspace, transform(fourierspace, 1/(2π) .- ∫dv(f)))

    ∇ = Derivative(fourierspace)
    Δ = ∇^2#Laplacian(fr)
    
    ϕ = -Δ \ h
    coefficients(ϕ)[1] = 0  # will be NaN if h is not periodic

    return (-∇*ϕ).(x_grid)
end

# 1 x dimension,  1 v dimension
RHS(f) = E(f) .* ∇ᵥ(f)  -  v_grid' .* ∇ₓ(f)

plot_density(RHS(f), title="RHS")

# step algorithm starts here
function step(X, S, V, τ)
        
    f = X * S * V' .* f0v'

    U = @view V[:, 1:m]
    W = @view V[:, m+1:r]

    ∂ₜf = RHS(f)

    K = X * S
    @einsum ∂ₜK[x,k] := v_weights[v] * V[v,k] * ∂ₜf[x,v]
    K = K  +  τ * ∂ₜK

    @einsum b1[k,v] := x_weights[x] * X[x,k] * ∂ₜf[x,v]
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
    @einsum N[k,l] := v_weights[v] * f0v[v] * V[v,k] * Ṽ[v,l]

    S̃ = M' * S * N
    f̂ = X̃ * S̃ * Ṽ' .* f0v'

    ∂ₜf̂ = RHS(f̂)

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

mass(f) = ∫dx(∫dv(f))
momentum(f) = ∫dx(∫dv(f .* v_grid'))
energy(f) = ∫dx( ∫dv( f .* (v_grid .^ 2)' )  +  E(f) .^ 2 ) / 2

mass_evolution = [mass(f)...]
momentum_evolution = [momentum(f)...]
energy_evolution = [energy(f)...]

@showprogress for t in t_grid[2:end]

    global X, S, V, f

    try
        X, S, V = step(X, S, V, τ)
    catch err
        p1 = plot_density(f, t=t)
        p2 = plot_density(RHS(f), title="RHS", t=t)
        p3 = plot(x_grid, [X[:,1], X[:,2], X[:,3], X[:,4], X[:,5]])
        p4 = plot(v_grid, [V[:,1], V[:,2], V[:,3], V[:,4], V[:,5]])
        display(plot(p1, p2))
        display(plot(p3, p4))
        rethrow(err)
    end

    f = X * S * V' .* f0v'

    push!(mass_evolution, mass(f)...)
    push!(momentum_evolution, momentum(f)...)
    push!(energy_evolution, energy(f)...)

    @info "step" time=t mass=mass_evolution[end] momentum=momentum_evolution[end] energy=energy_evolution[end]
    @info "step" S=S

    #=
    if all(f .> -1e-5)
    else
        @error "negative density"
        display(plot_density(f, t=t))
        break
    end
    =#

    if t in 0:0.1:t_end
        p1 = plot_density(f, t=t)
        p2 = plot_density(RHS(f), title="RHS", t=t)
        p3 = plot(x_grid, [X[:,1], X[:,2], X[:,3], X[:,4], X[:,5]])
        p4 = plot(v_grid, [V[:,1], V[:,2], V[:,3], V[:,4], V[:,5]])
        display(plot(p1, p2))
        display(plot(p3, p4))
    end

end
