using Plots, ProgressMeter
using LinearAlgebra, Einsum, PaddedViews, BlockDiagonals
using ApproxFun, FastGaussQuadrature
import ClassicalOrthogonalPolynomials as cl
#using ClassicalOrthogonalPolynomials: HermiteWeight
#using ApproxFun.ApproxFunOrthogonalPolynomials: weight

m = 3 # number of conserved v components
r = 4 # rank
m_x, m_v = 256, 128 # points in x and v 

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
    S[k, k] += 1e-4    # for invertibility
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


∇ᵥlogf0v = ∇ᵥ(log.(f0v'))'

# step algorithm starts here
function step(X, S, V, τ)
        
    f = X * S * V' .* f0v'

    U = @view V[:, 1:m]
    W = @view V[:, m+1:r]
    b = @view S[:, m+1:r]

    # compute RHS
    ∇ₓX = ∇ₓ(X)
    ∇ᵥV = ∇ᵥ(V')'
    Ef = E(f)
    vV = v_grid .* V
    ∇ᵥf0vV = ∇ᵥ((f0v .* V)')'

    @einsum c1[k,j] := V[v,k] * v_weights[v] * f0v[v] * vV[v,j]
    @einsum c2[k,j] := V[v,k] * v_weights[v] * ∇ᵥf0vV[v,j]

    @einsum d1[k,j] := X[x,k] * x_weights[x] * (Ef[x] * X[x,j])
    @einsum d2[k,j] := X[x,k] * x_weights[x] * ∇ₓX[x,j]

    # update K and L
    K = X * S
    L = W * b'b

    @einsum ∂ₜS[k,l] := ( - (d2[k,i] ⋅ c1[l,j]) + (d1[k,i] ⋅ c2[l,j]) ) * S[i,j]
    @einsum ∂ₜK[x,k] := ( - (c1[k,j] ⋅ ∇ₓX[x,i]) + (c2[k,j] ⋅ Ef[x]) * X[x,i] ) * S[i,j]

    #=
    RHSf = RHS(f)
    @einsum ∂ₜS2[k,l] := X[x,k] * x_weights[x] * V[v,l] * v_weights[v] * RHSf[x,v]
    @einsum ∂ₜK2[x,k] := V[v,k] * v_weights[v] * RHSf[x,v]
    @einsum ∂ₜL2[v,q] := ( b[i,q] * (X[x,i] * x_weights[x] * RHSf[x,v]) / f0v[v] )  -  b[i,q] * ∂ₜS[i,l] * V[v,l]
    @einsum p1[v,q] := b[i,q] * (X[x,i] * x_weights[x] * RHSf[x,v]) / f0v[v]
    =#
    
    @einsum g1[v,i] := d1[i,k] ⋅ ( S[k,l] * ∇ᵥV[v,l] + ∇ᵥlogf0v[v] * S[k,l] * V[v,l] )
    @einsum g2[v,i] := (v_grid[v] ⋅ d2[i,k]) * S[k,l] * V[v,l]
    #@einsum ∂ₜL[v,q] := b[i,q] * (g1[v,i] - g2[v,i])  -  b[i,q] * ∂ₜS[i,l] * V[v,l]
    @einsum p2[v,q] := b[i,q] * (g1[v,i] - g2[v,i])
    @einsum p3[v,q] := b[i,q] * ∂ₜS[i,l] * V[v,l]
    
    ∂ₜL = p2 - p3

    K += τ * ∂ₜK
    L += τ * ∂ₜL

    # extend basis
    X̃ = [X;; ∇ₓX;; K]
    X̃, _R = gram_schmidt(X̃, sqrt_x_gram)

    Ṽ = [U;; L;; W]
    Ṽ, _R = gram_schmidt(Ṽ, sqrt_v_gram)

    W̃ = @view Ṽ[:, m+1:end]

    @einsum M[k,l] := x_weights[x] * X[x,k] * X̃[x,l]
    @einsum N[k,l] := v_weights[v] * f0v[v] * V[v,k] * Ṽ[v,l]

    S̃ = M' * S * N
    f̂ = X̃ * S̃ * Ṽ' .* f0v'

    # compute RHS with updated basis
    ∇ₓX = ∇ₓ(X̃)
    ∇ᵥV = ∇ᵥ(Ṽ')'
    Ef = E(f̂)
    vV = v_grid .* Ṽ
    ∇ᵥf0vV = ∇ᵥ((f0v .* Ṽ)')'

    @einsum c1[k,j] := Ṽ[v,k] * v_weights[v] * f0v[v] * vV[v,j]
    @einsum c2[k,j] := Ṽ[v,k] * v_weights[v] * ∇ᵥf0vV[v,j]

    @einsum d1[k,j] := X̃[x,k] * x_weights[x] * (Ef[x] * X̃[x,j])
    @einsum d2[k,j] := X̃[x,k] * x_weights[x] * ∇ₓX[x,j]

    @einsum ∂ₜS[k,l] :=  ( - (d2[k,i] ⋅ c1[l,j]) + (d1[k,i] ⋅ c2[l,j]) ) * S̃[i,j]

    # update S̃
    S̃ += τ * ∂ₜS

    # split extended K
    K̃ = X̃ * S̃

    K̃cons = @view K̃[:, 1:m]
    K̃rem = @view K̃[:, m+1:end]

    # orthonormalize parts of X
    Xcons, Scons = gram_schmidt(K̃cons, sqrt_x_gram)
    X̃rem, S̃rem = gram_schmidt(K̃rem, sqrt_x_gram)

    # truncate via svd
    svdSrem = svd(S̃rem)
    Û = svdSrem.U[:, 1:r-m]
    Ŝ = Diagonal(svdSrem.S[1:r-m])
    Ŵ = svdSrem.Vt[1:r-m, :]'

    Srem = Ŝ
    W = W̃ * Ŵ
    Xrem = X̃rem * Û
    X̂ = [Xcons;; Xrem]

    X, R = gram_schmidt(X̂, sqrt_x_gram)    # X update step
    V = [U;; W]     # V update step
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
        p3 = plot(x_grid, [X[:,1], X[:,2], X[:,3], X[:,4]])
        p4 = plot(v_grid, [V[:,1], V[:,2], V[:,3], V[:,4]])
        display(plot(p1, p2, p3, p4, layout=(2,2)))
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

    if t in 0:0.05:t_end
        p1 = plot_density(f, t=t)
        p2 = plot_density(RHS(f), title="RHS", t=t)
        p3 = plot(x_grid, [X[:,1], X[:,2], X[:,3], X[:,4]])
        p4 = plot(v_grid, [V[:,1], V[:,2], V[:,3], V[:,4]])
        display(plot(p1, p2, p3, p4, layout=(2,2)))
    end

end
