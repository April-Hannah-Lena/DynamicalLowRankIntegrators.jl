using LinearAlgebra
using Einsum, PaddedViews, BlockDiagonals



∇ᵥlogf0v = -2 .* v_grid
∇ᵥf0v = ∇ᵥlogf0v .* f0v

function step_∂ₜ(X, S, V)

    f = X * S * V' .* f0v'

    #U = @view V[:, 1:m]
    #W = @view V[:, m+1:r]
    b = @view S[:, m+1:r]

    ∇ₓX = ∇ₓ(X)
    ∇ᵥV = ∇ᵥ(V')'
    Ef = E(f)
    #vV = v_grid .* V
    #∇ᵥf0vV = ∇ᵥf0v .* V  +  f0v .* ∇ᵥV
    #∇ᵥf0vV = ∇ᵥ((f0v .* V)')'

    #@vielsimd c1[k,j] := V[v,k] * v_weights[v] * f0v[v] * vV[v,j]
    #@vielsimd c2[k,j] := V[v,k] * v_weights[v] * ∇ᵥf0vV[v,j]

    c1 = V' * v_gram * (v_grid .* V)
    c2 = V' * v_gram * (∇ᵥlogf0v .* V  +  ∇ᵥV)

    #@vielsimd d1[k,j] := X[x,k] * x_weights[x] * (Ef[x] * X[x,j])
    #@vielsimd d2[k,j] := X[x,k] * x_weights[x] * ∇ₓX[x,j]

    d1 = X' * x_gram * (Ef .* X)
    d2 = X' * x_gram * ∇ₓX

    @vielsimd ∂ₜS[k,l] := ( - (d2[k,i] ⋅ c1[l,j]) + (d1[k,i] ⋅ c2[l,j]) ) * S[i,j]
    @vielsimd ∂ₜK[x,k] := ( - (c1[k,j] ⋅ ∇ₓX[x,i]) + (c2[k,j] ⋅ Ef[x]) * X[x,i] ) * S[i,j]
    
    @vielsimd g1[v,i] := d1[i,k] ⋅ ( S[k,l] * ∇ᵥV[v,l] + ∇ᵥlogf0v[v] * S[k,l] * V[v,l] )
    @vielsimd g2[v,i] := (v_grid[v] ⋅ d2[i,k]) * S[k,l] * V[v,l]
    #@einsum ∂ₜL[v,q] := b[i,q] * (g1[v,i] - g2[v,i])  -  b[i,q] * ∂ₜS[i,l] * V[v,l]
    @vielsimd p2[v,q] := b[i,q] * (g1[v,i] - g2[v,i])
    @vielsimd p3[v,q] := b[i,q] * ∂ₜS[i,l] * V[v,l]

    #=
    RHSf = RHS(f)
    @einsum ∂ₜS2[k,l] := X[x,k] * x_weights[x] * V[v,l] * v_weights[v] * RHSf[x,v]
    @einsum ∂ₜK2[x,k] := V[v,k] * v_weights[v] * RHSf[x,v]
    @einsum ∂ₜL2[v,q] := ( b[i,q] * (X[x,i] * x_weights[x] * RHSf[x,v]) / f0v[v] )  -  b[i,q] * ∂ₜS[i,l] * V[v,l]
    @einsum p1[v,q] := b[i,q] * (X[x,i] * x_weights[x] * RHSf[x,v]) / f0v[v]
    =#
    
    ∂ₜL = p2 - p3

    return ∂ₜK, ∂ₜS, ∂ₜL

end

#=
# Augmented BUG integrator
function step(X, S, V, τ)
        
    #f = X * S * V' .* f0v'

    U = @view V[:, 1:m]
    W = @view V[:, m+1:r]
    b = @view S[:, m+1:r]

    # update K and L
    K = X * S
    L = W * b'b

    ∂ₜK, _, ∂ₜL = step_∂ₜ(X, S, V)

    K += τ * ∂ₜK
    L += τ * ∂ₜL

    # extend basis
    X̃ = [X;; ∇ₓ(X);; K]
    X̃, _R = gram_schmidt(X̃, x_gram, x_basis)

    Ṽ = [V;; L]
    Ṽ, _R = gram_schmidt(Ṽ, v_gram, v_basis, m)

    W̃ = @view Ṽ[:, m+1:end]

    @vielsimd M[k,l] := x_weights[x] * X[x,k] * X̃[x,l]
    @vielsimd N[k,l] := v_weights[v] * f0v[v] * V[v,k] * Ṽ[v,l]

    S̃ = M' * S * N
    
    _, ∂ₜS, _ = step_∂ₜ(X̃, S̃, Ṽ)

    # update S̃
    S̃ += τ * ∂ₜS

    # split extended K
    K̃ = X̃ * S̃

    K̃cons = @view K̃[:, 1:m]
    K̃rem = @view K̃[:, m+1:end]

    # orthonormalize parts of X
    Xcons, Scons = gram_schmidt(K̃cons, x_gram, x_basis)
    X̃rem, S̃rem = gram_schmidt(K̃rem, x_gram, x_basis)

    # truncate via svd
    svdSrem = svd(S̃rem)
    Û = svdSrem.U[:, 1:r-m]
    Ŝ = Diagonal(svdSrem.S[1:r-m])
    Ŵ = svdSrem.Vt[1:r-m, :]'

    Srem = Ŝ
    W = W̃ * Ŵ
    Xrem = X̃rem * Û
    X̂ = [Xcons;; Xrem]

    X, R = gram_schmidt(X̂, x_gram, x_basis)    # X update step
    V = [U;; W]     # V update step
    S = R * BlockDiagonal([Scons, Srem])    # S update step

    #f = X * S * V' .* f0v'      # f update step

    return X, S, V

end
=#

# midpoint rule augmented BUG integrator
function step(X, S, V, τ)
        
    #f = X * S * V' .* f0v'

    U = @view V[:, 1:m]
    W = @view V[:, m+1:r]
    b = @view S[:, m+1:r]

    K = X * S
    L = W * b'b

    # midpoint step
    ∂ₜK, _, ∂ₜL = step_∂ₜ(X, S, V)

    K += τ/2 * ∂ₜK
    L += τ/2 * ∂ₜL

    # extend basis
    X̃ = [X;; ∇ₓ(X);; K]
    X̃, _ = gram_schmidt(X̃, x_gram, x_basis)

    Ṽ = [V;; L]
    Ṽ, _ = gram_schmidt(Ṽ, v_gram, v_basis, m)
    
    #@vielsimd M[k,l] := x_weights[x] * X[x,k] * X̃[x,l]
    #@vielsimd N[k,l] := v_weights[v] * f0v[v] * V[v,k] * Ṽ[v,l]

    M = X' * x_gram * X̃
    N = V' * v_gram * Ṽ
    
    S̃ = M' * S * N

    _, ∂ₜS, _ = step_∂ₜ(X̃, S̃, Ṽ)
    
    S̃ += τ/2 * ∂ₜS

    # full step
    ∂ₜK, _, ∂ₜL = step_∂ₜ(X̃, S̃, Ṽ)

    # extend basis
    X̃ = [X̃;; τ*∂ₜK]
    Ṽ = [Ṽ;; τ*∂ₜL]

    X̃, _ = gram_schmidt(X̃, x_gram, x_basis)
    Ṽ, _ = gram_schmidt(Ṽ, v_gram, v_basis, m)

    #@vielsimd M[k,l] := x_weights[x] * X[x,k] * X̃[x,l]
    #@vielsimd N[k,l] := v_weights[v] * f0v[v] * V[v,k] * Ṽ[v,l]

    M = X' * x_gram * X̃
    N = V' * v_gram * Ṽ

    S̃ = M' * S * N

    _, ∂ₜS, _ = step_∂ₜ(X̃, S̃, Ṽ)
    
    S̃ += τ * ∂ₜS
    
    # split extended K
    K̃ = X̃ * S̃
    
    K̃cons = @view K̃[:, 1:m]
    K̃rem = @view K̃[:, m+1:end]
    
    # orthonormalize parts of X
    Xcons, Scons = gram_schmidt(K̃cons, x_gram, x_basis)
    X̃rem, S̃rem = gram_schmidt(K̃rem, x_gram, x_basis)
    
    W̃ = @view Ṽ[:, m+1:end]

    # truncate via svd
    svdSrem = svd(S̃rem)
    Û = svdSrem.U[:, 1:r-m]
    Ŝ = Diagonal(svdSrem.S[1:r-m])
    Ŵ = svdSrem.Vt[1:r-m, :]'

    Srem = Ŝ
    W = W̃ * Ŵ
    Xrem = X̃rem * Û
    X̂ = [Xcons;; Xrem]

    X, R = gram_schmidt(X̂, x_gram, x_basis)    # X update step
    V = [U;; W]     # V update step
    S = R * BlockDiagonal([Scons, Srem])    # S update step

    #f = X * S * V' .* f0v'      # f update step

    return X, S, V

end

function try_step(X, S, V, t, τ, τ_min=1e-5, TOL=sqrt(eps()))
    
    f = X * S * V' .* f0v'
    m, = mass(f)
    j, = momentum(f)
    e, = energy(f)
    L1, = Lp(f, 1)
    L2, = Lp(f, 2)
    h, = entropy(f)

    X_new, S_new, V_new = step(X, S, V, τ)
    f_new = X_new * S_new * V_new' .* f0v'

    m_new, = mass(f_new)
    j_new, = momentum(f_new)
    e_new, = energy(f_new)
    L1_new, = Lp(f_new, 1)
    L2_new, = Lp(f_new, 2)
    h_new, = entropy(f_new)

    if ( abs(m - m_new) > TOL || 
         abs(j - j_new) > TOL || 
         abs(e - e_new) > TOL || 
         abs(L1 - L1_new) > sqrt(TOL) || 
         abs(L2 - L2_new) > sqrt(TOL) || 
         abs(h - h_new) > sqrt(TOL) )

        if τ ≤ τ_min
            @warn "state too unstable"
        else
            return try_step(X, S, V, t, τ / 2)
        end
    end

    return X_new, S_new, V_new, t + τ
    
end
