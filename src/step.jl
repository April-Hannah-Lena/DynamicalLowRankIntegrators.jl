using LinearAlgebra
using Einsum
using FillArrays, BlockArrays

function BlockDiagonal(M1, M2)
    mortar(
                    (M1,                  Zeros(size(M1,1), size(M2,2))),
        (Zeros(size(M2,1), size(M1,2)),               M2)
    )
end

maybe_invsqrt(x, TOL) = x > TOL  ?  1/sqrt(x) : 1.


function step_∂ₜ(X, S, V)

    f = X * S * V' .* f0v'

    b = @view S[:, m+1:end]

    ∇ₓX = ∇ₓ(X)
    ∇ᵥV = ∇ᵥ(V')'
    Ef = E(f)

    c1 = V' * v_gram * (v_grid .* V)
    c2 = V' * v_gram * (-2 .* v_grid .* V  +  ∇ᵥV)

    d1 = X' * x_gram * (Ef .* X)
    d2 = X' * x_gram * ∇ₓX

    @vielsimd ∂ₜS[k,l] := ( - (d2[k,i] ⋅ c1[l,j]) + (d1[k,i] ⋅ c2[l,j]) ) * S[i,j]
    @vielsimd ∂ₜK[x,k] := ( - (c1[k,j] ⋅ ∇ₓX[x,i]) + (c2[k,j] ⋅ Ef[x]) * X[x,i] ) * S[i,j]
    
    @vielsimd g1[v,i] := d1[i,k] ⋅ ( S[k,l] * ∇ᵥV[v,l] - 2 .* v_grid[v] * S[k,l] * V[v,l] )
    @vielsimd g2[v,i] := (v_grid[v] ⋅ d2[i,k]) * S[k,l] * V[v,l]
    
    @vielsimd p2[v,q] := b[i,q] * (g1[v,i] - g2[v,i])
    @vielsimd p3[v,q] := b[i,q] * ∂ₜS[i,l] * V[v,l]

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
function step(X, S, V, τ, TOL, TOL_quadrature=max(100eps(), 1e-3TOL))
        
    #f = X * S * V' .* f0v'

    U = @view V[:, 1:m]
    W = @view V[:, m+1:end]
    b = @view S[:, m+1:end]

    K = X * S
    L = W * b'b

    # midpoint step
    ∂ₜK, _, ∂ₜL = step_∂ₜ(X, S, V)

    K += τ/2 * ∂ₜK
    L += τ/2 * ∂ₜL

    # extend basis
    X̃ = [X;; ∇ₓ(X);; K]
    X̃, _ = gram_schmidt(X̃, x_gram, x_basis, TOL_quadrature)

    Ṽ = [V;; L]
    Ṽ, _ = gram_schmidt(Ṽ, v_gram, v_basis, m, TOL_quadrature)

    M = X' * x_gram * X̃
    N = V' * v_gram * Ṽ
    
    S̃ = M' * S * N

    _, ∂ₜS, _ = step_∂ₜ(X̃, S̃, Ṽ)
    
    S̃ += τ/2 * ∂ₜS

    # full step
    ∂ₜK, _, ∂ₜL = step_∂ₜ(X̃, S̃, Ṽ)

    # ∂ₜK and ∂ₜL only used to enrich the basis, their magnitude is not needed
    norm∂ₜK = sum(∂ₜK.^2 .* x_weights, dims=1)
    norm∂ₜL = sum(∂ₜL.^2 .* f0v .* v_weights, dims=1)

    ∂ₜK .*= maybe_invsqrt.(norm∂ₜK, TOL)
    ∂ₜL .*= maybe_invsqrt.(norm∂ₜL, TOL)

    # extend basis
    X̃ = [X̃;; ∂ₜK]
    Ṽ = [Ṽ;; ∂ₜL]

    X̃, _ = gram_schmidt(X̃, x_gram, x_basis, TOL_quadrature, pivot=false)
    Ṽ, _ = gram_schmidt(Ṽ, v_gram, v_basis, m, TOL_quadrature, pivot=false)

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
    Xcons, Scons = gram_schmidt(K̃cons, x_gram, x_basis, TOL_quadrature)
    X̃rem, S̃rem = gram_schmidt(K̃rem, x_gram, x_basis, TOL_quadrature)
    
    W̃ = @view Ṽ[:, m+1:end]

    # truncate via svd
    svdSrem = svd(S̃rem)

    r_new = max(min(size(S̃rem)..., r_max), r_min)
    @debug "maximal new rank" r_new

    Û = svdSrem.U[:, 1:r_new]
    Ŝ = Diagonal(svdSrem.S[1:r_new])
    Ŵ = svdSrem.Vt[1:r_new, :]'

    r_new = max(min(r_new, sum(Ŝ.diag .> TOL)), r_min)# - 1
    @debug "new rank based on error when truncating singular values" r_new

    Srem = Ŝ
    W = W̃ * Ŵ
    Xrem = X̃rem * Û
    X̂ = [Xcons;; Xrem]

    _X, R = gram_schmidt(X̂, x_gram, x_basis, TOL_quadrature, pivot=false)    # X update step
    _V = [U;; W]     # V update step
    _S = R * BlockDiagonal(Scons, Srem)    # S update step

    #=
    f = _X * _S * _V' .* f0v'
    mas = mas_new = mass(f)[1]
    momen = momen_new = momentum(f)[1]
    el_energy = el_energy_new = electric_energy(f)[1]

    while @show ( 
            abs( el_energy_new - el_energy ) < TOL ||
            abs( mas_new - mas ) < TOL ||
            abs( momen_new - momen ) < TOL 
          ) && 
          r_new ≥ r_min
        
        X_new = @view _X[:, 1:r_new]
        S_new = @view _S[1:r_new, 1:r_new]
        V_new = @view _V[:, 1:r_new]

        f = X_new * S_new * V_new' .* f0v'
        mas_new, = mass(f)
        momen_new, = momentum(f)
        el_energy_new, = electric_energy(f)
        r_new -= 1
    end

    =#

    #r_new = r_new + 1

    X = _X[:, 1:r_new]
    S = _S[1:r_new, 1:r_new]
    V = _V[:, 1:r_new]

    #f = X * S * V' .* f0v'      # f update step

    return X, S, V

end

function try_step(X, S, V, t, τ, τ_min=1e-7, TOL=1e-12, TOL_conservation=1e-8)
    
    f = X * S * V' .* f0v'
    m, = mass(f)
    j, = momentum(f)
    e, = energy(f)
    s, = entropy(f)
    mini = minimum(f)
    Xmax = maximum(abs.(X))

    X_new, S_new, V_new = step(X, S, V, τ, TOL)
    f_new = X_new * S_new * V_new' .* f0v'

    m_new, = mass(f_new)
    j_new, = momentum(f_new)
    e_new, = energy(f_new)
    s_new, = entropy(f_new)
    mini_new = minimum(f_new)
    Xmax_new = maximum(abs.(X_new))

    if ( abs(m - m_new) > TOL_conservation || 
         abs(j - j_new) > TOL_conservation || 
         abs(e - e_new) > TOL_conservation ||
         abs(s_new - s) > 500TOL_conservation ||        # little bit more tolerance
         abs(mini_new - mini) > 500TOL_conservation ||  # since these aren't provably conserved
         abs(Xmax_new - Xmax) > 500TOL_conservation )

        if τ ≤ τ_min
            #@warn "state too unstable"
        else
            return try_step(X, S, V, t, τ / 2)
        end
    end

    return X_new, S_new, V_new, t + τ, τ
    
end
