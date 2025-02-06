using LinearAlgebra, Einsum
using ApproxFun



# rhs functions

∫dx(f) = vec(sum(f .* x_weights, dims=1))
∫dv(f) = vec(sum(f .* v_weights', dims=2))

x_domain = PeriodicSegment(xlims...)
fourierspace = Fourier(x_domain)
chebspace = Chebyshev(Segment(xlims...))

∂_fourier = Tridiagonal(
    [iseven(k) ? k÷2 : 0 for k in 1:Mx-1],
    zeros(Mx),
    -[iseven(k) ? k÷2 : 0 for k in 1:Mx-1]
)

v_domain = Segment(vlims...)
legendrespace = Legendre(v_domain)

n_v = m_v ÷ 2  -  1
legendre_vandermonde = zeros(m_v, n_v)
for k in axes(legendre_vandermonde, 2)
    legendre_vandermonde[:, k] .= Fun(legendrespace, [zeros(k-1);1]).(v_grid)
    #legendre_vandermonde[:, k] ./= sqrt( 12 * 2 / (2*(k-1) + 1) )
end

function ∇ₓ(f)  # 1-dimensional circle domain
    coeffs = x_basis' * x_gram * f
    return x_basis * ∂_fourier * coeffs
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

mass(f) = ∫dx(∫dv(f))
momentum(f) = ∫dx(∫dv(f .* v_grid'))
energy(f) = ∫dx( ∫dv( f .* (v_grid .^ 2)' )  +  E(f) .^ 2 ) / 2
Lp(f, p) = ( ∫dx(∫dv( abs.(f) .^ p )) ) .^ (1/p)
entropy(f) = -∫dx(∫dv( f .* log.(max.(f, eps())) ))
