using LinearAlgebra, Einsum
using ApproxFun



# rhs functions

∫dx(f) = vec(sum(f .* x_weights, dims=1))
∫dv(f) = vec(sum(f .* v_weights', dims=2))

x_domain = PeriodicSegment(xlims...)
fourierspace = Fourier(x_domain)
chebspace = Chebyshev(Segment(xlims...))

const ∂_fourier = Tridiagonal(
    [iseven(k) ? k÷2 : 0 for k in 1:Mx-1],
    zeros(Mx),
    -[iseven(k) ? k÷2 : 0 for k in 1:Mx-1]
)
const ∂²_fourier = Diagonal(-[(iseven(k) ? k : k+1) / 2 for k in 0:Mx-1].^2)
const ∂_legendre = Bidiagonal(zeros(Mlegendre), [0.1k for k in 2:Mlegendre], :U)


function ∇ₓ(f)  # 1-dimensional circle domain
    coeffs = x_basis' * x_gram * f
    return x_basis * ∂_fourier * coeffs
end

function ∇ᵥ(f)  # 1-dimensional real domain
    coeffs = legendre_basis' * v_gram_unweighted * f'
    return (∂_legendre_basis * ∂_legendre * coeffs)'
end

function E(f)
    m, = mass(f)
    ∫dvf = vec(∫dv(f))
    coeffs = x_basis' * x_gram * (m .- ∫dvf)
    
    # we know coeffs[1] == 0 but julia still throws SingularException when using stock "\"
    coeffs = -∂²_fourier.diag .\ coeffs
    coeffs[1] = 0

    return x_basis * (-∂_fourier) * coeffs
end

# 1 x dimension,  1 v dimension
RHS(f) = E(f) .* ∇ᵥ(f)  -  v_grid' .* ∇ₓ(f)

mass(f) = ∫dx(∫dv(f))
particle_flux_density(f) = ∫dv(f .* v_grid')
momentum(f) = ∫dx(particle_flux_density(f))
momentum_flux_density(f) = ∫dv( f .* (v_grid' .- particle_flux_density(f)).^2 )
temperature(f) = ∫dx(momentum_flux_density(f))
heat_flux_density(f) = ∫dv( f .* (v_grid' .- particle_flux_density(f)).^3 )
heat_flux(f) = ∫dx(heat_flux_density(f))
kinetic_energy(f) = ∫dx( ∫dv( f .* (v_grid .^ 2)' ) ) / 2
electric_energy(f) = ∫dx( E(f) .^ 2 ) / 2
energy(f) = kinetic_energy(f) + electric_energy(f)
Lp(f, p) = ( ∫dx(∫dv( abs.(f) .^ p )) ) .^ (1/p)
entropy(f) = -∫dx(∫dv( f .* log.(max.(f, eps())) ))
v_moment(f, p) = ∫dx(∫dv( f .* (v_grid' .- particle_flux_density(f)).^p ))
