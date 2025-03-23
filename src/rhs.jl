using LinearAlgebra, Einsum, BandedMatrices
using ApproxFun



# rhs functions

∫dx(f) = vec(sum(f .* x_weights, dims=1))
∫dv(f) = vec(sum(f .* v_weights', dims=2))


const ∂_fourier = Tridiagonal(
    [iseven(k) ? k÷2 : 0 for k in 1:Mx-1],
    zeros(Mx),
    -[iseven(k) ? k÷2 : 0 for k in 1:Mx-1]
)
const ∂²_fourier = Diagonal(-[(iseven(k) ? k : k+1) / 2 for k in 0:Mx-1].^2)

const ∂_legendre = BandedMatrix(1 => [0.1k for k in 2:Mlegendre])
#∂_legendre .*= legendre_basis_norms
∂_legendre ./= legendre_basis_norms'

const ∂_hermite = BandedMatrix(1 => [2.0*k for k in 1:Mv-1])
∂_hermite .*= v_basis_norms    # ∂( Hⱼ / || Hⱼ|| ) = ( || Hⱼ₋₁ || / || Hⱼ || )  ∂Hⱼ / || Hⱼ₋₁ ||
∂_hermite ./= v_basis_norms'   #                   = ( || Hⱼ₋₁ || / || Hⱼ || )  2(j-1) Hⱼ₋₁ / || Hⱼ₋₁ ||



function ∇ₓ(f)  # 1-dimensional circle domain
    coeffs = x_basis' * x_gram * f
    return x_basis * ∂_fourier * coeffs
end

function ∇ᵥ(f)  # 1-dimensional real domain
    coeffs = legendre_basis' * v_gram_unweighted * f
    return ∂_legendre_basis * ∂_legendre * coeffs
end

function ∇ᵥ_hermite(f)  # 1-dimensional real domain
    coeffs = v_basis' * v_gram * f
    return v_basis * ∂_hermite * coeffs
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
RHS(f) = E(f) .* ∇ᵥ(f')'  -  v_grid' .* ∇ₓ(f)

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


x_norm(h) = sqrt(h' * x_gram * h)
v_norm(g) = sqrt(g' * v_gram * g)

function orthogonal_complement(g, basis, gram)
    projection = basis' * gram * g
    return g - basis * projection
end

function directional_continuity_error(fₜ₊, fₜ, τ, p)
    
    left = v_grid .^ p#orthogonal_complement(v_grid .^ p, v_basis[:, 1:m], v_gram)

    forward_diff = (fₜ₊ - fₜ) / τ
    rhs = RHS(fₜ)
    right = forward_diff - rhs

    x_dependent_error = right * v_gram_unweighted * left
    return x_weights' * x_dependent_error
end

function RHS_over_f0v(X, S, V)
    term1 = v_grid' .* ( ∇ₓ(X) * S * V' )
    term2 = ( -2 * v_grid' .* E((X * S * V') .* f0v') ) .* (X * S * V')  +  X * S * ∇ᵥ(V)'
    return - term1 - term2
end

function directional_continuity_error(Xₜ₊, Sₜ₊, Vₜ₊, Xₜ, Sₜ, Vₜ, τ, p)
    
    left = v_grid .^ p#orthogonal_complement(v_grid.^p, v_basis[:, 1:m], v_gram)

    forward_diff = ( Xₜ₊ * Sₜ₊ * Vₜ₊'  -  Xₜ * Sₜ * Vₜ' ) / τ
    rhs = RHS_over_f0v(Xₜ, Sₜ, Vₜ)
    right = forward_diff - rhs

    x_dependent_error = right * v_gram * left
    return x_dependent_error' * x_gram * x_dependent_error
end

function norm_continuity_error(Xₜ₊, Sₜ₊, Vₜ₊, Xₜ, Sₜ, Vₜ, τ, p)
    
    orth_complement = orthogonal_complement(v_grid .^ p, v_basis[:, 1:m], v_gram)
    norm_orth_complement = sqrt(orth_complement' * v_gram * orth_complement)

    forward_diff = ( Xₜ₊ * Sₜ₊ * Vₜ₊'  -  Xₜ * Sₜ * Vₜ' ) / τ
    rhs = RHS_over_f0v(Xₜ, Sₜ, Vₜ)

    x_dependent_error = sqrt.(diag( (forward_diff - rhs) * v_gram * (forward_diff - rhs)' ))
    return norm_orth_complement * ( x_weights' * x_dependent_error )
end
