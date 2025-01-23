using Plots, ProgressMeter
using Statistics, LinearAlgebra, Einsum, PaddedViews, BlockDiagonals
using ApproxFun, FastGaussQuadrature
using LinearAlgebra: NoPivot, ColumnNorm
import ClassicalOrthogonalPolynomials as cl
#using ClassicalOrthogonalPolynomials: HermiteWeight
#using ApproxFun.ApproxFunOrthogonalPolynomials: weight

include("./plot.jl")



# parameters 

m = 3 # number of conserved v components
r = 6 # rank
m_x, m_v = 256, 128 # points in x and v 

τ = 1e-3    # time step
t_start = 0.
t_end = 30.
t_grid = t_start:τ:t_end

# must be centered around 0 for now
xlims = (-π, π)
vlims = (-3, 3)

include("./quadrature.jl")



# initial conditions

X0 = x_basis[:, 1:r]
V0 = v_basis[:, 1:r]

α = 0.5
S0 = zeros(r, r)
S0[1, 1] = 1 / ( sqrt(π) * maximum(X0[:, 1]) * maximum(V0[:, 1]) )
S0[3, 1] = -α / ( sqrt(π) * maximum(X0[:, 3]) * maximum(V0[:, 1]) )

#=
for k in 1:r
    S0[k, k] += 1e-4    # for invertibility
end
=#

S0 /= 2π   # normalized e⁻ density

f = X0 * S0 * V0' .* f0v'

@assert f ≈ @. (1 - α * cos(x_grid)) * f0v' / (2π*sqrt(π))

#@assert ∫dx(∫dv(f)) ≈ [1]



include("./rhs.jl")
include("./step.jl")


plot_density(f, t=0)
plot_density(RHS(f), title="RHS")



# time evolution

X = copy(X0)
S = copy(S0)
V = copy(V0)

f = X * S * V' .* f0v'

time_evolution = fill(t_start, 4)
#mass_evolution = fill(mass(f)..., 4)
#momentum_evolution = fill(momentum(f)..., 4)
#energy_evolution = fill(energy(f)..., 4)

t = last_t = t_start
done = false
prog = Progress(round(Int, t_end, RoundDown))

while !done

    global X, S, V, t, last_t, f, done
    X, S, V, t = try_step(X, S, V, t, τ)
    
    update!(
        prog, round(Int, t, RoundDown), 
        showvalues=[
            ("time", t), 
            ("mass", mass(f)...), 
            ("momentum", momentum(f)...), 
            ("energy", energy(f)...)
        ]
    )

    if t - last_t > 0.05
        f = X * S * V' .* f0v'
        plot_step(x_grid, v_grid, f, X, S, V, t)#, mass_evolution, momentum_evolution, energy_evolution)
        last_t = t
    end

    if t ≥ t_end
        done = true
    end

end

#plot([mass_evolution, momentum_evolution, energy_evolution], ylims=(-10, 10))
