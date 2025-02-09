using Plots, ProgressMeter
using Serialization, DataFrames, CSV
using Statistics, LinearAlgebra, Einsum, PaddedViews, BlockDiagonals
using ApproxFun, FastGaussQuadrature
const transform = ApproxFun.transform
using LinearAlgebra: NoPivot, ColumnNorm
import ClassicalOrthogonalPolynomials as cl
#using ClassicalOrthogonalPolynomials: HermiteWeight
#using ApproxFun.ApproxFunOrthogonalPolynomials: weight

include("./plot.jl")



# parameters 

const m = 3 # number of conserved v components
const r = 6 # rank
const m_x, m_v = 200, 100 # points in x and v 

const τ = 1e-3    # time step
const t_start = 0.
const t_end = 20.
const t_grid = t_start:τ:t_end

# must be centered around 0 for now
const xlims = (-π, π)
const vlims = (-5, 5)

include("./quadrature.jl")



# initial conditions

const X0 = x_basis[:, 1:r]
const V0 = v_basis[:, 1:r]

const α = 0.5
const S0 = zeros(r, r)

# Landau damping
S0[1, 1] = 1 / ( sqrt(π) * maximum(X0[:, 1]) * maximum(V0[:, 1]) )
S0[3, 1] = -α / ( sqrt(π) * maximum(X0[:, 3]) * maximum(V0[:, 1]) )


# two-stream instability
#=
v̄ = sqrt(2.4^2 / 2)
γ = exp(-2 * v̄^2) * ( (2v̄) .^ (0:r-1) + (-2v̄) .^ (0:r-1) ) ./ (factorial.(0:r-1) .* 2 .^ (0:r-1) ) 
S0[1, :] .= γ / ( sqrt(π) * maximum(X0[:, 1]) * maximum(V0[:, 1]) )
S0[3, :] .= -α*γ / ( sqrt(π) * maximum(X0[:, 3]) * maximum(V0[:, 1]) )
=#


#=
for k in 1:r
    S0[k, k] += 1e-4    # for invertibility
end
=#

include("./rhs.jl")
include("./step.jl")

f = X0 * S0 * V0' .* f0v'
S0 ./= mass(f)   # normalized e⁻ density


plot_density(f, t=0)
plot_density(RHS(f), title="RHS")



# time evolution

X = copy(X0)
S = copy(S0)
V = copy(V0)

f = X * S * V' .* f0v'

time_evolution = [t_start]
mass_evolution = [mass(f)...]
momentum_evolution = [momentum(f)...]
energy_evolution = [energy(f)...]
L1_evolution = [Lp(f, 1)...]
L2_evolution = [Lp(f, 2)...]
entropy_evolution = [entropy(f)...]
minimum_evolution = [minimum(f)]

record_step = 0.01
t = last_t = last_t_low_res = t_start
done = false
#prog = Progress(round(Int, t_end, RoundDown))
prog = Progress(length(t_start:record_step:t_end)-1)

while !done

    global X, S, V, f, t, last_t, last_t_low_res, done

    X, S, V, t = try_step(X, S, V, t, τ)    # adaptive step size
    #X, S, V = step(X, S, V, τ)      # static step size
    #t += τ

    if t ≥ t_end
        done = true
    end
    
    if t - last_t ≥ record_step   # used for the adaptive step alg
        
        f = X * S * V' .* f0v'
        next!(
            prog,
            showvalues=[
                ("time", t), 
                ("mass", mass(f)...), 
                ("momentum", momentum(f)...), 
                ("energy", energy(f)...),
                ("L¹ norm", Lp(f, 1)...),
                ("L² norm", Lp(f, 2)...),
                ("entropy", entropy(f)...),
                ("minimum", minimum(f))
            ]
        )

        push!(time_evolution, t)
        push!(mass_evolution, mass(f)...)
        push!(momentum_evolution, momentum(f)...)
        push!(energy_evolution, energy(f)...)
        push!(L1_evolution, Lp(f, 1)...)
        push!(L2_evolution, Lp(f, 2)...)
        push!(entropy_evolution, entropy(f)...)
        push!(minimum_evolution, minimum(f))

        last_t = t

    end

    if t - last_t_low_res ≥ 0.05

        plot_step(x_grid, v_grid, f, X, S, V, t)#, mass_evolution, momentum_evolution, energy_evolution)
        last_t_low_res = t
    
    end

end


begin
    p1 = plot(time_evolution, mass_evolution, lab="mass", ylims=(0.95, 1.05))
    p2 = plot(time_evolution, momentum_evolution, lab="momentum", ylims=(-0.05, 0.05))
    p3 = plot(time_evolution, energy_evolution, lab="energy", ylims=(0.2, 0.3))
    p4 = plot(time_evolution, L1_evolution, lab="L¹ norm", ylims=(0.95, 1.05))
    p5 = plot(time_evolution, L2_evolution, lab="L² norm", ylims=(0.2, 0.3))
    p6 = plot(time_evolution, entropy_evolution, lab="entropy", ylims=(2.6,3.0))
    plot(p1, p2, p3, p4, p5, p6, layout=(2, 3))
end

df = DataFrame(
    "time" => time_evolution, 
    "mass" => mass_evolution, 
    "momentum" => momentum_evolution, 
    "energy" => energy_evolution,
    "L1 norm" => L1_evolution,
    "L2 norm" => L2_evolution,
    "entropy" => entropy_evolution,
    "minimum" => minimum_evolution
)

CSV.write("evolution_data.csv", df)
