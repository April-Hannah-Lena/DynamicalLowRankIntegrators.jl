using ProgressMeter#, Plots
using Serialization, DataFrames, CSV
using Statistics, LinearAlgebra, Einsum
using ApproxFun, FastGaussQuadrature, Interpolations
const transform = ApproxFun.transform
using LinearAlgebra: NoPivot, ColumnNorm
import ClassicalOrthogonalPolynomials as cl
#using ClassicalOrthogonalPolynomials: HermiteWeight
#using ApproxFun.ApproxFunOrthogonalPolynomials: weight

#include("./plot.jl")



# parameters 

const m = parse(Int, ARGS[2]) # number of conserved v components
const r = 10 # starting rank
const r_min = 10
const r_max = 10
@assert m ≤ r_min ≤ r ≤ r_max

const m_x, m_v = 512, 512 # points in x and v 

const τ = 5e-4    # time step
const t_start = parse(Float64, ARGS[3])
const t_end = parse(Float64, ARGS[4])
const t_grid = t_start:τ:t_end

# must be centered around 0 for now
const xlims = (-π, π)
const vlims = (-6, 6)

include("./quadrature.jl")



# initial conditions

const X0 = x_basis[:, 1:r]
const V0 = v_basis[:, 1:r]

const S0 = zeros(r, r)


const conditions = Symbol(ARGS[1])
@assert conditions ∈ [:landau, :twostream, :nonsymmetric]

if conditions == :landau
    # Landau damping
    α = 0.5
    S0[1, 1] = 1 / ( sqrt(π) * maximum(X0[:, 1]) * maximum(V0[:, 1]) )
    S0[3, 1] = -α / ( sqrt(π) * maximum(X0[:, 3]) * maximum(V0[:, 1]) )

elseif conditions == :twostream
    # two-stream instability
    α = 0.3
    v̄ = 2.6
    γ = exp(v̄^2) * ( v̄ .^ (0:r-1) + (-v̄) .^ (0:r-1) ) ./ (factorial.(0:r-1) .* 2 .^ (0:r-1) ) 
    S0[1, :] .= γ / ( sqrt(π) * maximum(X0[:, 1]) * maximum(V0[:, 1]) )
    S0[3, :] .= -α*γ / ( sqrt(π) * maximum(X0[:, 3]) * maximum(V0[:, 1]) )

elseif conditions == :nonsymmetric
    α = 0.3
    β = 0.2
    v̄ = 2.6
    
    γ1 = exp(v̄^2)  *  (v̄) .^ (0:r-1)  ./  (factorial.(0:r-1) .* 2 .^ (0:r-1) ) 
    S0[1, :] .= γ1 / ( 2 * sqrt(π) * maximum(X0[:, 1]) * maximum(V0[:, 1]) )
    S0[3, :] .= α * γ1 / ( 2 * sqrt(π) * maximum(X0[:, 3]) * maximum(V0[:, 1]) )

    γ2 = exp(v̄^2)  *  (-v̄) .^ (0:r-1)  ./  (factorial.(0:r-1) .* 2 .^ (0:r-1) ) 
    S0[1, :] .+= γ2 / ( 2 * sqrt(π) * maximum(X0[:, 1]) * maximum(V0[:, 1]) )
    S0[2, :] .= β * γ2 / ( 2 * sqrt(π) * maximum(X0[:, 3]) * maximum(V0[:, 1]) )
end


#=
for k in 1:r
    S0[k, k] += 1e-4    # for invertibility
end
=#

include("./rhs.jl")
include("./step.jl")

f = X0 * S0 * V0' .* f0v'
S0 ./= mass(f)   # normalized e⁻ density
#f = X0 * S0 * V0' .* f0v'


#plot_density(f, t=0)
#plot_density(RHS(f), title="RHS")



# time evolution

X = copy(X0)
S = copy(S0)
V = copy(V0)

X_last = copy(X)
S_last = copy(S)
V_last = copy(V)

f = X * S * V' .* f0v'

df = DataFrame(
    "time" => [t_start],
    "mass" => mass(f),
    "momentum" => momentum(f),
    "energy" => energy(f),
    #"temperature" => temperature(f),
    "heat flux" => heat_flux(f),
    "4th moment" => v_moment(f, 4),
    "5th moment" => v_moment(f, 5),
    "6th moment" => v_moment(f, 6),
    "7th moment" => v_moment(f, 7),
    ["moment $p continuity error" => directional_continuity_error(X, S, V, X, S, V, τ, p) for p in 0:7]...,
    ["moment $p norm error" => norm_continuity_error(X, S, V, X, S, V, τ, p) for p in 0:7]...,
    ["representability error in v^$p" => v_norm(orthogonal_complement(v_grid .^ p, V, v_gram)) for p in 0:7]...,
    "entropy" => entropy(f),
    "L1 norm" => Lp(f, 1),
    "L2 norm" => Lp(f, 2),
    "L3 norm" => Lp(f, 3),
    "Linfinity norm" => [maximum(abs.(f))],
    "minimum" => [minimum(f)], 
    "rank" => [size(X,2)]
)

record_step = 1e-2
t = last_t = last_t_low_res = t_start
done = false
#prog = Progress(round(Int, t_end, RoundDown))
prog = Progress(length(t_start:record_step:t_end)-1, showspeed=true)


# precompile
try_step(X, S, V, t, τ, 1e-5, 1e-12)

@info "run begin $conditions, $m"

while !done

    global X, S, V, f, t, last_t, last_t_low_res, done
    global X_last, S_last, V_last

    #X, S, V, t, τ_used = try_step(X, S, V, t, τ, 1e-7, 1e-11)    # adaptive step size
    X_last, S_last, V_last = copy(X), copy(S), copy(V)
    X, S, V = step(X, S, V, τ, 1e-9)      # static step size
    t += τ

    if t ≥ t_end
        done = true
        finish!(
            prog,
            showvalues=[zip(names(df), df[end,:])...]#; ("final step size", τ_used)]
        )
    end
    
    if t - last_t ≥ record_step   # used for the adaptive step alg
        
        f = X * S * V' .* f0v'

        push!(df, [
            t;
            mass(f)...;
            momentum(f)...;
            energy(f)...;
            #temperature(f)...;
            heat_flux(f)...;
            v_moment(f, 4)...;
            v_moment(f, 5)...;
            v_moment(f, 6)...;
            v_moment(f, 7)...;
            [directional_continuity_error(X, S, V, X_last, S_last, V_last, τ, p) for p in 0:7]...;
            [norm_continuity_error(X, S, V, X_last, S_last, V_last, τ, p) for p in 0:7]...;
            [v_norm(orthogonal_complement(v_grid .^ p, V, v_gram)) for p in 0:7]...;
            entropy(f)...;
            Lp(f, 1)...;
            Lp(f, 2)...;
            Lp(f, 3)...;
            maximum(abs.(f));
            minimum(f);
            size(X,2)
        ])

        next!(
            prog,
            showvalues=[zip(names(df)[1:4], df[end,1:4])...]#; ("current step size", τ_used)]
        )

        last_t = t

    end

    if t - last_t_low_res ≥ 0.5

        #plot_step(x_grid, v_grid, f, X, S, V, t)#, mass_evolution, momentum_evolution, energy_evolution)
        last_t_low_res = t
    
    end

end


CSV.write(
    "../data/evolution_$(string(conditions))_rank_min_$(r_min)_max_$(r_max)_step_" * replace("$τ", "." => "d") * ".csv", 
    df
)
