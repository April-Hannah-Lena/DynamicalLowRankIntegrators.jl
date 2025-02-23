using #=Plots,=# ProgressMeter
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

const m = 3 # number of conserved v components
const r = 11 # starting rank
const r_min = 11
const r_max = 11
const m_x, m_v = 250, 350 # points in x and v 

const τ = 5e-4    # time step
const t_start = 0.
const t_end = 30.
const t_grid = t_start:τ:t_end

# must be centered around 0 for now
const xlims = (-π, π)
const vlims = (-6, 6)

include("./quadrature.jl")



# initial conditions

const X0 = x_basis[:, 1:r]
const V0 = v_basis[:, 1:r]

const α = 0.5
const S0 = zeros(r, r)


const landau = false
if landau
    # Landau damping
    S0[1, 1] = 1 / ( sqrt(π) * maximum(X0[:, 1]) * maximum(V0[:, 1]) )
    S0[3, 1] = -α / ( sqrt(π) * maximum(X0[:, 3]) * maximum(V0[:, 1]) )
else
    # two-stream instability
    v̄ = sqrt(2.4^2 / 2)
    γ = exp(-2 * v̄^2) * ( (2v̄) .^ (0:r-1) + (-2v̄) .^ (0:r-1) ) ./ (factorial.(0:r-1) .* 2 .^ (0:r-1) ) 
    S0[1, :] .= γ / ( sqrt(π) * maximum(X0[:, 1]) * maximum(V0[:, 1]) )
    S0[3, :] .= -α*γ / ( sqrt(π) * maximum(X0[:, 3]) * maximum(V0[:, 1]) )
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


#plot_density(f, t=0)
#plot_density(RHS(f), title="RHS")



# time evolution

X = copy(X0)
S = copy(S0)
V = copy(V0)

f = X * S * V' .* f0v'

df = DataFrame(
    "time" => [t_start],
    "mass" => mass(f),
    "momentum" => momentum(f),
    "energy" => energy(f),
    "entropy" => entropy(f),
    "temperature" => temperature(f),
    "heat flux" => heat_flux(f),
    "4th moment" => v_moment(f, 4),
    "5th moment" => v_moment(f, 5),
    "6th moment" => v_moment(f, 6),
    "7th moment" => v_moment(f, 7),
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
prog = Progress(length(t_start:record_step:t_end)-1)


# precompile
try_step(X, S, V, t, τ, 1e-5, 1e-12)

while !done

    global X, S, V, f, t, last_t, last_t_low_res, done

    #X, S, V, t, τ_used = try_step(X, S, V, t, τ, 1e-7, 1e-11)    # adaptive step size
    X, S, V = step(X, S, V, τ, 1e-10, 1e-13)      # static step size
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
            entropy(f)...;
            temperature(f)...;
            heat_flux(f)...;
            v_moment(f, 4)...;
            v_moment(f, 5)...;
            v_moment(f, 6)...;
            v_moment(f, 7)...;
            Lp(f, 1)...;
            Lp(f, 2)...;
            Lp(f, 3)...;
            maximum(abs.(f));
            minimum(f);
            size(X,2)
        ])

        next!(
            prog,
            showvalues=[zip(names(df), df[end,:])...]#; ("current step size", τ_used)]
        )

        last_t = t

    end

    if t - last_t_low_res ≥ 0.1

        #plot_step(x_grid, v_grid, f, X, S, V, t)#, mass_evolution, momentum_evolution, energy_evolution)
        last_t_low_res = t
    
    end

end

#=
pl = []

for col in names(df)[2:end-1]
    p = plot(
        df[!, "time"], df[!, col],
        ylims=(minimum(df[!,col]) - 1e-3, maximum(df[!,col]) + 1e-3),
        xlabel="time, step = $τ", ylabel=col
    )
    push!(pl, p)
    #savefig(p, "./data/$(landau ? "landau" : "two_stream")/evolution_$(col)_rank_$(r).pdf")
end


pl2 = []

for col in names(df)[2:end-1]
    p = plot(
        df[!, "time"], 
        abs.(df[!, col] .- df[1, col]) .+ 1e-30,
        yscale=:log10,
        ylims=(eps(), 1), yticks=([1e-15, 1e-10, 1e-5, 1e-1, 1e0], ["10⁻¹⁵", "10⁻¹⁰", "10⁻⁵", "10⁻¹", "10⁰"]),
        lab=false,
        xlabel="time, step = $τ", ylabel="error in $col"
    )
    p = plot!(p,
        df[!, "time"], 
        [[x -> 1e-12x^k + 1e-30 for k in [4, 6, 8, 10]]...; x -> 1e-12exp(x)],
        lab=false,#permutedims([["O(x^$k)" for k in [4, 6, 8, 10]]...; "O(exp(x))"]),
        style=:dash
    )
    push!(pl2, p)
    #savefig(p, "./data/$(landau ? "landau" : "two_stream")/error_$(col)_rank_$(r).pdf")
end
=#

CSV.write(
    "../data/evolution_$(landau ? "landau" : "two_stream")_rank_min_$(r_min)_max_$(r_max)_step_" * replace("$τ", "." => "d") * ".csv", 
    df
)
