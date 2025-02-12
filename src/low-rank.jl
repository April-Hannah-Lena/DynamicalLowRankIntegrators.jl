using Plots, ProgressMeter
using Serialization, DataFrames, CSV
using Statistics, LinearAlgebra, Einsum
using ApproxFun, FastGaussQuadrature, Interpolations
const transform = ApproxFun.transform
using LinearAlgebra: NoPivot, ColumnNorm
import ClassicalOrthogonalPolynomials as cl
#using ClassicalOrthogonalPolynomials: HermiteWeight
#using ApproxFun.ApproxFunOrthogonalPolynomials: weight

include("./plot.jl")



# parameters 

const m = 3 # number of conserved v components
const r = 6 # rank
const m_x, m_v = 200, 150 # points in x and v 

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

df = DataFrame(
    "time" => [t_start],
    "mass" => mass(f),
    "momentum" => momentum(f),
    "energy" => energy(f),
    "entropy" => entropy(f),
    "temperature" => temperature(f),
    "heat flux" => heat_flux(f),
    "L1 norm" => Lp(f, 1),
    "L2 norm" => Lp(f, 2),
    "L3 norm" => Lp(f, 3),
    "Linfinity norm" => [maximum(abs.(f))],
    "minimum" => [minimum(f)]
)

record_step = 0.01
t = last_t = last_t_low_res = t_start
done = false
#prog = Progress(round(Int, t_end, RoundDown))
prog = Progress(length(t_start:record_step:t_end)-1)

while !done

    global X, S, V, f, t, last_t, last_t_low_res, done

    #X, S, V, t = try_step(X, S, V, t, τ, 1e-6, 1e-10)    # adaptive step size
    X, S, V = step(X, S, V, τ)      # static step size
    t += τ

    if t ≥ t_end
        done = true
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
            Lp(f, 1)...;
            Lp(f, 2)...;
            Lp(f, 3)...;
            maximum(abs.(f));
            minimum(f);
        ])

        next!(
            prog,
            showvalues=[zip(names(df), df[end,:])...]
        )

        last_t = t

    end

    if t - last_t_low_res ≥ 0.05

        plot_step(x_grid, v_grid, f, X, S, V, t)#, mass_evolution, momentum_evolution, energy_evolution)
        last_t_low_res = t
    
    end

end


begin
    p1 = plot(df[!,"time"], df[!,"L1 norm"], lab="L¹ norm", ylims=(minimum(df[!,"L1 norm"]) - 0.1, maximum(df[!,"L1 norm"]) + 0.1))
    p2 = plot(df[!,"time"], df[!,"L2 norm"], lab="L² norm", ylims=(minimum(df[!,"L2 norm"]) - 0.1, maximum(df[!,"L2 norm"]) + 0.1))
    p3 = plot(df[!,"time"], df[!,"L3 norm"], lab="L³ norm", ylims=(minimum(df[!,"L3 norm"]) - 0.1, maximum(df[!,"L3 norm"]) + 0.1))
    p4 = plot(df[!,"time"], df[!,"entropy"], lab="Entropy", ylims=(minimum(df[!,"entropy"]) - 0.1, maximum(df[!,"entropy"]) + 0.1))
    p5 = plot(df[!,"time"], df[!,"temperature"], lab="Temperature", ylims=(minimum(df[!,"temperature"]) - 0.1, maximum(df[!,"temperature"]) + 0.1))
    p6 = plot(df[!,"time"], df[!,"heat flux"], lab="Heat flux", ylims=(minimum(df[!,"heat flux"]) - 0.1, maximum(df[!,"heat flux"]) + 0.1))
    plot(p1, p2, p3, p4, p5, p6, layout=(2, 3))
end


df_accurate = CSV.read("./evolution_data.csv", DataFrame)
pl = []

for col in names(df)[2:end]
    p = plot(df[!,"time"], df[!,col], lab=col, ylims=(minimum(df[!,col]) - 0.01, maximum(df[!,col]) + 0.01))
    p = plot!(p, df_accurate[!,"time"], df_accurate[!,col], lab="accurate $col")
    push!(pl, p)
end

plot(pl[1:4]...)
plot(pl[5:8]...)
plot(pl[9:11]...)



pl = []

for col in names(df)[2:end]
    interp = linear_interpolation(
        df_accurate[!,"time"], df_accurate[!,col], 
        extrapolation_bc=Interpolations.Line()
    )
    p = plot(
        df[!,"time"], abs.(df[!,col] - map(interp, df[!,"time"])), 
        lab="Error in $col",
        leg=:bottomright,
        yscale=:log10, ylims=(1e-16, 5e-1)
    )
    push!(pl, p)
end

p1 = plot(pl[1:4]...)
p2 = plot(pl[5:8]...)
p3 = plot(pl[9:11]...)

savefig(p1, "errors1.pdf")
savefig(p2, "errors2.pdf")
savefig(p3, "errors3.pdf")
