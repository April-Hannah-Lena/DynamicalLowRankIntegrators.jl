using Plots, ProgressMeter
using Statistics, LinearAlgebra, Einsum, PaddedViews, BlockDiagonals
using ApproxFun, FastGaussQuadrature
using LinearAlgebra: NoPivot, ColumnNorm
import ClassicalOrthogonalPolynomials as cl
#using ClassicalOrthogonalPolynomials: HermiteWeight
#using ApproxFun.ApproxFunOrthogonalPolynomials: weight

include("./lazy_tensor.jl")
include("./plot.jl")



# parameters 

m = 7 # number of conserved v components
r = 10 # rank
m_x, m_v = 28, 25 # points in x and v 

τ = 1e-3    # time step
t_start = 0.
t_end = 10.
t_grid = t_start:τ:t_end

# must be centered around 0 for now
xlims = (-π, π)
#vlims = (-3, 3)

include("./quadrature.jl")



# initial conditions

X0 = x_basis[:, 1:r]
V0 = v_basis[:, 1:r]

α = 0.1
S0 = zeros(r, r)
S0[1, 1] = 1 / ( π^(3/2) * maximum(X0[:, 1]) * maximum(V0[:, 1]) )
S0[2, 1] = -α / ( π^(3/2) * maximum(X0[:, 2]) * maximum(V0[:, 1]) )

#=
for k in 1:r
    S0[k, k] += 1e-4    # for invertibility
end
=#

S0 /= (2π)^3   # normalized e⁻ density

XSV = X0 * S0 * V0'
f = XSV .* f0v'

sin1(x) = sin(x[1])
@assert f ≈ @. ( 1 - α * sin1(x_grid) ) * f0v' / ( (2π)^3 * π^(3/2) )

#@assert ∫dx(∫dv(f)) ≈ [1]



include("./rhs.jl")
include("./step.jl")


plot_density(f, t=0)



# time evolution

X = copy(X0)
S = copy(S0)
V = copy(V0)

f = X * S * V' .* f0v'

#=
time_evolution = [t_start]
mass_evolution = [mass(f)...]
momentum_evolution = [momentum(f)...]
energy_evolution = [energy(f)...]
L1_evolution = [Lp(f, 1)...]
L2_evolution = [Lp(f, 2)...]
entropy_evolution = [entropy(f)...]
minimum_evolution = [minimum(f)]
=#

t = last_t = t_start
done = false
prog = Progress(round(Int, t_end, RoundDown))

while !done

    global X, S, V, t, last_t, f, done

    X, S, V = step(X, S, V, τ)
    XSV = X * S * V'
    t += τ
    #f = X * S * V' .* f0v'
    
    update!(
        prog, round(Int, t, RoundDown), 
        showvalues=[
            ("time", t), 
            ("mass", mass(XSV)...), 
            ("momentum", momentum(XSV)...), 
            ("energy", energy(XSV)...),
            ("L¹ norm", Lp(XSV, 1)...),
            ("L² norm", Lp(XSV, 2)...),
            ("entropy", entropy(XSV)...),
            ("minimum", minimum(XSV .* f0v'))
        ]
    )

    if t - last_t > 0.05
        plot_density(XSV; t=t)
        #f = X * S * V' .* f0v'
        #plot_step(x_grid, v_grid, f, X, S, V, t)#, mass_evolution, momentum_evolution, energy_evolution)
        last_t = t

        #=
        push!(time_evolution, t)
        push!(mass_evolution, mass(f)...)
        push!(momentum_evolution, momentum(f)...)
        push!(energy_evolution, energy(f)...)
        push!(L1_evolution, Lp(f, 1)...)
        push!(L2_evolution, Lp(f, 2)...)
        push!(entropy_evolution, entropy(f)...)
        push!(minimum_evolution, minimum(f))
        =#
    end

    if t ≥ t_end
        done = true
    end

end

#=
begin
    p1 = plot(time_evolution, mass_evolution, lab="mass", ylims=(0.95, 1.05))
    p2 = plot(time_evolution, momentum_evolution, lab="momentum", ylims=(-0.05, 0.05))
    p3 = plot(time_evolution, energy_evolution, lab="energy", ylims=(0.2, 0.3))
    p4 = plot(time_evolution, L1_evolution, lab="L¹ norm", ylims=(0.95, 1.05))
    p5 = plot(time_evolution, L2_evolution, lab="L² norm", ylims=(0.2, 0.3))
    p6 = plot(time_evolution, entropy_evolution, lab="entropy", ylims=(2.85, 2.95))
    plot(p1, p2, p3, p4, p5, p6, layout=(2, 3))
end
=#

