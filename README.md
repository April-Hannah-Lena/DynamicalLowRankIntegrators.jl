# DynamicalLowRankIntegrators

This is a project done in 2025 on a dynamical low-rank solver for the Vlasov-Poission equation. After initializing the `Project.toml` by 
```julia
julia> using Pkg    # run from the Project folder

julia> Pkg.activate(".")
```
take a look at `src/low-rank.jl`. There, the integrator is run on various 1+1-dimensional examples. Code performing the integration is in 
```
src/quadrature.jl
src/rhs.jl
src/step.jl
```
