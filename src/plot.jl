using Plots



function plot_density(f; title="e⁻ density", t=0, size=(1_000,800), kwargs...)
    heatmap(
        x_grid, v_grid, f', 
        title="$title, time = $(round(t, digits=3))", 
        xlabel="x",
        ylabel="v",
        xticks=([-π, -π/2, 0, π/2, π], ["-π", "-π/2", "0", "π/2", "π"]),
        ylims=(-4,4),
        size=size;
        kwargs...
    )
end

function plot_step(x_grid, v_grid, f, X, S, V, t; size=(1_000,800))#, mass_evolution, momentum_evolution, energy_evolution)
    p1 = plot_density(f, t=t)
    p2 = plot_density(RHS(f), title="RHS", t=t)
    p3 = plot(x_grid, collect(eachcol(X)))
    p4 = plot(
        v_grid, collect(eachcol( sign.(V) .* log10.(abs.(V).+1) )),
        yticks=(
            [-4., -2., -log10(11), -log10(2), 0, log10(2), log10(11), 2., 4.], 
            ["-10000", "-100", "-10", "-1", "0", "1", "10", "100", "10000"]
        ),
        ylims=(-4,4)
    )
    display(plot(p1, p2, p3, p4, size=size, layout=(2,2)))
    #@info "step" time=t mass=mass(f)[1] momentum=momentum(f)[1] energy=energy(f)[1]
    #@info "step" S=S
end
