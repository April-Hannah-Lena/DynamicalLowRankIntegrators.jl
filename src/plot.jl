using Plots



function plot_density(f; title="e⁻ density", t=0)
    heatmap(
        x_grid, v_grid, f', 
        title="$title, time = $(round(t, digits=4))", 
        xlabel="x",
        ylabel="v",
        xticks=([-π, -π/2, 0, π/2, π], ["-π", "-π/2", "0", "π/2", "π"])
    )
end

function plot_basis(xs, f)
    plot(xs, collect(eachcol(f)))
end

function plot_step(x_grid, v_grid, f, X, S, V, t)#, mass_evolution, momentum_evolution, energy_evolution)
    p1 = plot_density(f, t=t)
    p2 = plot_density(RHS(f), title="RHS", t=t)
    p3 = plot_basis(x_grid, X)
    p4 = plot_basis(v_grid, V)
    display(plot(p1, p2, p3, p4, layout=(2,2)))
    #@info "step" time=t mass=mass(f)[1] momentum=momentum(f)[1] energy=energy(f)[1]
    #@info "step" S=S
end