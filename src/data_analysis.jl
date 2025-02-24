using Plots
using DataFrames, CSV


landau = true

ranks = [5, 7, 9, 11, 13]
dataframes = [
    CSV.read(
        "./data/evolution_$(landau ? "landau" : "two_stream")_rank_min_$(rank)_max_$(rank)_step_0d0005.csv",
        DataFrame
    )
    for rank in ranks
]

observables = names(dataframes[1])[2:end-1]      # 1st col is time and last col is rank


evolution_plots = map(observables) do observable
    p = plot(size=(900,600), xlabel="time (step = 0.0005)", ylabel=observable)
    for ind in eachindex(ranks)
        df = dataframes[ind]
        rank = ranks[ind]
        plot!(p, 
            df[!, "time"], df[!, observable],
            label="r = $rank"
        )
    end
    p
end

error_plots = map(observables) do observable
    p = plot(
        size=(900,600),
        xlabel="time (step = 0.0005)", 
        ylabel="error in $observable", 
        legend=:bottomright,
        yscale=:log10,
        ylims=(eps(), 2), 
        yticks=(
            [ 1e-15,   1e-10,   1e-5,   1e-1,   1e0 ], 
            ["10⁻¹⁵", "10⁻¹⁰", "10⁻⁵", "10⁻¹", "10⁰"]
        ),
    )
    for ind in eachindex(ranks)
        df = dataframes[ind]
        rank = ranks[ind]
        plot!(p, 
            df[!, "time"], 
            abs.(df[!, observable] .- df[1, observable]) .+ 1e-30,
            label="r = $rank"
        )
    end
    plot!(p,
        0:0.001:30, 
        [[x -> 1e-12x^k + 1e-30 for k in [4, 6, 8, 10]]...; x -> 1e-12exp(x)],
        label=false,#permutedims([["O(x^$k)" for k in [4, 6, 8, 10]]...; "O(exp(x))"]),
        style=:dash
    )
    p
end


for ind in eachindex(observables)
    safe_string = replace(observables[ind], " " => "_")
    savefig(
        evolution_plots[ind], 
        "./plots/evolution/$(landau ? "landau" : "two_stream")_$safe_string.pdf"
    )
    savefig(
        error_plots[ind], 
        "./plots/error/$(landau ? "landau" : "two_stream")_$safe_string.pdf"
    )
end
