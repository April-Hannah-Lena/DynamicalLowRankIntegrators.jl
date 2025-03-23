using Plots
using DataFrames, CSV


landau = true

ranks = [7]
#ranks = [5, 7, 9, 11, 13]

dataframes = [
    CSV.read(
        #"./data/evolution_two_stream_rank_min_5_max_16_step_0d0005.csv",
        "./data/evolution_$(landau ? "landau" : "two_stream")_rank_min_$(rank)_max_$(rank)_conserved_5_step_0d0005.csv",
        DataFrame
    )
    for rank in ranks
]

observables = names(dataframes[1])[2:end-1]      # 1st col is time and last col is rank

data_start = 0
data_end = 30

evolution_plots = map(observables) do observable
    p = plot(size=(900,600), xlabel="time (step = 0.0005)", ylabel=observable)
    for ind in eachindex(ranks)
        df = dataframes[ind]
        df = df[data_start .< df[!, "time"] .< data_end, :]
        rank = ranks[ind]
        plot!(p, 
            df[!, "time"], df[!, observable],
            label="r = $rank",
            #label="",
            color=df[:, "rank"],
            linewidth=3
        )
    end
    #=for rank in 5:15
        plot!(p,
            [data_start], [dataframes[1][1,observable]], 
            label="rank = $rank",
            linewidth=3,
            color=rank
        )
    end=#
    p
end

#ylims!(evolution_plots[1], (1-1e-3, 1+1e-3))
#ylims!(evolution_plots[2], (-1e-3, 1e-3))

error_plots = map(observables) do observable
    p = plot(
        size=(900,600),
        xlabel="time (step = 0.0005)", 
        ylabel="error in $observable", 
        legend=:bottomright,
        yscale=:log10,
        ylims=(eps()/2, 2), 
        yticks=(
            [ 1e-15,   1e-10,   1e-5,   1e-1,   1e0 ], 
            ["10⁻¹⁵", "10⁻¹⁰", "10⁻⁵", "10⁻¹", "10⁰"]
        ),
    )
    for ind in eachindex(ranks)
        df = dataframes[ind]
        df = df[data_start .< df[!, "time"] .< data_end, :]
        rank = ranks[ind]
        plot!(p, 
            df[!, "time"], 
            abs.(df[!, observable] .- df[1, observable]) .+ 1e-30,
            label="r = $rank",
            #label="",
            linewidth=3,
            color=df[:, "rank"]
        )
    end
    #=for rank in 5:15
        plot!(p,
            [data_start], [dataframes[1][1,observable]], 
            label="rank = $rank",
            linewidth=3,
            color=rank
        )
    end
    plot!(p,
        0:0.001:30, 
        [[x -> 1e-12x^k + 1e-30 for k in [4, 6, 8, 10]]...; x -> 1e-12exp(x)],
        label=permutedims([["O(x^$k)" for k in [4, 6, 8, 10]]...; "O(exp(x))"]),
        style=:dash,
        color=[16:20;]'
    )=#
    p
end


df1 = dataframes[1]

inner_product_norm_plots = map(3:7) do power

    rows = data_start .< df1[!, "time"] .< data_end
    
    p =  plot(
        title="Error in $power" * (power == 3 ? "rd" : "th") * " moment",
        size=(900,600),
        xlabel="time (step = 0.0005)", 
        #ylabel="error in $observable", 
        legend=:bottomright,
        yscale=:log10,
        ylims=(1e-10, 1e5), 
        yticks=(
            [ 1e-10,   1e-5,   1e-1,   1e0,   1e1,   1e5 ], 
            ["10⁻¹⁰", "10⁻⁵", "10⁻¹", "10⁰", "10¹", "10⁵"]
        ),
    )

    plot!(p, 
        df[rows, "time"],
        abs.(df1[rows, "moment $power continuity error"]),
        label="Inner product error representation",
        linewidth=3
    )
    plot!(p, 
        df[rows, "time"],
        df1[rows, "moment $power norm error"],
        label="Pessimistic norm estimate",
        linewidth=3
    )

    p
end


for ind in eachindex(observables)
    safe_string = replace(observables[ind], " " => "_")
    savefig(
        evolution_plots[ind], 
        "./plots/evolution/mixed_terms_$safe_string.pdf"
    )
    savefig(
        error_plots[ind], 
        "./plots/error/mixed_terms_$safe_string.pdf"
    )
end





using JLD2, UnPack

filenames = readdir("./data/mixed_terms")
times = [parse(Float64, replace(file[1:end-5], "d" => ".")) for file in filenames]
perm = sortperm(times)
filenames .= filenames[perm]
times .= times[perm]

prog = Progress(length(times), showspeed=true)
anim = @animate for (time, filename) in zip(times, filenames)
    file = jldopen("./data/mixed_terms/" * filename, "r")
    @unpack X, S, V = file
    f = X * S * V' .* f0v'
    next!(prog)
    plot_density(f, time=time, clims=(0., 0.08))
end

mp4(anim)


