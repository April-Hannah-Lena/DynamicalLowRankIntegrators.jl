using Plots
using DataFrames, CSV


conditions = :twostream

rank = 10
ranks = [10]
#ranks = [5, 7, 9, 11, 13]
ms = [1, 2, 3, 5]

dataframes = [
    CSV.read(
        #"./data/evolution_two_stream_rank_min_5_max_16_step_0d0005.csv",
        "./data/evolution_$(string(conditions))_rank_min_$(rank)_max_$(rank)_conserved_$(m)_step_0d0005.csv",
        DataFrame
    )
    for m in ms
]

observables = names(dataframes[1])[2:end-1]      # 1st col is time and last col is rank

data_start = 0
data_end = 30

evolution_plots = map(observables) do observable
    p = plot(size=(900,600), xlabel="time (step = 0.0005)", ylabel=observable)
    for ind in eachindex(ms)
        df = dataframes[ind]
        df = df[data_start .< df[!, "time"] .< data_end, :]
        m = ms[ind]
        plot!(p, 
            df[!, "time"], df[!, observable],
            label="m = $m",
            #label="",
            color=ind,
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
    for ind in eachindex(ms)
        df = dataframes[ind]
        df = df[data_start .< df[!, "time"] .< data_end, :]
        m = ms[ind]
        plot!(p, 
            df[!, "time"], 
            abs.(df[!, observable] .- df[1, observable]) .+ 1e-30,
            label="m = $m",
            #label="",
            linewidth=3,
            color=ind
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

inner_product_norm_plots = map(2:7) do power

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

    for ind in eachindex(ms)
        df = dataframes[ind]
        df = df[data_start .< df[!, "time"] .< data_end, :]
        m = ms[ind]

        plot!(p, 
            df[:, "time"],
            abs2.(df[:, "moment $power continuity error"]),
            label="Error, m = $m",
            linewidth=3,
            color=ind
        )
        plot!(p, 
            df[:, "time"],
            df[:, "moment $power norm error"],
            label="Pessimistic estimate, m = $m",
            linewidth=3,
            color=ind,
            style=:dash
        )
    end

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


