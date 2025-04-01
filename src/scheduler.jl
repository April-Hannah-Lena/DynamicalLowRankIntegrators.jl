for condition in [:landau, :twostream, :nonsymmetric]
    for m in 1:5
        run(pipeline(
            `julia --project=.. --threads=20 low-rank-script.jl $condition $m 0 2`,
            stderr="../data/run.log"
        ))
    end
end