function cutR(
    x;
    start = "minimum",
    stop = "maximum",
    n_bins = 2,
    return_breaks = true,
    return_mid_points = false,
)
    if (start == "minimum")
        start = minimum(x)
    end
    if (stop == "maximum")
        stop = maximum(x)
    end
    bw = (stop - start) / (n_bins - 1)
    midPts = zeros(n_bins)
    for i = 1:n_bins
        midPts[i] = start + (i - 1) * bw
    end
    breaks = collect(range(start - (bw / 2); length = n_bins + 1, stop = stop + (bw / 2)))
    y = zeros(size(x, 1))
    for j = 1:size(x, 1)
        for i = 1:n_bins
            if (x[j] >= breaks[i]) && (x[j] < breaks[i+1])
                y[j] = i
            end
            if i == n_bins && x[j] == breaks[i+1]
                y[j] = i
            end
        end
    end
    if (return_breaks == true || return_mid_points == true)
        if return_mid_points == false
            return (Int.(y), breaks)
        elseif return_breaks == false
            return (Int.(y), midPts)
        else
            return (Int.(y), breaks, midPts)
        end
    else
        return Int.(y)
    end
end
function subset_data(
    dt::Data,
    subset::Vector{Int64},
    est::Block,
    sd::Block,
) #method= Booklet(subset=Scalar) or Students(subset=array)
    N = size(dt.responses, 1)
    subset = sort(subset)
    new_responses = dt.responses[:, subset]
    new_design = dt.design[:, subset]
    est.pars = est.pars[:, findall(sum(new_design, dims = 2) .>= 1)]
    est.latent_values = est.latent_values[subset, :]
    new_n_items, new_N = size(new_responses)
    if size(sd.pars, 1) > 0
        new_simulated_data = sd
        new_simulated_data.pars = sd.pars[:, findall(sum(new_design, dims = 2) .>= 1)]
        new_simulated_data.latent_values = sd.latent_values[subset, :]
        return new_N::Int64,
        new_n_items::Int64,
        new_responses::Matrix{Float64},
        new_design::Matrix{Float64},
        est::Block,
        new_simulated_data::Block
    else
        return new_N::Int64,
        new_n_items::Int64,
        new_responses::Matrix{Float64},
        new_design::Matrix{Float64},
        est::Block
    end
end
