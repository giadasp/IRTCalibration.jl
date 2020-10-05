function cutR(
    x;
    start = "minimum",
    stop = "maximum",
    n_bins = 2,
    return_breaks = true,
    return_mid_pts = false,
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
    if (return_breaks == true || return_mid_pts == true)
        if return_mid_pts == false
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
) #method= Booklet(subset=Scalar) or Students(subset=array), su
    N = size(dt.responses, 1)
    subset = sort(subset)
    new_responses = dt.responses[:, subset]
    newDesign = dt.design[:, subset]
    est.pars = est.pars[:, findall(sum(newDesign, dims = 2) .>= 1)]
    est.latent_values = est.latent_values[subset, :]
    newI, new_N = size(new_responses)
    if size(sd.pars, 1) > 0
        newSd = sd
        newSd.pars = sd.pars[:, findall(sum(newDesign, dims = 2) .>= 1)]
        newSd.latent_values = sd.latent_values[subset, :]
        return new_N::Int64,
        newI::Int64,
        new_responses::Matrix{Float64},
        newDesign::Matrix{Float64},
        est::Block,
        newSd::Block
    else
        return new_N::Int64,
        newI::Int64,
        new_responses::Matrix{Float64},
        newDesign::Matrix{Float64},
        est::Block
    end
end
