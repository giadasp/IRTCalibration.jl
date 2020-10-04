function save_pool(pars::Matrix{Float64}, n_par::Int64, folder)
    #names of parameters
    nTotPars, n_items = size(pars)
    parNames = Vector{String}(undef, nTotPars)
    parNames[1] = "b"
    for p = 2:nTotPars
        parNames[p] = string("a_", p - 1)
    end
    if n_par == 3
        parNames[end] = "c"
    end
    simPool = DataFrames.DataFrame(id = 1:n_items)
    for p = 1:nTotPars
        DataFrames.insertcols!(simPool, p + 1, Symbol(parNames[p]) => pars[p, :])
    end
    CSV.write(string(folder, "/simPool.csv"), simPool)
end

function save_latent_values(latent_values::Matrix{Float64}, folder::String)
    nStud, n_latent = size(latent_values)
    latNames = Vector{String}(undef, n_latent)
    latNames[1] = "intercept"
    for l = 2:n_latent
        latNames[l] = string("t_", l - 1)
    end
    simLatent = DataFrames.DataFrame(id = 1:nStud)
    for l = 1:n_latent
        DataFrames.insertcols!(simLatent, l + 1, Symbol(latNames[l]) => latent_values[:, l])
    end
    CSV.write(string(folder, "/simLatentVals.csv"), simLatent)
end
