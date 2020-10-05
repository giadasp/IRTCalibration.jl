function calibrate(mdl::LatentModel)
    if mdl.irt.n_par > 1
        first_latent = 2
    else
        first_latent = 1
    end
    local n_items = mdl.irt.n_items
    local n_par = mdl.irt.n_par
    local n_latent = mdl.irt.n_latent
    local n_tot_par = n_par - 1 + n_latent
    local N = mdl.irt.N
    local T = mdl.dt.T
    local n = mdl.dt.n
    local K = mdl.ext_opt.K
    local toadd = 2
    X = ones(Float64, K, n_latent + 1)
    W = zeros(Float64, K, n_latent + 1) .+ (1 / K)
    for l = 1:n_latent
        X[:, l+1] = Distributions.support(mdl.estimates.latents[l].dist)
        W[:, l+1] = Distributions.probs(mdl.estimates.latents[l].dist)
    end
    Wk = copy(W[:, first_latent])
    Xk = copy(X[:, first_latent])
    starting_pars = copy(mdl.estimates.pars)
    starting_estimates = copy(mdl.estimates)
    if mdl.bootstrap.perform
        #takes only the first Latent
        (bins, none) = cutR(
            starting_estimates.latent_values[:, first_latent];
            start = mdl.bounds.min_latent[1],
            stop = mdl.bounds.max_latent[1],
            n_bins = K,
            return_breaks = false,
            return_mid_points = true,
        )
        #pθ=Wk[bins]
        #pθ=pθ./sum(pθ)
        BSPar = Vector{DataFrames.DataFrame}(undef, n_tot_par) #n_tot_par x R+1 (id + bootstrap)
        for p = 1:n_tot_par
            BSPar[p] = DataFrames.DataFrame(id = collect(1:n_items))
            if p == 1
                name = "b"
            elseif n_par == 2
                name = string("a_", p - 1)
            elseif p == n_tot_par
                name = string("c")
            else
                name = string("a_", p - 1)
            end
            for r = 1:mdl.bootstrap.R
                DataFrames.insertcols!(
                    BSPar[p],
                    r + 1,
                    Symbol(string(name, "_", r)) => zeros(Float64, n_items),
                )
            end
        end
        bootstrap_latent_values = Vector{DataFrames.DataFrame}(undef, n_latent) #n_tot_par x R+1 (id + bootstrap)
        for l = 1:n_latent
            bootstrap_latent_values[l] = DataFrames.DataFrame(id = collect(1:N))
            name = string("theta_", l)
            for r = 1:mdl.bootstrap.R
                DataFrames.insertcols!(
                    bootstrap_latent_values[l],
                    r + 1,
                    Symbol(string(name, "_", r)) => zeros(Float64, N),
                )
            end
        end
    end
    s = 1
    if mdl.bootstrap.perform == false
        R = 1
    else
        R = mdl.bootstrap.R
    end
    for r = 1:R
        Wk = copy(W[:, first_latent])
        Xk = copy(X[:, first_latent])
        n_index = Array{Array{Int64,1},1}(undef, n_items)
        i_index = Array{Array{Int64,1},1}(undef, N)
        for n = 1:N
            i_index[n] = findall(mdl.dt.design[:, n] .== 1.0)
            if n <= n_items
                n_index[n] = findall(mdl.dt.design[n, :] .== 1.0)
            end
        end #15ms
        if mdl.bootstrap.perform
            mdl.estimates = copy(starting_estimates)
            if mdl.bootstrap.type == "parametric"
                if mdl.dt.unbalanced
                    #n_sample=sample(collect(1:N),StatsBase.ProbabilityWeights(pθ), Int(floor(mdl.bootstrap.sample_frac*N)), replace=true)
                    #while size(unique(vcat([i_index[n] for n in n_sample]...)),1)<n_items
                    #n_sample=rand(Distributions.DiscreteNonParametric(collect(1:N), pθ),Int(floor(mdl.sample_frac*N)))
                    #	n_sample=sample(collect(1:N),StatsBase.ProbabilityWeights(pθ), Int(floor(mdl.bootstrap.sample_frac*N)), replace=true)
                    #end
                    n_sample = zeros(Int(floor(mdl.bootstrap.sample_frac * N)))
                    n_sample = rand(
                        Distributions.DiscreteNonParametric(collect(1:K), Wk),
                        Int(floor(mdl.bootstrap.sample_frac * N)),
                    )
                    for n = 1:size(n_sample, 1)
                        ns = copy(n_sample[n])
                        while size(findall(bins .== ns), 1) == 0
                            if n_sample[n] > (K / 2)
                                ns -= 1
                            else
                                ns += 1
                            end
                        end
                        n_sample[n] = sample(findall(bins .== ns))
                    end
                    while size(unique(vcat([i_index[n] for n in n_sample]...)), 1) < n_items
                        n_sample = rand(
                            Distributions.DiscreteNonParametric(collect(1:K), Wk),
                            Int(floor(mdl.bootstrap.sample_frac * N)),
                        )
                        for n = 1:size(n_sample, 1)
                            ns = copy(n_sample[n])
                            while size(findall(bins .== ns), 1) == 0
                                if n_sample[n] > (K / 2)
                                    ns -= 1
                                else
                                    ns += 1
                                end
                            end
                            n_sample[n] = sample(findall(bins .== ns))
                        end
                    end
                    println(size(n_sample, 1))
                else
                    tsamp = Vector{Vector{Int64}}(undef, T)
                    for t = 1:T
                        nt = Int.(collect(0:(N/(T)))[1:end-1] .* T .+ t) #subjects that have take the test t, sample among these
                        bins_t = bins[nt]
                        pop = sort!(rand(
                            Distributions.DiscreteNonParametric(collect(1:K), Wk),
                            size(nt, 1),
                        ))
                        sample_t = Int64[]
                        for k = 1:K
                            if sum(pop .== k) > 0 && sum(bins_t .== k) > 0
                                s = sample(nt[bins_t.==k], sum(pop .== k), replace = true)
                                sample_t = vcat(sample_t, s)
                            end
                        end
                        #add more samples if they are not enough
                        while size(sample_t, 1) != size(pop, 1)
                            k = sample(collect(1:K))
                            if sum(pop .== k) > 0 && sum(bins_t .== k) > 0
                                sample_t = vcat(sample_t, sample(nt[bins_t.==k], 1))
                            end
                        end
                        tsamp[t] = copy(sample_t)
                    end
                    n_sample = sort(vcat(tsamp...))
                end
            else #non-parametric
                if mdl.dt.unbalanced
                    n_sample = sample(
                        collect(1:N),
                        Int(ceil(mdl.bootstrap.sample_frac * N)),
                        replace = true,
                    )
                    while size(unique(vcat([i_index[n] for n in n_sample]...)), 1) < n_items
                        n_sample = sample(
                            collect(1:N),
                            Int(floor(mdl.bootstrap.sample_frac * N)),
                            replace = true,
                        )
                    end
                else
                    tsamp = Vector{Vector{Int64}}(undef, T)
                    for t = 1:T
                        nt = Int.(collect(0:(N/(T)))[1:end-1] .* T .+ t)
                        tsamp[t] = sample(
                            nt,
                            Int(floor(mdl.bootstrap.sample_frac * N / T)),
                            replace = true,
                        )
                    end
                    n_sample = sort(vcat(tsamp...))
                end
            end
            if size(mdl.simulated_data.pars, 1) > 0
                (new_N, new_n_items, new_responses, new_design, new_estimates, new_simulated_data) =
                    subset_data(mdl.dt, n_sample, mdl.estimates, mdl.simulated_data)
            else
                (new_N, new_n_items, new_responses, new_design, new_estimates) =
                    subset_data(mdl.dt, n_sample, mdl.estimates, mdl.simulated_data)
            end
        else #no bootstrap
            if size(mdl.simulated_data.pars, 1) > 0
                (new_N, new_n_items, new_responses, new_design, new_estimates, new_simulated_data) =
                    (N, n_items, mdl.dt.responses, mdl.dt.design, mdl.estimates, mdl.simulated_data)
            else
                (new_N, new_n_items, new_responses, new_design, new_estimates) =
                    (N, n_items, mdl.dt.responses, mdl.dt.design, mdl.estimates)
            end
        end
        #index of missings
        n_index = Array{Array{Int64,1},1}(undef, new_n_items)
        i_index = Array{Array{Int64,1},1}(undef, new_N)
        for n = 1:new_N
            i_index[n] = findall(new_design[:, n] .== 1.0)
            if n <= new_n_items
                n_index[n] = findall(new_design[n, :] .== 1.0)
            end
        end
        ########################################################################
        ### initialize variables
        ########################################################################
        startTime = time()
        new_pars = copy(new_estimates.pars)
        oldPars = Vector{Matrix{Float64}}(undef, 2)
        oldPars[1] = copy(new_pars)
        # oldPars[2]=copy(new_pars)
        xGap = Inf
        xGapOld = ones(Float64, 7) .* Inf
        new_likelihood = -Inf
        oldLh = ones(Float64, 3) .* (-Inf)
        oldLatentVals = copy(new_estimates.latent_values)
        new_latent_vals = copy(oldLatentVals)
        ###############################################################################
        ### optimize (item parameters)
        ###############################################################################
        BestLh = -Inf
        newTime = time()
        oldTime = time()
        fGap = Inf
        nloptalg = Symbol("LD_SLSQP")
        limES = 0.5
        s = 1
        endOfWhile = 0
        startTime = time()
        phi = X * new_pars #K x I
        posterior = zeros(Float64, new_N, K)
        likelihood_matrix = similar(posterior)
        oneoverN = fill(1 / new_N, new_N)
        new_latent_vals = zeros(new_N, n_latent + 1)
        #r=zeros(Float64,n_par,new_n_items)
        println(new_pars[1, 1])
        println(Xk[30:40])
        println(Wk[30:40])
        while endOfWhile < 1
            ################################################################
            ####					ACCELLERATE
            ################################################################
            #if s>1
            # 	if exp(oldLh[1]-new_likelihood)>0.1
            # 		d2=oldPars[1]-oldPars[2]
            # 		d1=new_pars-oldPars[1]
            # 		d1d2=d1-d2
            # 		ratio=sqrt(sum(d1.^2) / sum(d1d2.^2))
            # 		accel=clamp(1-ratio,-5.0,Inf)
            # 		new_pars=((1-accel).*new_pars)+(accel.*oldPars[1])#+(1-exp(new_likelihood-oldLh[1])).*new_pars
            # 		println("accellerate!")
            # 		LinearAlgebra.BLAS.gemm!('N', 'N', one(Float64), X, new_pars, zero(Float64), phi)# phi=New_pars*X1', if A'*B then 'T', 'N'
            ####ADAGRAD
            # delta=1e-7
            # g=oldPars[1]-new_pars
            # r=r+g.^2
            # new_pars=new_pars-g-(1e-5.*(sqrt.(r).+delta).*g)
            #
            # 	end
            #end

            ####################################################################

            if mdl.ext_opt.first == "items"
                ################################################################
                ####					MStep
                ################################################################
                #before_time=time()
                new_pars, phi = max_LH_MMLE(
                    new_pars,
                    phi,
                    posterior,
                    i_index,
                    new_design,
                    X,
                    Wk,
                    new_responses,
                    mdl.int_opt,
                    mdl.bounds,
                )
                likelihood_matrix = likelihood(likelihood_matrix, new_N, K, i_index, new_responses, phi)
                new_likelihood = sum(log_c.(LinearAlgebra.BLAS.gemv('N', one(Float64), likelihood_matrix, Wk)))
                #println("time elapsed for Mstep IntOpt ",time()-before_time)
                ####################################################################
            else
                #theta
                #before_time=time()
                ################################################################
                ####					RESCALE
                ################################################################
                if mdl.ext_opt.den_type == "EH" &&
                   (
                       s % mdl.ext_opt.int_W == 0 &&
                       s <= mdl.ext_opt.min_max_W[2] &&
                       s >= mdl.ext_opt.min_max_W[1]
                   ) &&
                   mdl.bootstrap.perform == false# && xGap>0.5 && mdl.bootstrap.perform==false
                    posterior, new_likelihood = compute_posterior(
                        posterior,
                        likelihood_matrix,
                        new_N,
                        K,
                        i_index,
                        new_responses,
                        Wk,
                        phi,
                    )
                    Wk = LinearAlgebra.BLAS.gemv('T', one(Float64), posterior, oneoverN) #if Wk depends only on the likelihoods
                    observed =
                        [LinearAlgebra.dot(Wk, Xk), sqrt(LinearAlgebra.dot(Wk, Xk .^ 2))]
                    observed = [
                        observed[1] - mdl.estimates.latents[1].metric[1],
                        observed[2] / mdl.estimates.latents[1].metric[2],
                    ]
                    #check mean
                    #if  (abs(observed[1])>1e-3 ) || observed[2]<0.99
                    Xk2, Wk2 = my_rescale(Xk, Wk, mdl.estimates.latents[1].metric, observed)
                    Wk = cubic_spline_int(Xk, Xk2, Wk2)
                    #end
                    new_likelihood =
                        sum(log_c.(LinearAlgebra.BLAS.gemv('N', one(Float64), likelihood_matrix, Wk)))
                elseif typeof(mdl.ext_opt.den_type) == Distributions.Distribution
                    if s >= 1
                        for l = 1:n_latent
                            (bins, X[:, l+1]) = cutR(
                                new_latent_vals[:, l+1];
                                start = mdl.bounds.min_latent[l],
                                stop = mdl.bounds.max_latent[l],
                                n_bins = K + (toadd * 2) - 1,
                                return_breaks = false,
                                return_mid_points = true,
                            )
                            X[:, l+1] = X[(toadd+1):K+toadd, l+1]
                            W[:, l+1] = Distributions.pdf.(mdl.ext_opt.den_type, X[:, l+1])
                            W[:, l+1] = W[:, l+1] ./ sum(Wk)
                        end
                    end
                end
                #println("time elapsed for Rescale IntOpt ",time()-before_time)
                ################################################################
            end

            ############################################################
            ####					SAVE
            ############################################################
            #lh
            oldLh[3] = oldLh[2]
            oldLh[2] = oldLh[1]
            oldLh[1] = copy(new_likelihood)
            #pars
            deltaPars = (new_pars - oldPars[1]) ./ oldPars[1]
            oldPars[2] = oldPars[1]
            oldPars[1] = copy(new_pars)
            #xGap
            xGap = maximum(abs.(deltaPars))
            bestxGap = min(xGap, xGapOld[1])
            xGapOld[6] = copy(xGapOld[5])
            xGapOld[5] = copy(xGapOld[4])
            xGapOld[4] = copy(xGapOld[3])
            xGapOld[3] = copy(xGapOld[2])
            xGapOld[2] = copy(xGapOld[1])
            xGapOld[1] = copy(bestxGap)
            oldLatentVals = copy(new_latent_vals)
            ############################################################

            #SECOND STEP
            if mdl.ext_opt.first == "items"
                before_time = time()
                ################################################################
                ####					RESCALE
                ################################################################
                if mdl.ext_opt.den_type == "EH" &&
                   (
                       s % mdl.ext_opt.int_W == 0 &&
                       s <= mdl.ext_opt.min_max_W[2] &&
                       s >= mdl.ext_opt.min_max_W[1]
                   ) &&
                   mdl.bootstrap.perform == false
                    posterior = posterior_simplified(
                        posterior,
                        new_N,
                        K,
                        i_index,
                        new_responses,
                        Wk,
                        phi,
                    )
                    Wk = LinearAlgebra.BLAS.gemv('T', one(Float64), posterior, oneoverN) #if Wk depends only on the likelihoods
                    observed =
                        [LinearAlgebra.dot(Wk, Xk), sqrt(LinearAlgebra.dot(Wk, Xk .^ 2))]
                    observed = [
                        observed[1] - mdl.estimates.latents[1].metric[1],
                        observed[2] / mdl.estimates.latents[1].metric[2],
                    ]
                    #check mean
                    #if  (abs(observed[1])>1e-3 ) || observed[2]<0.99
                    Xk2, Wk2 = my_rescale(Xk, Wk, mdl.estimates.latents[1].metric, observed)
                    Wk = cubic_spline_int(Xk, Xk2, Wk2)
                    #end
                    new_likelihood =
                        sum(log_c.(LinearAlgebra.BLAS.gemv('N', one(Float64), likelihood_matrix, Wk)))
                elseif typeof(mdl.ext_opt.den_type) == Distributions.Distribution
                    if s >= 1
                        for l = 1:n_latent
                            (bins, X[:, l+1]) = cutR(
                                new_latent_vals[:, l+1];
                                start = mdl.bounds.min_latent[l],
                                stop = mdl.bounds.max_latent[l],
                                n_bins = K + (toadd * 2) - 1,
                                return_breaks = false,
                                return_mid_points = true,
                            )
                            X[:, l+1] = X[(toadd+1):K+toadd, l+1]
                            W[:, l+1] = Distributions.pdf.(mdl.ext_opt.den_type, X[:, l+1])
                            W[:, l+1] = W[:, l+1] ./ sum(Wk)
                        end
                    end
                end
                println("time elapsed for rescale IntOpt ", time() - before_time)
                ################################################################
            else
                ################################################################
                ####					MStep
                ################################################################
                before_time = time()
                new_pars, phi = max_LH_MMLE(
                    new_pars,
                    phi,
                    posterior,
                    i_index,
                    new_design,
                    X,
                    Wk,
                    new_responses,
                    mdl.int_opt,
                    mdl.bounds,
                )
                likelihood_matrix = likelihood(likelihood_matrix, new_N, K, i_index, new_responses, phi)
                new_likelihood = sum(log_c.(LinearAlgebra.BLAS.gemv('N', one(Float64), likelihood_matrix, Wk)))
                #println("time elapsed for Mstep IntOpt",time()-before_time)

                ################################################################
            end
            # if size(simulated_data.θ,1)>0 && size(simulated_data.pool,1)>0
            #println("RMSE for pars is ",RMSE(new_pars,new_simulated_data.pars))
            #println("RMSE for latents is ",RMSE(new_latent_vals',new_simulated_data.latent_values'))
            #println("RMSE for θ is ",RMSE(new_latent_vals,newSimθ))
            # end
            println("end of iteration  #", s)
            println("newlikelihood is ", new_likelihood)

            newTime = time()
            ####################################################################
            #                           CHECK CONVERGENCE
            ####################################################################
            if (s >= mdl.ext_opt.max_iter)
                println(
                    "max_iter reached after ",
                    newTime - oldTime,
                    " and ",
                    Int(s),
                    " iterations",
                )
                endOfWhile = 1
                # ItemPars=DataFrames.DataFrame(a=new_a,b=new_b)
                # Bestθ=copy(new_latent_vals)
            end
            if newTime - oldTime > mdl.ext_opt.time_limit
                println(
                    "time_limit reached after ",
                    newTime - oldTime,
                    " and ",
                    Int(s),
                    " iterations",
                )
                endOfWhile = 1
            end
            fGap = abs(new_likelihood - oldLh[1]) / oldLh[1]
            if fGap < mdl.ext_opt.l_tol_rel && fGap >= 0
                println(
                    "f ToL reached after ",
                    newTime - oldTime,
                    " and ",
                    Int(s),
                    " iterations",
                )
                endOfWhile = 1
            end
            if s > 3
                deltaPars = (new_pars - oldPars[1]) ./ oldPars[1]
                xGap = maximum(abs.(deltaPars))
                bestxGap = min(xGap, xGapOld[1])
                println("Max-change is ", xGap)
                if xGap <= mdl.ext_opt.x_tol_rel
                    println(
                        "X ToL reached after ",
                        newTime - oldTime,
                        " and ",
                        Int(s),
                        " iterations",
                    )
                    endOfWhile = 1
                else
                    if s > 20
                        if !(new_likelihood > oldLh[1] && oldLh[1] > oldLh[2] && oldLh[2] > oldLh[3])
                            if xGap < 0.1 &&
                               xGapOld[1] == bestxGap &&
                               xGapOld[1] == xGapOld[2] &&
                               xGapOld[3] == xGapOld[2] &&
                               xGapOld[3] == xGapOld[4] &&
                               xGapOld[4] == xGapOld[5] &&
                               xGapOld[6] == xGapOld[5]
                                endOfWhile = 1
                                println("No better result can be obtained")
                            end
                        end
                    end
                end
            end
            if endOfWhile == 1
                posterior =
                    posterior_simplified(posterior, new_N, K, i_index, new_responses, Wk, phi)
                new_latent_vals = (posterior * X) ./ (posterior * ones(K, n_latent + 1))
            end
            s = s + 1
            ####################################################################

        end

        if mdl.bootstrap.perform
            isample = findall(sum(new_design, dims = 2) .> 0)
            for p = 1:n_tot_par
                BSPar[p][isample, r+1] .= new_pars[p, :]
            end
            for l = 1:n_latent
                bootstrap_latent_values[l][n_sample, r+1] .= new_latent_vals[:, l]
            end
        else
            mdl.performance.time = time() - startTime
            mdl.estimates.pars = new_pars
            mdl.estimates.latent_values = new_latent_vals
            mdl.estimates.latents[1].dist = Distributions.DiscreteNonParametric(Xk, Wk)
            mdl.performance.n_iter = s - 1
        end
        println("end of ", r, " bootstrap replication")
    end
    if mdl.bootstrap.perform
        JLD2.@save "pars.jld2" BSPar
        JLD2.@save "latent_values.jld2" bootstrap_latent_values
    end

    return mdl
    # CSV.write("bootstrap 1/estPoolGlobal.csv", ItemPars)
    # writedlm("bootstrap 1/estAbilitiesGlobal.csv",Bestθ,'\t')
end
