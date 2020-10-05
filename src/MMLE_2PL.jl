function calibrate(mdl::LatentModel)
    if mdl.irt.n_par > 1
        firstLatent = 2
    else
        firstLatent = 1
    end
    n_items = mdl.irt.n_items
    n_par = mdl.irt.n_par
    n_latent = mdl.irt.n_latent
    nTotPar = n_par - 1 + n_latent
    N = mdl.irt.N
    T = mdl.dt.T
    n = mdl.dt.n
    K = mdl.ext_opt.K
    X = ones(Float64, K, n_latent + 1)
    W = zeros(Float64, K, n_latent + 1) .+ (1 / K)
    for l = 1:n_latent
        X[:, l+1] = Distributions.support(mdl.estimates.latents[l].dist)
        W[:, l+1] = Distributions.probs(mdl.estimates.latents[l].dist)
    end
    Wk = copy(W[:, firstLatent])
    Xk = copy(X[:, firstLatent])
    startPars = copy(mdl.estimates.pars)
    startEst = copy(mdl.estimates)
    if mdl.bootstrap.perform
        #takes only the first Latent
        (bins, none) = cutR(
            startEst.latent_values[:, firstLatent];
            start = mdl.bounds.min_latent[1],
            stop = mdl.bounds.max_latent[1],
            n_bins = K,
            return_breaks = false,
            return_mid_pts = true,
        )
        #pθ=Wk[bins]
        #pθ=pθ./sum(pθ)
        BSPar = Vector{DataFrames.DataFrame}(undef, nTotPar) #nTotPar x R+1 (id + bootstrap)
        for p = 1:nTotPar
            BSPar[p] = DataFrames.DataFrame(id = collect(1:n_items))
            if p == 1
                name = "b"
            elseif n_par == 2
                name = string("a_", p - 1)
            elseif p == nTotPar
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
        BSLatentVals = Vector{DataFrames.DataFrame}(undef, n_latent) #nTotPar x R+1 (id + bootstrap)
        for l = 1:n_latent
            BSLatentVals[l] = DataFrames.DataFrame(id = collect(1:N))
            name = string("theta_", l)
            for r = 1:mdl.bootstrap.R
                DataFrames.insertcols!(
                    BSLatentVals[l],
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
        Wk = copy(W[:, firstLatent])
        Xk = copy(X[:, firstLatent])
        n_index = Array{Array{Int64,1},1}(undef, n_items)
        i_index = Array{Array{Int64,1},1}(undef, N)
        for n = 1:N
            i_index[n] = findall(mdl.dt.design[:, n] .== 1.0)
            if n <= n_items
                n_index[n] = findall(mdl.dt.design[n, :] .== 1.0)
            end
        end #15ms
        if mdl.bootstrap.perform
            mdl.estimates = copy(startEst)
            if mdl.bootstrap.type == "parametric"
                if mdl.dt.unbalanced
                    #nsample=sample(collect(1:N),StatsBase.ProbabilityWeights(pθ), Int(floor(mdl.bootstrap.sample_frac*N)), replace=true)
                    #while size(unique(vcat([i_index[n] for n in nsample]...)),1)<n_items
                    #nsample=rand(Distributions.DiscreteNonParametric(collect(1:N), pθ),Int(floor(mdl.sample_frac*N)))
                    #	nsample=sample(collect(1:N),StatsBase.ProbabilityWeights(pθ), Int(floor(mdl.bootstrap.sample_frac*N)), replace=true)
                    #end
                    nsample = zeros(Int(floor(mdl.bootstrap.sample_frac * N)))
                    nsample = rand(
                        Distributions.DiscreteNonParametric(collect(1:K), Wk),
                        Int(floor(mdl.bootstrap.sample_frac * N)),
                    )
                    for n = 1:size(nsample, 1)
                        ns = copy(nsample[n])
                        while size(findall(bins .== ns), 1) == 0
                            if nsample[n] > (K / 2)
                                ns -= 1
                            else
                                ns += 1
                            end
                        end
                        nsample[n] = sample(findall(bins .== ns))
                    end
                    while size(unique(vcat([i_index[n] for n in nsample]...)), 1) < n_items
                        nsample = rand(
                            Distributions.DiscreteNonParametric(collect(1:K), Wk),
                            Int(floor(mdl.bootstrap.sample_frac * N)),
                        )
                        for n = 1:size(nsample, 1)
                            ns = copy(nsample[n])
                            while size(findall(bins .== ns), 1) == 0
                                if nsample[n] > (K / 2)
                                    ns -= 1
                                else
                                    ns += 1
                                end
                            end
                            nsample[n] = sample(findall(bins .== ns))
                        end
                    end
                    println(size(nsample, 1))
                else
                    tsamp = Vector{Vector{Int64}}(undef, T)
                    for t = 1:T
                        nt = Int.(collect(0:(N/(T)))[1:end-1] .* T .+ t) #subjects that have take the test t, sample among these
                        bins_t = bins[nt]
                        pop = sort!(rand(
                            Distributions.DiscreteNonParametric(collect(1:K), Wk),
                            size(nt, 1),
                        ))
                        samplet = Int64[]
                        for k = 1:K
                            if sum(pop .== k) > 0 && sum(bins_t .== k) > 0
                                s = sample(nt[bins_t.==k], sum(pop .== k), replace = true)
                                samplet = vcat(samplet, s)
                            end
                        end
                        #add more samples if they are not enough
                        while size(samplet, 1) != size(pop, 1)
                            k = sample(collect(1:K))
                            if sum(pop .== k) > 0 && sum(bins_t .== k) > 0
                                samplet = vcat(samplet, sample(nt[bins_t.==k], 1))
                            end
                        end
                        tsamp[t] = copy(samplet)
                    end
                    nsample = sort(vcat(tsamp...))
                end
            else #non-parametric
                if mdl.dt.unbalanced
                    nsample = sample(
                        collect(1:N),
                        Int(ceil(mdl.bootstrap.sample_frac * N)),
                        replace = true,
                    )
                    while size(unique(vcat([i_index[n] for n in nsample]...)), 1) < n_items
                        nsample = sample(
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
                    nsample = sort(vcat(tsamp...))
                end
            end
            if size(mdl.simulated_data.pars, 1) > 0
                (new_N, newI, new_responses, newDesign, newEst, newSd) =
                    subset_data(mdl.dt, nsample, T, n, mdl.estimates, mdl.simulated_data)
            else
                (new_N, newI, new_responses, newDesign, newEst) =
                    subset_data(mdl.dt, nsample, T, n, mdl.estimates, mdl.simulated_data)
            end
        else #no bootstrap
            if size(mdl.simulated_data.pars, 1) > 0
                (new_N, newI, new_responses, newDesign, newEst, newSd) =
                    (N, n_items, mdl.dt.responses, mdl.dt.design, mdl.estimates, mdl.simulated_data)
            else
                (new_N, newI, new_responses, newDesign, newEst) =
                    (N, n_items, mdl.dt.responses, mdl.dt.design, mdl.estimates)
            end
        end
        #index of missings
        n_index = Array{Array{Int64,1},1}(undef, newI)
        i_index = Array{Array{Int64,1},1}(undef, new_N)
        for n = 1:new_N
            i_index[n] = findall(newDesign[:, n] .== 1.0)
            if n <= newI
                n_index[n] = findall(newDesign[n, :] .== 1.0)
            end
        end
        ########################################################################
        ### initialize variables
        ########################################################################
        startTime = time()
        newPars = copy(newEst.pars)
        oldPars = Vector{Matrix{Float64}}(undef, 2)
        oldPars[1] = copy(newPars)
        # oldPars[2]=copy(newPars)
        xGap = Inf
        xGapOld = ones(Float64, 7) .* Inf
        new_likelihood = -Inf
        oldLh = ones(Float64, 3) .* (-Inf)
        oldLatentVals = copy(newEst.latent_values)
        newLatentVals = copy(oldLatentVals)
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
        phi = X * newPars #K x I
        posterior = zeros(Float64, new_N, K)
        likelihood_matrix = similar(posterior)
        oneoverN = fill(1 / new_N, new_N)
        newLatentVals = zeros(new_N, n_latent + 1)
        #r=zeros(Float64,n_par,newI)
        println(newPars[1, 1])
        println(Xk[30:40])
        println(Wk[30:40])
        while endOfWhile < 1
            ################################################################
            ####					ACCELLERATE
            ################################################################
            #if s>1
            # 	if exp(oldLh[1]-new_likelihood)>0.1
            # 		d2=oldPars[1]-oldPars[2]
            # 		d1=newPars-oldPars[1]
            # 		d1d2=d1-d2
            # 		ratio=sqrt(sum(d1.^2) / sum(d1d2.^2))
            # 		accel=clamp(1-ratio,-5.0,Inf)
            # 		newPars=((1-accel).*newPars)+(accel.*oldPars[1])#+(1-exp(new_likelihood-oldLh[1])).*newPars
            # 		println("accellerate!")
            # 		LinearAlgebra.BLAS.gemm!('N', 'N', one(Float64), X, newPars, zero(Float64), phi)# phi=New_pars*X1', if A'*B then 'T', 'N'
            ####ADAGRAD
            # delta=1e-7
            # g=oldPars[1]-newPars
            # r=r+g.^2
            # newPars=newPars-g-(1e-5.*(sqrt.(r).+delta).*g)
            #
            # 	end
            #end

            ####################################################################

            if mdl.ext_opt.first == "items"
                ################################################################
                ####					MStep
                ################################################################
                #before_time=time()
                newPars, phi = max_LH_MMLE(
                    newPars,
                    phi,
                    posterior,
                    i_index,
                    newDesign,
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
                println(typeof(likelihood_matrix))
                println(typeof(new_responses))
                println(typeof(phi))

                if mdl.ext_opt.denType == "EH" && (
                    s % mdl.ext_opt.int_W == 0 &&
                    s <= mdl.ext_opt.min_max_W[2] &&
                    s >= mdl.ext_opt.min_max_W[1]
                   ) && mdl.bootstrap.perform == false# && xGap>0.5 && mdl.bootstrap.perform==false
                    
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
                elseif typeof(mdl.ext_opt.denType) == Distributions.Distribution
                    if s >= 1
                        for l = 1:n_latent
                            (bins, X[:, l+1]) = cutR(
                                newLatentVals[:, l+1];
                                star = mdl.bounds.min_latent[l],
                                stop = mdl.bounds.max_latent[l],
                                n_bins = K + (toadd * 2) - 1,
                                return_breaks = false,
                                return_mid_pts = true,
                            )
                            X[:, l+1] = X[(toadd+1):K+toadd, l+1]
                            W[:, l+1] = pdf.(denType, X[:, l+1])
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
            deltaPars = (newPars - oldPars[1]) ./ oldPars[1]
            oldPars[2] = oldPars[1]
            oldPars[1] = copy(newPars)
            #xGap
            xGap = maximum(abs.(deltaPars))
            bestxGap = min(xGap, xGapOld[1])
            xGapOld[6] = copy(xGapOld[5])
            xGapOld[5] = copy(xGapOld[4])
            xGapOld[4] = copy(xGapOld[3])
            xGapOld[3] = copy(xGapOld[2])
            xGapOld[2] = copy(xGapOld[1])
            xGapOld[1] = copy(bestxGap)
            oldLatentVals = copy(newLatentVals)
            ############################################################

            #SECOND STEP
            if mdl.ext_opt.first == "items"
                before_time = time()
                ################################################################
                ####					RESCALE
                ################################################################
                if mdl.ext_opt.denType == "EH" &&
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
                elseif typeof(mdl.ext_opt.denType) == Distributions.Distribution
                    if s >= 1
                        for l = 1:n_latent
                            (bins, X[:, l+1]) = cutR(
                                newLatentVals[:, l+1];
                                star = mdl.bounds.min_latent[l],
                                stop = mdl.bounds.max_latent[l],
                                n_bins = K + (toadd * 2) - 1,
                                return_breaks = false,
                                return_mid_pts = true,
                            )
                            X[:, l+1] = X[(toadd+1):K+toadd, l+1]
                            W[:, l+1] = pdf.(denType, X[:, l+1])
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
                newPars, phi = max_LH_MMLE(
                    newPars,
                    phi,
                    posterior,
                    i_index,
                    newDesign,
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
            #println("RMSE for pars is ",RMSE(newPars,newSd.pars))
            #println("RMSE for latents is ",RMSE(newLatentVals',newSd.latent_values'))
            #println("RMSE for θ is ",RMSE(newLatentVals,newSimθ))
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
                # Bestθ=copy(newLatentVals)
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
                deltaPars = (newPars - oldPars[1]) ./ oldPars[1]
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
                newLatentVals = (posterior * X) ./ (posterior * ones(K, n_latent + 1))
            end
            s = s + 1
            ####################################################################

        end

        if mdl.bootstrap.perform
            isample = findall(sum(newDesign, dims = 2) .> 0)
            for p = 1:nTotPar
                BSPar[p][isample, r+1] .= newPars[p, :]
            end
            for l = 1:n_latent
                BSLatentVals[l][nsample, r+1] .= newLatentVals[:, l]
            end
        else
            mdl.performance.time = time() - startTime
            mdl.estimates.pars = newPars
            mdl.estimates.latent_values = newLatentVals
            mdl.estimates.latents[1].dist = Distributions.DiscreteNonParametric(Xk, Wk)
            mdl.performance.n_iter = s - 1
        end
        println("end of ", r, " bootstrap replication")
    end
    if mdl.bootstrap.perform
        JLD2.@save "pars.jld2" BSPar
        JLD2.@save "latent_values.jld2" BSLatentVals
    end

    return mdl
    # CSV.write("bootstrap 1/estPoolGlobal.csv", ItemPars)
    # writedlm("bootstrap 1/estAbilitiesGlobal.csv",Bestθ,'\t')
end
