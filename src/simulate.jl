function rescale_values(vals::Vector{Float64}, metric::Vector{Float64})
    return (vals .- (mean(vals) - metric[1])) ./ (std(vals) / metric[2])
end#export these
function rescaleLatentValues(latent_values::Matrix{Float64}, metric::Vector{Vector{Float64}})
    nLatentInt = size(latent_values, 2)
    for l = 2:nLatentInt
        latent_values[:, l] = rescale_values(latent_values[:, l], metric[l-1])
    end
    return latent_values
end

function simulate_data(
    irt::IRT,
    latent_metric::Vector{Vector{Float64}},
    latent_dist::Vector{<:Distributions.Distribution},
    pars_dist::Vector{<:Distributions.Distribution},
)
    simPars = Matrix{Float64}(undef, irt.n_latent + 1, irt.n_items)
    simulated_latent_vector = Vector{Latent}(undef, irt.n_latent)
    simulated_latent_values = Matrix{Float64}(undef, irt.N, irt.n_latent + 1)
    simulated_latent_values[:, 1] = ones(irt.N)
    if irt.n_par > 1 #intercept
        simPars[1, :] = rand(pars_dist[1], irt.n_items) #easiness
        for l = 1:(irt.n_latent) #discriminations
            simPars[l+1, :] = rand(pars_dist[l+min(irt.n_par - 1)], irt.n_items)
            simulated_latent_vector[l] = Latent(latent_dist[l], latent_metric[l])
            simulated_latent_values[:, l+1] = rand(latent_dist[l], irt.N)
        end
    else #no easiness
        for l = 1:(irt.n_latent) #discriminations
            simPars[l+1, :] = ones(irt.n_items)
            simulated_latent_vector[l] = Latent(latent_dist[l], latent_metric[l])
            simulated_latent_values[:, l+1] = rand(latent_dist[l], irt.N)
        end
    end
    if irt.n_par > 2 #guessing
        simPars[end, :] = ones(pars_dist[end], irt.n_items) .* 0.5
    end
    return Block(simPars, simulated_latent_values, simulated_latent_vector)
end

function generate_responses(
    f::Vector{Float64},
    pars::Matrix{Float64},
    latent_values::Matrix{Float64},
    design::Matrix{Float64};
    model = "2PL",
    method = "classicUniform",
)
    n_items = size(pars, 2)#3
    N, L = size(latent_values)
    L = L - 1
    nindex = Array{Array{Int64,1},1}(undef, n_items)
    iindex = Array{Array{Int64,1},1}(undef, N)
    for n = 1:N
        iindex[n] = findall(design[:, n] .== 1)
    end
    for i = 1:n_items
        nindex[i] = findall(design[i, :] .== 1)
    end
    pr = probability(pars, latent_values, n_items, N)
    lp = log_c.(pr)
    lq = log_c.(1 .- pr)
    resp = zeros(Float64, n_items, N)
    if method == "cumulatedPersons"
        @fastmath @inbounds for i = 1:n_items
            gapScore = 3
            while gapScore >= 2
                p2 = pr[i, :]
                p2 = hcat((1 .- p2), p2)#2
                unif = rand(Distributions.Uniform(0, 1), N)#5
                n = one(Float64)#6
                while n <= N#7
                    csum = p2[n, 1]#8
                    cat = 0#9
                    if design[i, n] == 0
                        resp[i, n] = 0
                        n = n + 1
                    else
                        while csum < unif[n]
                            cat = cat + 1
                            if (cat == 2)
                                break
                            end
                            csum = csum + p2[n, cat+1]
                        end
                        resp[i, n] = cat
                        n = n + 1
                    end
                end
                gapScore = abs(sum(resp[i, :]) - sum(p2[i, n] for n in nindex[i]))
            end
        end
    end
    if method == "cumulatedItems"
        @fastmath @inbounds for n = 1:N#4
            #gapScore=3
            #while gapScore>=2
            p2 = pr[:, n]
            p2 = hcat((1 .- p2), p2)#2
            unif = rand(Distributions.Uniform(0, 1), n_items)#5
            samplei = sample(collect(1:n_items), n_items, replace = false)
            i = one(Float64)#6
            while i <= n_items#7
                csum = p2[samplei[i], 1]#8
                cat = 0#9
                if design[samplei[i], n] == 0
                    resp[samplei[i], n] = 0#missing
                    i = i + 1
                else
                    while csum < unif[samplei[i]]
                        cat = cat + 1
                        if (cat == 2)
                            break
                        end
                        csum = csum + p2[samplei[i], cat+1]
                    end
                    resp[samplei[i], n] = cat
                    i = i + 1
                end
            end
            #gapScore=abs(sum(skipmissing(resp[:,n]))-sum(p[i,n] for i in iindex[n]))
            #if gapScore>=0.5
            #	println("person ",n," gap=",gapScore)
            #end
            #end
        end
    end
    if method == "classicUniform"
        for n = 1:N#4
            #gapScore=2
            #while gapScore>=0.5
            unif = rand(Distributions.Uniform(0, 1), n_items)#5
            samplei = sample(collect(1:n_items), n_items, replace = false)
            i = 1#6
            while i <= n_items#7
                if design[samplei[i], n] == 0
                    resp[samplei[i], n] = 0#missing
                    i = i + 1
                else
                    if unif[samplei[i]] < pr[samplei[i], n]
                        resp[samplei[i], n] = one(Float64)
                    else
                        resp[samplei[i], n] = 0
                    end
                    i = i + 1
                end
            end
            #gapScore=abs(sum(skipmissing(resp[:,n]))-sum(p[i,n] for i in iindex[n]))
            #if gapScore>=0.5
            #	println("person ",n," gap=",gapScore)
            #end
            #end
        end
    end
    # if method=="MIP"
    # 	for n=1:N
    # 		i_index=iindex[n]
    # 		p=[c[i]+((1-c[i])*(1 / (1 + exp_c(-a[i]*(θ[n]-b[i]))))) for i in i_index]
    # 		p[p.==0].=0.00001
    # 		m=Model(solver=CplexSolver(CPX_PARAM_PREIND=0,CPX_PARAM_MIPEMPHASIS=one(Float64)))
    # 		@variable(m, x[i=one(Float64):size(i_index,1)], Bin)
    # 		@objective(m, Max, sum((x[i]*log_c(p[i]) + ((f[i] - x[i]) * log_c(1-p[i]))) for  i=one(Float64):size(i_index,1)))
    # 		@constraint(m, sum(x[i] for i=one(Float64):size(i_index,1))-sum(p) <=+1)
    # 		@constraint(m, sum(x[i] for i=one(Float64):size(i_index,1))-sum(p) >=-1)
    #
    # 		#@constraint(m, [n=one(Float64):(size(nindex[i],1)-1)], x[n] <= x[n+1] )
    # 		solve(m)
    # 		xopt=getvalue(x)
    # 		resp[:,n].=missing
    # 		resp[iindex[n],n].=Int.(xopt)
    # 		# m=one(Float64)
    # 		# for n=1:N
    # 		#  		if design[i,n]==0
    # 		#  			resp[i,n]=missing
    # 		#  		else
    # 		#  			resp[i,n]=Int(xopt[m])
    # 		# 			m=m+1
    # 		#  		end
    # 		#  end
    # 	end
    # end
    if method == "cumItemsPattern"
        for n = 1:N#4+
            println("start person ", n)
            i_index = iindex[n]
            i_index = size(i_index, 1)
            p2 = pr[i_index, n]
            p2 = hcat((1 .- p2), p2)#2
            patterns = Vector{Vector{Int64}}(undef, 1)
            n_pattern = Vector{Int64}(undef, 1)
            for r = one(Float64):1000
                respn = Vector{Int64}(undef, i_index)
                unif = rand(Distributions.Uniform(0, 1), i_index)#5
                samplei = sample(collect(1:i_index), i_index, replace = false)
                for i = one(Float64):i_index
                    csum = p2[samplei[i], 1]#8
                    cat = 0#9
                    while csum < unif[samplei[i]]
                        cat = cat + 1
                        if (cat == 2)
                            break
                        end
                        csum = csum + p2[samplei[i], cat+1]
                    end
                    respn[i] = cat
                end
                if r == 1
                    patterns[1] = respn
                    n_pattern[1] = one(Float64)
                else
                    println(size(patterns, 1))
                    corr = 0
                    for pat = one(Float64):size(patterns, 1)
                        if patterns[pat] == respn
                            corr = pat
                        end
                    end
                    if corr == 0
                        push!(patterns, respn)
                        n_pattern = hcat(n_pattern, 1)
                    else
                        n_pattern[corr] = n_pattern[corr] + 1
                    end
                end

            end
            n_patterns = size(patterns, 1)
            prob_patterns = n_pattern ./ 1000
            println(prob_patterns)
            resp[i_index, n] .= sample(patterns, Distributions.pweights(prob_patterns), 1)
            println("end person ", n)
        end
    end
    if method == "classicUniformPattern"
        for n = 1:N#4+
            println("start person ", n)
            i_index = iindex[n]
            i_index = size(i_index, 1)
            patterns = Vector{Vector{Int64}}(undef, 1)
            n_pattern = Vector{Int64}(undef, 1)
            for r = one(Float64):1000
                respn = Vector{Int64}(undef, i_index)
                unif = rand(Distributions.Uniform(0, 1), i_index)#5
                samplei = sample(collect(1:i_index), i_index, replace = false)
                for i = one(Float64):i_index#7
                    if unif[samplei[i]] < pr[samplei[i], n]
                        respn[i] = one(Float64)
                    else
                        respn[i] = 0
                    end
                end
                if r == one(Float64)
                    patterns[1] = respn
                    n_pattern[1] = one(Float64)
                else
                    println(size(patterns, 1))
                    corr = 0
                    for pat = one(Float64):size(patterns, 1)
                        if patterns[pat] == respn
                            corr = pat
                        end
                    end
                    if corr == 0
                        push!(patterns, respn)
                        n_pattern = hcat(n_pattern, 1)
                    else
                        n_pattern[corr] = n_pattern[corr] + 1
                    end
                end
            end
            n_patterns = size(patterns, 1)
            prob_patterns = n_pattern ./ 1000
            resp[i_index, n] .= sample(patterns, Distributions.pweights(prob_patterns), 1)
            println("end person ", n)
        end

    end
    return resp::Matrix{Float64}
end
