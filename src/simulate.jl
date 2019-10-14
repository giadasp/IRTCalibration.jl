function rescaleValues(vals::Vector{Float64},metric::Vector{Float64})
	return (vals.-(mean(vals)-metric[1]))./(std(vals)/metric[2])
end#export these
function rescaleLatentValues(latentVals::Matrix{Float64},metric::Vector{Vector{Float64}})
	nLatentInt=size(latentVals,2)
	for l=2:nLatentInt
		latentVals[:,l]=rescaleValues(latentVals[:,l],metric[l-1])
	end
	return latentVals
end
function simulateData(irt::IRT,latentMetric::Vector{Vector{Float64}},latentDist::Vector{<:Distribution},parsDist::Vector{<:Distribution})
	simPars=Matrix{Float64}(undef,irt.nLatent+1,irt.nItems)
	simLatentVector=Vector{latent}(undef,irt.nLatent)
	simLatentVals=Matrix{Float64}(undef,irt.N,irt.nLatent+1)
	simLatentVals[:,1]=ones(irt.N)
	if irt.nPar>1 #intercept
		simPars[1,:]=rand(parsDist[1],irt.nItems) #easiness
		for l=1:(irt.nLatent) #discriminations
			simPars[l+1,:]=rand(parsDist[l+min(irt.nPar-1)],irt.nItems)
			simLatentVector[l]=latent(latentDist[l],latentMetric[l])
			simLatentVals[:,l+1]=rand(latentDist[l],irt.N)
		end
	else #no easiness
		for l=1:(irt.nLatent) #discriminations
			simPars[l+1,:]=ones(irt.nItems)
			simLatentVector[l]=latent(latentDist[l],latentMetric[l])
			simLatentVals[:,l+1]=rand(latentDist[l],irt.N)
		end
	end
	if irt.nPar>2 #guessing
		simPars[end,:]=ones(parsDist[end],irt.nItems).*0.5
	end
	return block(simPars,simLatentVals,simLatentVector)
end

function genResp(f::Vector{Float64},pars::Matrix{Float64},latentVals::Matrix{Float64},design::Matrix{Float64};
	model="2PL",
	method="classicUniform"
	)
	nItems = size(pars,2)#3
	N, L = size(latentVals)
	L=L-1
	nindex=Array{Array{Int64,1},1}(undef,nItems)
	iindex=Array{Array{Int64,1},1}(undef,N)
	for n=1:N
		iindex[n]=findall(design[:,n].==1)
	end
	for i=1:nItems
		nindex[i]=findall(design[i,:].==1)
	end
	pr=probFun(pars,latentVals,nItems,N)
	lp=log_c.(pr)
	lq=log_c.(1 .- pr)
	resp=zeros(Float64,nItems,N)
	if method=="cumulatedPersons"
		@fastmath @inbounds  for i=1:nItems
			gapScore=3
			while gapScore>=2
				p2 = pr[i,:]
				p2 = hcat((1 .- p2) , p2)#2
				unif=rand(Uniform(0,1),N)#5
				n=one(Float64)#6
				while n<=N#7
					csum=p2[n,1]#8
					cat=0#9
					if design[i,n]==0
						resp[i,n]=0
						n=n+1
					else
						while csum<unif[n]
							cat=cat+1
							if (cat == 2) break end
							csum = csum + p2[n,cat+1]
						end
						resp[i,n]=cat
						n=n+1
					end
				end
				gapScore=abs(sum(resp[i,:])-sum(p[i,n] for n in nindex[i]))
			end
		end
	end
	if method=="cumulatedItems"
		@fastmath @inbounds  for n=1:N#4
			#gapScore=3
			#while gapScore>=2
			p2 = p[:,n]
			p2 = hcat((1 .- p2) , p2)#2
			unif=rand(Uniform(0,1),nItems)#5
			samplei=sample(collect(1:nItems),nItems,replace=false)
			i=one(Float64)#6
			while i<=nItems#7
				csum=p2[samplei[i],1]#8
				cat=0#9
				if design[samplei[i],n]==0
					resp[samplei[i],n]=0#missing
					i=i+1
				else
					while csum<unif[samplei[i]]
						cat=cat+1
						if (cat == 2) break end
						csum = csum + p2[samplei[i], cat+1]
					end
					resp[samplei[i],n]=cat
					i=i+1
				end
			end
			#gapScore=abs(sum(skipmissing(resp[:,n]))-sum(p[i,n] for i in iindex[n]))
			#if gapScore>=0.5
			#	println("person ",n," gap=",gapScore)
			#end
			#end
		end
	end
	if method=="classicUniform"
		for n=1:N#4
			#gapScore=2
			#while gapScore>=0.5
			unif=rand(Distributions.Uniform(0,1),nItems)#5
			samplei=sample(collect(1:nItems),nItems,replace=false)
			i=1#6
			while i<=nItems#7
				if design[samplei[i],n]==0
					resp[samplei[i],n]=0#missing
					i=i+1
				else
					if unif[samplei[i]]<pr[samplei[i],n]
						resp[samplei[i],n]=one(Float64)
					else
						resp[samplei[i],n]=0
					end
					i=i+1
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
	# 		iIndex=iindex[n]
	# 		p=[c[i]+((1-c[i])*(1 / (1 + exp_c(-a[i]*(θ[n]-b[i]))))) for i in iIndex]
	# 		p[p.==0].=0.00001
	# 		m=Model(solver=CplexSolver(CPX_PARAM_PREIND=0,CPX_PARAM_MIPEMPHASIS=one(Float64)))
	# 		@variable(m, x[i=one(Float64):size(iIndex,1)], Bin)
	# 		@objective(m, Max, sum((x[i]*log_c(p[i]) + ((f[i] - x[i]) * log_c(1-p[i]))) for  i=one(Float64):size(iIndex,1)))
	# 		@constraint(m, sum(x[i] for i=one(Float64):size(iIndex,1))-sum(p) <=+1)
	# 		@constraint(m, sum(x[i] for i=one(Float64):size(iIndex,1))-sum(p) >=-1)
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
	if method=="cumItemsPattern"
		for n=1:N#4+
			println("start person ", n)
			iIndex=iindex[n]
			iIndex=size(iIndex,1)
			p2 = pr[iIndex,n]
			p2 = hcat((1 .- p2) , p2)#2
			patterns=Vector{Vector{Int64}}(undef,1)
			nPattern=Vector{Int64}(undef,1)
			for r=one(Float64):1000
				respn=Vector{Int64}(undef,iIndex)
				unif=rand(Uniform(0,1),iIndex)#5
				samplei=sample(collect(1:iIndex),iIndex,replace=false)
				for i=one(Float64):iIndex
					csum=p2[samplei[i],1]#8
					cat=0#9
					while csum<unif[samplei[i]]
						cat=cat+1
						if (cat == 2) break end
						csum = csum + p2[samplei[i], cat+1]
					end
					respn[i]=cat
				end
				if r==1
					patterns[1]=respn
					nPattern[1]=one(Float64)
				else
					println(size(patterns,1))
					corr=0
					for pat=one(Float64):size(patterns,1)
						if patterns[pat]==respn
							corr=pat
						end
					end
					if corr==0
						push!(patterns,respn)
						nPattern=hcat(nPattern,1)
					else
						nPattern[corr]=nPattern[corr]+1
					end
				end

			end
			nPatterns=size(patterns,1)
			probPatterns=nPattern./1000
			println(probPatterns)
			resp[iIndex,n].=sample(patterns,pweights(probPatterns),1)
			println("end person ", n)
		end
	end
	if method=="classicUniformPattern"
		for n=1:N#4+
			println("start person ", n)
			iIndex=iindex[n]
			iIndex=size(iIndex,1)
			patterns=Vector{Vector{Int64}}(undef,1)
			nPattern=Vector{Int64}(undef,1)
			for r=one(Float64):1000
				respn=Vector{Int64}(undef,iIndex)
				unif=rand(Uniform(0,1),iIndex)#5
				samplei=sample(collect(1:iIndex),iIndex,replace=false)
				for  i=one(Float64):iIndex#7
					if unif[samplei[i]]<p[samplei[i],n]
						respn[i]=one(Float64)
					else
						respn[i]=0
					end
				end
				if r==one(Float64)
					patterns[1]=respn
					nPattern[1]=one(Float64)
				else
					println(size(patterns,1))
					corr=0
					for pat=one(Float64):size(patterns,1)
						if patterns[pat]==respn
							corr=pat
						end
					end
					if corr==0
						push!(patterns,respn)
						nPattern=hcat(nPattern,1)
					else
						nPattern[corr]=nPattern[corr]+1
					end
				end
			end
			nPatterns=size(patterns,1)
			probPatterns=nPattern./1000
			resp[iIndex,n].=sample(patterns,pweights(probPatterns),1)
			println("end person ", n)
		end

	end
	return resp::Matrix{Float64}
end
