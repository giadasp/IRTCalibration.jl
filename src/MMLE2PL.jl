function calibrate(mdl::latentModel;debug=true)
	if mdl.irt.nPar>1
		firstLatent=2
	else
		firstLatent=1
	end
	nItems=mdl.irt.nItems
	nPar=mdl.irt.nPar
	nLatent=mdl.irt.nLatent
	nTotPar=nPar-1+nLatent
	N=mdl.irt.N
	T=mdl.dt.T
	n=mdl.dt.n
	K=mdl.eo.K
	X=ones(Float64,K,nLatent+1)
	W=zeros(Float64,K,nLatent+1) .+(1/K)
	for l=1:nLatent
		X[:,l+1]=Distributions.support(mdl.estimates.latents[l].dist)
		W[:,l+1]=Distributions.probs(mdl.estimates.latents[l].dist)
	end
	Wk=W[:,firstLatent]
	Xk=X[:,firstLatent]
	startPars=copy(mdl.estimates.pars)
	if mdl.bs.BS
		startEst=copy(mdl.estimates)
		#takes only the first latent
		(bins,none)=cutR(mdl.estimates.latentVals[:,firstLatent];start=mdl.bds.minLatent[1], stop=mdl.bds.maxLatent[1], nBins=K, returnBreaks=false,returnMidPts=true)
		pθ=Wk[bins]
		pθ=pθ./sum(pθ)
		BSPar=Vector{DataFrame}(undef,nTotPar) #nTotPar x R+1 (id + BS)
		for p=1:nTotPar
			BSPar[p]=DataFrame(id=collect(1:nItems))
			if p==1
				name="b"
			elseif nPar==2
				name=string("a_",p-1)
			elseif p==nTotPar
				name=string("c")
			else
				name=string("a_",p-1)
			end
			for r=1:mdl.bs.R
				DataFrames.insertcols!(BSPar[p],r+1, Symbol(string(name,"_",r)) => zeros(Float64,nItems))
			end
		end
		BSLatentVals=Vector{DataFrame}(undef,nLatent) #nTotPar x R+1 (id + BS)
		for l=1:nLatent
			BSLatentVals[l]=DataFrame(id=collect(1:N))
			name=string("theta_",l)
			for r=1:mdl.bs.R
				DataFrames.insertcols!(BSLatentVals[l],r+1, Symbol(string(name,"_",r)) => zeros(Float64,N))
			end
		end
	end
	s=1
	if mdl.bs.BS==false
		R=1
	else
		R=mdl.bs.R
	end

	for r=1:R
		nIndex=Array{Array{Int64,1},1}(undef,nItems)
		iIndex=Array{Array{Int64,1},1}(undef,N)
		for n=1:N
			iIndex[n]=findall(mdl.dt.design[:,n].==1.0)
			if n<=nItems
				nIndex[n]=findall(mdl.dt.design[n,:].==1.0)
			end
		end #15ms
		if mdl.bs.BS
			mdl.estimates=copy(startEst)
			if mdl.bs.type=="parametric"
				if mdl.dt.unbalanced
					nsample=sample(collect(1:N),StatsBase.ProbabilityWeights(pθ), Int(floor(mdl.bs.sampleFrac*N)), replace=true)
					while size(unique(vcat([iIndex[n] for n in nsample]...)),1)<nItems
						#nsample=rand(Distributions.DiscreteNonParametric(collect(1:N), pθ),Int(floor(mdl.sampleFrac*N)))
						nsample=sample(collect(1:N),StatsBase.ProbabilityWeights(pθ), Int(floor(mdl.bs.sampleFrac*N)), replace=true)
					end
					println(size(nsample,1))
				else
					tsamp=Vector{Vector{Int64}}(undef,T)
					for t=1:T
						nt=Int.(collect(0:(N/(T)))[1:end-1].*T.+t) #subjects that have take the test t, sample among these
						bins_t=bins[nt]
						pop=sort!(rand(Distributions.DiscreteNonParametric(collect(1:K),Wk),size(nt,1)))
						samplet=Int64[]
						for k=1:K
							if sum(pop.==k)>0 && sum(bins_t.==k)>0
								s=sample(nt[bins_t.==k],sum(pop.==k), replace=true)
								samplet=vcat(samplet,s)
							end
						end
						#add more samples if they are not enough
						while size(samplet,1)!=size(pop,1)
							k=sample(collect(1:K))
							if sum(pop.==k)>0 && sum(bins_t.==k)>0
								samplet=vcat(samplet,sample(nt[bins_t.==k],1))
							end
						end
						tsamp[t]=copy(samplet)
					end
					nsample=sort(vcat(tsamp...))
				end
			else #non-parametric
				if mdl.dt.unbalanced
					nsample=sample(collect(1:N), Int(ceil(mdl.bs.sampleFrac*N)), replace=true)
					while size(unique(vcat([iIndex[n] for n in nsample]...)),1)<nItems
						nsample=sample(collect(1:N), Int(floor(mdl.bs.sampleFrac*N)), replace=true)
					end
				else
					tsamp=Vector{Vector{Int64}}(undef,T)
					for t=1:T
						nt=Int.(collect(0:(N/(T)))[1:end-1].*T.+t)
						tsamp[t]=sample(nt,Int(floor(mdl.bs.sampleFrac*N/T)),replace=true)
					end
					nsample=sort(vcat(tsamp...))
				end
			end
			if size(mdl.simData.pars,1)>0
				(newN,newI,newResponses,newDesign,newEst,newSd)=subsetData(mdl.dt,nsample,T,n,mdl.estimates,mdl.simData)
			else
				(newN,newI,newResponses,newDesign,newEst)=subsetData(mdl.dt,nsample,T,n,mdl.estimates,mdl.simData)
			end
		else #no bs
			if size(mdl.simData.pars,1)>0
				(newN,newI,newResponses,newDesign,newEst,newSd)=(N,nItems,mdl.dt.responses,mdl.dt.design,mdl.estimates,mdl.simData)
			else
				(newN,newI,newResponses,newDesign,newEst)=(N,nItems,mdl.dt.responses,mdl.dt.design,mdl.estimates)
			end
		end
		#index of missings
		nIndex=Array{Array{Int64,1},1}(undef,newI)
		iIndex=Array{Array{Int64,1},1}(undef,newN)
		for n=1:newN
			iIndex[n]=findall(newDesign[:,n].==1.0)
			if n<=newI
				nIndex[n]=findall(newDesign[n,:].==1.0)
			end
		end
		########################################################################
		### initialize variables
		########################################################################
		startTime=time()
		newPars=copy(newEst.pars)
		oldPars=Vector{Matrix{Float64}}(undef,2)
		oldPars[1]=copy(newPars)
		# oldPars[2]=copy(newPars)
		xGap=Inf
		xGapOld=ones(Float64,7).*Inf
		Wk=W[:,firstLatent]
		Xk=X[:,firstLatent]
		newLh=-Inf
		oldLh=ones(Float64,3).*(-Inf)
		oldLatentVals=copy(newEst.latentVals)
		newLatentVals=copy(oldLatentVals)
		###############################################################################
		### optimize (item parameters)
		###############################################################################
		BestLh=-Inf
		newTime=time()
		oldTime=time()
		fGap=Inf
		nloptalg=Symbol("LD_SLSQP")
		limES=0.5
		s=1
		endOfWhile=0
		startTime=time()
		phi=X*newPars #K x I
		posterior=zeros(Float64,newN,K)
		lhMat=similar(posterior)
		oneoverN=fill(1/newN,newN)
		newLatentVals=zeros(newN,nLatent+1)
		#r=zeros(Float64,nPar,newI)
		println(newPars[1,1])
		while endOfWhile<1
			################################################################
			####					ACCELLERATE
			################################################################
			 #if s>1
			# 	if exp(oldLh[1]-newLh)>0.1
			# 		d2=oldPars[1]-oldPars[2]
			# 		d1=newPars-oldPars[1]
			# 		d1d2=d1-d2
			# 		ratio=sqrt(sum(d1.^2) / sum(d1d2.^2))
			# 		accel=clamp(1-ratio,-5.0,Inf)
			# 		newPars=((1-accel).*newPars)+(accel.*oldPars[1])#+(1-exp(newLh-oldLh[1])).*newPars
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

			if mdl.eo.first=="items"
				################################################################
				####					MStep
				################################################################
				#before_time=time()
				newPars, phi =maxLHMMLE(newPars,phi,posterior,iIndex,newDesign,X,Wk,newResponses,mdl.io,mdl.bds)
				lhMat=compLh(lhMat,newN,K,iIndex,newResponses,phi)
				newLh=sum(log_c.(LinearAlgebra.BLAS.gemv('N', one(Float64), lhMat, Wk)))
				#println("time elapsed for Mstep intOpt ",time()-before_time)
				####################################################################
			else
				#theta
				#before_time=time()
				################################################################
				####					RESCALE
				################################################################
				if mdl.eo.denType=="EH" && (s%mdl.eo.intW==0 && s<=mdl.eo.minMaxW[2] && s>=mdl.eo.minMaxW[1] )  && mdl.bs.BS==false# && xGap>0.5 && mdl.bs.BS==false
					posterior, newLh= compPost(posterior,lhMat,newN,K,newI,iIndex,newResponses,Wk,phi)
					Wk=LinearAlgebra.BLAS.gemv('T', one(Float64), posterior, oneoverN) #if Wk depends only on the likelihoods
					observed=[LinearAlgebra.dot(Wk,Xk),sqrt(LinearAlgebra.dot(Wk,Xk.^2))]
					observed=[observed[1]-mdl.estimates.latents[1].metric[1],observed[2]/mdl.estimates.latents[1].metric[2]]
					#check mean
					#if  (abs(observed[1])>1e-3 ) || observed[2]<0.99
						Xk2, Wk2=myRescale(Xk,Wk,mdl.estimates.latents[1].metric,observed)
						Wk=cubicSplineInt(Xk,Xk2,Wk2)
					#end
					newLh=sum(log_c.(LinearAlgebra.BLAS.gemv('N', one(Float64), lhMat, Wk)))
				elseif typeof(mdl.eo.denType)==Distribution
					if s>=1
						for l=1:nLatent
							(bins,X[:,l+1])=cutR(newLatentVals[:,l+1];star=mdl.bds.minLatent[l], stop=mdl.bds.maxLatent[l], nBins=K+(toadd*2)-1, returnBreaks=false,returnMidPts=true)
							X[:,l+1]=X[(toadd+1):K+toadd,l+1]
							W[:,l+1]=pdf.(denType,X[:,l+1])
							W[:,l+1]=W[:,l+1]./sum(Wk)
						end
					end
				end
				#println("time elapsed for Rescale intOpt ",time()-before_time)
				################################################################
			end

			############################################################
			####					SAVE
			############################################################
			#lh
			oldLh[3]=oldLh[2]
			oldLh[2]=oldLh[1]
			oldLh[1]=copy(newLh)
			#pars
			deltaPars=(newPars-oldPars[1])./oldPars[1]
			oldPars[2]=oldPars[1]
			oldPars[1]=copy(newPars)
			#xGap
			xGap=maximum(abs.(deltaPars))
			bestxGap=min(xGap,xGapOld[1])
			xGapOld[6]=copy(xGapOld[5])
			xGapOld[5]=copy(xGapOld[4])
			xGapOld[4]=copy(xGapOld[3])
			xGapOld[3]=copy(xGapOld[2])
			xGapOld[2]=copy(xGapOld[1])
			xGapOld[1]=copy(bestxGap)
			oldLatentVals=copy(newLatentVals)
			############################################################

			#SECOND STEP
			if mdl.eo.first=="items"
				before_time=time()
				################################################################
				####					RESCALE
				################################################################
				if mdl.eo.denType=="EH" && (s%mdl.eo.intW==0 && s<=mdl.eo.minMaxW[2] && s>=mdl.eo.minMaxW[1])  && mdl.bs.BS==false
					posterior, newLh= compPost(posterior,lhMat,newN,K,newI,iIndex,newResponses,Wk,phi)
					Wk=LinearAlgebra.BLAS.gemv('T', one(Float64), posterior, oneoverN) #if Wk depends only on the likelihoods
					observed=[LinearAlgebra.dot(Wk,Xk),sqrt(LinearAlgebra.dot(Wk,Xk.^2))]
					observed=[observed[1]-mdl.estimates.latents[1].metric[1],observed[2]/mdl.estimates.latents[1].metric[2]]
					#check mean
					#if  (abs(observed[1])>1e-3 ) || observed[2]<0.99
						Xk2, Wk2=myRescale(Xk,Wk,mdl.estimates.latents[1].metric,observed)
						Wk=cubicSplineInt(Xk,Xk2,Wk2)
					#end
					newLh=sum(log_c.(LinearAlgebra.BLAS.gemv('N', one(Float64), lhMat, Wk)))
				elseif typeof(mdl.eo.denType)==Distribution
					if s>=1
						for l=1:nLatent
							(bins,X[:,l+1])=cutR(newLatentVals[:,l+1];star=mdl.bds.minLatent[l], stop=mdl.bds.maxLatent[l], nBins=K+(toadd*2)-1, returnBreaks=false,returnMidPts=true)
							X[:,l+1]=X[(toadd+1):K+toadd,l+1]
							W[:,l+1]=pdf.(denType,X[:,l+1])
							W[:,l+1]=W[:,l+1]./sum(Wk)
						end
					end
				end
				println("time elapsed for rescale intOpt ",time()-before_time)
				################################################################
			else
				################################################################
				####					MStep
				################################################################
				before_time=time()
				newPars,  phi =maxLHMMLE(newPars,phi,posterior,iIndex,newDesign,X,Wk,newResponses,mdl.io,mdl.bds)
				lhMat=compLh(lhMat,newN,K,iIndex,newResponses,phi)
				newLh=sum(log_c.(LinearAlgebra.BLAS.gemv('N', one(Float64), lhMat, Wk)))
				#println("time elapsed for Mstep intOpt",time()-before_time)

				################################################################
			end
			# if size(simData.θ,1)>0 && size(simData.pool,1)>0
			 #println("RMSE for pars is ",RMSE(newPars,newSd.pars))
			#println("RMSE for latents is ",RMSE(newLatentVals',newSd.latentVals'))
			#println("RMSE for θ is ",RMSE(newLatentVals,newSimθ))
			# end
			if debug
			println("end of iteration  #",s)
			println("newlikelihood is ",newLh)
		    end
			newTime=time()
			####################################################################
			#                           CHECK CONVERGENCE
			####################################################################
			if (s >= mdl.eo.maxIter)
				println("maxIter reached after ", newTime-oldTime," and ",Int(s)," iterations")
				endOfWhile=1
				# ItemPars=DataFrame(a=new_a,b=new_b)
				# Bestθ=copy(newLatentVals)
			end
			if newTime-oldTime>mdl.eo.timeLimit
				println("timeLimit reached after ", newTime-oldTime," and ",Int(s)," iterations")
				endOfWhile=1
			end
			fGap=abs(newLh-oldLh[1])/oldLh[1]
			if fGap<mdl.eo.lTolRel && fGap>=0
				println("f ToL reached after ", newTime-oldTime," and ",Int(s)," iterations")
				endOfWhile=1
			end
			if s>3
				deltaPars=(newPars-oldPars[1])./oldPars[1]
				xGap=maximum(abs.(deltaPars))
				bestxGap=min(xGap,xGapOld[1])
				if debug
				println("Max-change is ",xGap)
				end
				if xGap<=mdl.eo.xTolRel
					println("X ToL reached after ", newTime-oldTime," and ",Int(s)," iterations")
					endOfWhile=1
				else
					if s>20
						if !(newLh>oldLh[1] && oldLh[1]>oldLh[2] && oldLh[2]>oldLh[3])
							if xGap<0.1 && xGapOld[1]==bestxGap && xGapOld[1]==xGapOld[2] && xGapOld[3]==xGapOld[2] && xGapOld[3]==xGapOld[4] && xGapOld[4]==xGapOld[5] && xGapOld[6]==xGapOld[5]
								endOfWhile=1
								println("No better result can be obtained")
							end
						end
					end
				end
			end
			if endOfWhile==1
				posterior= compPostSimp(posterior,newN,K,newI,iIndex,newResponses,Wk,phi)
				newLatentVals=(posterior*X)./(posterior*ones(K,nLatent+1))
			end
			s=s+1
			####################################################################

		end

		if mdl.bs.BS
			isample=findall(sum(newDesign,dims=2).>0)
			for p=1:nTotPar
				BSPar[p][isample,r+1].=newPars[p,:]
			end
			for l=1:nLatent
				BSLatentVals[l][nsample,r+1].=newLatentVals[:,l]
			end
		end
		mdl.prf.time=time()-startTime
		mdl.estimates.pars=newPars
		mdl.estimates.latentVals=newLatentVals
		mdl.estimates.latents[1].dist=Distributions.DiscreteNonParametric(Xk,Wk)
		mdl.prf.nIter=s-1
		println("end of ",r, " bs replication")
	end
	if mdl.bs.BS
	JLD2.@save "pars.jld2" BSPar
	JLD2.@save "latentVals.jld2" BSLatentVals
	end

	return mdl
	# CSV.write("BS 1/estPoolGlobal.csv", ItemPars)
	# writedlm("BS 1/estAbilitiesGlobal.csv",Bestθ,'\t')
end
