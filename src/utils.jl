function exp_c(x::Float64)
	ccall(:exp, Float64, (Float64,), x)
end
function log_c(x::Float64)
	ccall(:log, Float64, (Float64,), x)
end
function log1p_c(x::Float64)
	ccall(:log1p, Float64, (Float64,), x)
end

function sig_c(x::Float64)
	1/(1+exp_c(-x))
end
function sig_cplus(x::Float64)
	1/(1+exp_c(x))
end
# function sig_c(x::Float64)
# 	(ccall(:tanh, Float64, (Float64,), x)+1)/2
# end

function compLh(likelihood::Matrix{Float64},N::Int64,K::Int64,iIndex::Vector{Vector{Int64}},r::Matrix{Float64},phi::Matrix{Float64})
	ephizero=sig_cplus.(phi)
	ephione=sig_c.(phi)
	post_n=zeros(Float64,K)
	for n=1:N
		for k=1:K
			post_k=one(Float64)
			for i in iIndex[n]
				if r[i,n]>0
					post_k*=ephione[k,i]
				else
					post_k*=ephizero[k,i]
				end
			end
				likelihood[n,k]=copy(post_k)
		end
	end
	return likelihood::Matrix{Float64}
end

function compPost(post::Matrix{Float64},lh::Matrix{Float64},N::Int64,K::Int64,I::Int64,iIndex::Vector{Vector{Int64}},r::Matrix{Float64},Wk::Vector{Float64},phi::Matrix{Float64})
	#ephi=exp_c.(phi)
	ephizero=sig_cplus.(phi)
	ephione=sig_c.(phi)
	post_n=zeros(Float64,K)
	Distributed.@sync Distributed.@distributed for n=1:N
		for k=1:K
			post_k=one(Float64)
			for i in iIndex[n]
				if r[i,n]>0
					post_k*=ephione[k,i]
				else
					post_k*=ephizero[k,i]
				end
			end
				post_n[k]=copy(post_k)
		end
		# post_n=pmap(1:K) do k
		# 	prod(pmap(iIndex[n]) do i
		# 		(ephi[i,k]^r[i,n])/(1+ephi[i,k])
		# 	end;)
		# end;
		lh[n,:]=copy(post_n)
		post_n=post_n.*Wk
		normalizer=sum(post_n)
		if normalizer>typemin(Float64)
			post_n=post_n./normalizer
		end
		post[n,:]=copy(post_n)
	end
	return post::Matrix{Float64}, lh::Matrix{Float64}
end

function compPostSimp(post::Matrix{Float64},N::Int64,K::Int64,I::Int64,iIndex::Vector{Vector{Int64}},r::Matrix{Float64},Wk::Vector{Float64},phi::Matrix{Float64})
	ephizero=sig_cplus.(phi)
	ephione=sig_c.(phi)
	post_n=zeros(Float64,K)
	for n=1:N
		for k=1:K
			post_k=one(Float64)
			for i in iIndex[n]
				if r[i,n]>0
					post_k*=ephione[k,i]
				else
					post_k*=ephizero[k,i]
				end
			end
				post_n[k]=copy(post_k)
		end
		post_n=(post_n.*Wk) #modify with firstLatent
		exp_cd=sum(post_n)
		if exp_cd>typemin(Float64)
			post_n=post_n./exp_cd
		end
		post[n,:]=copy(post_n)
	end
	return post::Matrix{Float64}
end


function probFun(pars::Matrix{Float64},latentVals::Matrix{Float64},nItems::Int64,nStud::Int64)
	p=zeros(Float64,nItems,nStud)
	LinearAlgebra.BLAS.gemm!('T', 'T', one(Float64), pars,  latentVals,  zero(Float64), p)# IxN
	return 1 ./(1 .+ exp_c.(.-p)) #IxN
	#probs=ze   ros(Float64,nIems,max(nPars,nPars-1))
	#probs=ccall((:prob,"ccode.o"),Matrix{Float64}, (Ptr{Cdouble},Ptr{Cdouble},Cint,Cint),pars,latents,nItems,max(nPars,nPars-1))
end


#RMSE for vectors
function RMSE(est::Vector{Float64},real::Vector{Float64})
	N=size(est,1)
	sqrt(mean((est.-real).^2))
end
#RMSE for matrices
function RMSE(est::SharedArray{Float64,2},real::Matrix{Float64})#each column a parameter
	N,K=size(est)
	ret=zeros(N)
	for n=1:N
		ret[n]=sqrt(mean((est[n,:]-real[n,:]).^2))
	end
	ret::Vector{Float64}
end
function RMSE(est::Array{Float64,2},real::Matrix{Float64})#by row we have different parameters
	N,K=size(est)
	ret=zeros(N)
	for n=1:N
		ret[n]=sqrt(mean((est[n,:]-real[n,:]).^2))
	end
	ret::Vector{Float64}
end
#BIAS for vectors
function BIAS(est::Matrix{Float64},real::Vector{Float64})
	K,N=size(est)
	ret=zeros(N)
	for n=1:N
		ret[n]=mean(est[:,n])-real[n]
	end
	ret::Vector{Float64}
end

function RMSE(est::Matrix{Float64},real::Vector{Float64})
	K,N=size(est)
	ret=zeros(N)
	for n=1:N
		ret[n]=sqrt(mean((est[:,n].-real[n]).^2))
	end
	ret::Vector{Float64}
end
