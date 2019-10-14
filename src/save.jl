function savePool(pars::Matrix{Float64},nPar::Int64, folder)
	#names of parameters
	nTotPars,nItems=size(pars)
	parNames=Vector{String}(undef,nTotPars)
	parNames[1]="b"
	for p=2:nTotPars
		parNames[p]=string("a_",p-1)
	end
	if nPar==3
		parNames[end]="c"
	end
	simPool = DataFrame(id=1:nItems)
	for p=1:nTotPars
		DataFrames.insertcols!(simPool, p+1, Symbol(parNames[p]) => pars[p,:])
	end
	CSV.write(string(folder,"/simPool.csv"),simPool)
end

function saveLatentValues(latentVals::Matrix{Float64},folder::String)
	nStud,nLatent=size(latentVals)
	latNames=Vector{String}(undef,nLatent)
	latNames[1]="intercept"
	for l=2:nLatent
		latNames[l]=string("t_",l-1)
	end
	simLatent = DataFrame(id=1:nStud)
	for l=1:nLatent
		DataFrames.insertcols!(simLatent,l+1, Symbol(latNames[l]) => latentVals[:,l])
	end
	CSV.write(string(folder,"/simLatentVals.csv"),simLatent)
end
