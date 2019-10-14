function cutR(x; start="minimum", stop="maximum", nBins=2, returnBreaks=true,returnMidPts=false)
	if (start=="minimum") start=minimum(x) end
	if (stop=="maximum") stop=maximum(x) end
	bw=(stop-start)/(nBins-1)
	midPts=zeros(nBins)
	for i=1:nBins
		midPts[i]=start+(i-1)*bw
	end
	breaks=collect(range(start-(bw/2); length=nBins+1,stop=stop+(bw/2)))
	y=zeros(size(x,1))
	for j in 1:size(x,1)
		for i in 1:nBins
			if (x[j] >= breaks[i]) && (x[j] < breaks[i+1])
				y[j]=i
			end
			if i==nBins && x[j] == breaks[i+1]
				y[j]=i
			end
		end
	end
	if (returnBreaks==true || returnMidPts==true)
		if returnMidPts==false
			return (Int.(y),breaks)
		elseif returnBreaks==false
			return (Int.(y),midPts)
		else
			return (Int.(y),breaks,midPts)
		end
	else
		return Int.(y)
	end
end
function subsetData(dt::data,subset::Vector{Int64},NumberOfTests::Int64,LengthOfTests::Int64,est::block,sd::block) #method= Booklet(subset=Scalar) or Students(subset=array), su
	N=size(dt.responses,1)
	subset=sort(subset)
	newResponses=dt.responses[:,subset]
	newDesign=dt.design[:,subset]
	est.pars=est.pars[:,findall(sum(newDesign, dims=2).>=1)]
	est.latentVals=est.latentVals[subset,:]
	newI,newN=size(newResponses)
	if size(sd.pars,1)>0
		newSd=sd
		newSd.pars=sd.pars[:,findall(sum(newDesign, dims=2).>=1)]
		newSd.latentVals=sd.latentVals[subset,:]
		return newN::Int64,newI::Int64,newResponses::Matrix{Float64},newDesign::Matrix{Float64},est::block,newSd::block
	else
		return newN::Int64,newI::Int64,newResponses::Matrix{Float64},newDesign::Matrix{Float64},est::block
	end
end
