 __precompile__()
module IRTCalibration
import CSV
import DataFrames
import Distributions
import Distributed
import DelimitedFiles
import LinearAlgebra
import PGFPlots
import StatsBase
import LaTeXStrings.@L_str
import Interpolations
import NLopt
import JuMP
import CPLEX
#import Dierckx
import JLD2
import SharedArrays
const DataFrame=DataFrames.DataFrame
const Distribution=Distributions.Distribution
const readdlm=DelimitedFiles.readdlm
const writedlm=DelimitedFiles.writedlm
const SharedArray=SharedArrays.SharedArray
const std=StatsBase.std
const sample=StatsBase.sample


function mean(x::Vector{Float64})
	n=size(x,1)
	if n>0
		 return sum(x)/n
	 else
		 return 0
	 end
 end

 mutable struct performance
 	time::Float64#Dates.Millisecond
 	nIter::Int64
 	performance()=new(0.0,0)
 	performance(time,nIter)=new(time,nIter)
 end

 mutable struct latent
 	dist::Distributions.Distribution
 	metric::Vector{Float64}
 	latent()=new(Normal(0,1),[zero(Float64),one(Float64)])
 	latent(dist,metric)=new(dist,metric)
 end

 mutable struct IRT
 	model::String
 	nPar::Int64
 	nLatent::Int64
 	nItems::Int64
 	N::Int64 #n test takers
 	IRT()=new("2PL",2,1,0,0)
 	IRT(model,nPar,nLatent,nItems,N)=new(model,nPar,nLatent,nItems,N)
 end

 mutable struct data
 	responses::Matrix{Float64}
 	design::Matrix{Float64}
 	unbalanced::Bool
 	T::Int64
 	n::Int64
 	f::Vector{Float64}
 	data()=new(zeros(Float64,0,0),Matrix{Float64}(undef,0,0),true,0,0,zeros(1))
 	data(responses,design,unbalanced,T,n,f)=new(responsesMat,responsesVec,design,unbalanced,T,n,f) #no pattern mode
 end

 mutable struct block
 	pars::Matrix{Float64}
 	latentVals::Matrix{Float64}
 	latents::Vector{latent}
 	block()=new(Matrix{Float64}(undef,0,0),Matrix{Float64}(undef,0,0),[latent()])
 	block(pars,latentVals,latents)=new(pars,latentVals,latents)
 end

 mutable struct bootstrap
 	BS::Bool #true or false
 	R::Int64 #number of replications
 	sampleFrac::Float64
 	type::String #parametric or nonParametric
 	nbins::Int64
 	bootstrap()=new(false,1,2/3,"parametric",50)
 	bootstrap(BS,R,sampleFrac,type,nbins)=new(BS,R,sampleFrac,type,nbins)
 end

  mutable struct intOpt #if you want to specify a termination parameter you have to specify all of them
 	solver::String
 	xTolRel::Float64
 	fTolRel::Float64
 	timeLimit::Int64
 	intOpt()=new("NLopt",1e-4,1e-5,10)
 	intOpt(solver,xTolRel,fTolRel,timeLimit)=new(solver,xTolRel,fTolRel,timeLimit)
 end

 mutable struct extOpt #if you want to specify a termination parameter you have to specify all of them
 	method::String# =optima weighted sum #can be: ["OWS" , "WLE" , "JML"]
 	denType::Union{Distribution,String}
 	K::Int64 #theta points
 	intW::Int64 #can be: ["1" , "1/s"  , "1/sqrt(s)"]
	minMaxW::Vector{Int64}
 	first::String #can be: ["theta", "items"]
 	lTolRel::Float64
 	xTolRel::Float64
 	timeLimit::Int64
 	maxIter::Int64
 	extOpt()=new("OWS",Normal(0,1),21,3,[9,15],"theta",1e-20,1e-5,1000,1000)
 	extOpt(method,denType,K,intW,minMaxW,first,lTolRel,xTolRel,timeLimit,maxIter)=new(method,denType,K,intW,minMaxW,first,lTolRel,xTolRel,timeLimit,maxIter)
 end

  mutable struct bounds
 	minPars::Vector{Float64}
 	maxPars::Vector{Float64}
 	minLatent::Vector{Float64}
 	maxLatent::Vector{Float64}
 	bounds()=new([0.00001,-10.0],[10.0,10.0],[-10.0],[10.0]) #the last vectors must be the same size of the IRT , es. 2PL, size=2
 	bounds(minPars,maxPars,minLatent,maxLatent)=new(minPars,maxPars,minLatent,maxLatent)
 end

 mutable struct latentModel
 	dt::data
 	irt::IRT
 	simData::block
 	estimates::block
 	bs::bootstrap
 	bds::bounds
 	io::intOpt
 	eo::extOpt
 	prf::performance
 	latentModel()=new(data(),IRT(),block(),block(),bootstrap(),bounds(),intOpt(),extOpt(),performance())
 	latentModel(dt,irt,simData,estimates,bs,bds,io,eo,prf)=new(dt,irt,simData,estimates,bs,bds,io,eo,prf)
 end
 include("utils.jl")
 include("bootstrap.jl")
 include("cubicSpline.jl")
 include("preTestAssembly.jl")
 include("simulate.jl")
 include("save.jl")
 include("MMLE2PL.jl")
 include("Mstep.jl")

export  rescaleLatentValues, probFun, genResp, savePool, saveLatentValues, simulateData, overlapPairs, pretestCPLEX, calibrate, performance, latent, IRT ,data, block, bootstrap, intOpt, extOpt, bounds, latentModel, RMSE, BIAS
IRTCalibration
end # module
