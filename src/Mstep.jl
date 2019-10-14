function max_i(X::Matrix{Float64},sumpk_i::Vector{Float64},r1_i::Vector{Float64},pars_i::Vector{Float64},nPar::Int64,opt::NLopt.Opt)
    function myf(x::Vector,grad::Vector)
        nPar=size(x,1)
        if nPar==2
            y=X*x
        else
            y=x
        end
        if size(grad,1) > 0
                p= r1_i - (sumpk_i ./ (1 .+(exp_c.(.- y))))
                p= X'* p
                for i=1:size(grad,1)
                    grad[i]=p[i]
                end
        end
        z=log1p_c.(exp_c.(y))
        return sum(r1_i .* y - (sumpk_i.*z))
    end
    opt.max_objective = myf
    opt_f = Array{Cdouble}(undef,1)
    ccall((:nlopt_optimize,NLopt.libnlopt), NLopt.Result, (NLopt._Opt, Ptr{Cdouble}, Ptr{Cdouble}), opt, pars_i, opt_f)
    return pars_i::Vector{Float64}
end
#  function maxItems(pars_start::Array{Float64,2},r1::Matrix{Float64}, sumpk::Matrix{Float64},X::Matrix{Float64},io::intOpt, bds::bounds)
#     nItems=size(pars_start,2)
#     nPar=size(bds.minPars,1)
#     opt=NLopt.Opt(:LD_SLSQP,nPar)
#     opt.lower_bounds = bds.minPars
#     opt.upper_bounds = bds.maxPars
#     opt.xtol_rel = io.xTolRel
#     opt.maxtime = io.timeLimit
#     opt.ftol_rel=  io.fTolRel
#     opt.maxeval=50
#     opt_f = Array{Cdouble}(undef,1)
#     for i=1:nItems
#         pars_i=max_i(X,sumpk[:,i],r1_i[:,i],pars_start[:,i],nPar,opt)
#         if nPar==1
#             pars_start[2,i]=copy(pars_i)
#         else
#             pars_start[:,i]=copy(pars_i)
#         end
#     end
#     return pars_start::Matrix{Float64}
# end

function maxLHMMLE(parsStart::Matrix{Float64},
    phi::Matrix{Float64},
    posterior::Matrix{Float64},
    iIndex::Vector{Vector{Int64}},
    design::Matrix{Float64},
    X::Matrix{Float64},
    Wk::Vector{Float64},
    responses::Matrix{Float64},
    io::intOpt,
    bds::bounds)
    nItems=size(parsStart,2)
    N=size(iIndex,1)
    K=size(X,1)
    sumpk=zeros(Float64,K,nItems)
    r1=similar(sumpk)
    posterior=compPostSimp(posterior,N,K,nItems,iIndex,responses,Wk,phi)
    LinearAlgebra.BLAS.gemm!('T', 'T', one(Float64), posterior, design, zero(Float64), sumpk)# sumpk KxI
    LinearAlgebra.BLAS.gemm!('T', 'T', one(Float64), posterior, responses, zero(Float64), r1)# r1 KxI
    nPar=size(bds.minPars,1)
    opt=NLopt.Opt(:LD_SLSQP,nPar)
    opt.lower_bounds = bds.minPars
    opt.upper_bounds = bds.maxPars
    opt.xtol_rel = io.xTolRel
    opt.maxtime = io.timeLimit
    opt.ftol_rel=  io.fTolRel
    #opt.maxeval=50
    Distributed.@sync Distributed.@distributed for i=1:nItems
        pars_i=max_i(X,sumpk[:,i],r1[:,i],parsStart[:,i],nPar,opt)
        if nPar==1
            parsStart[2,i]=copy(pars_i)
        else
            parsStart[:,i]=copy(pars_i)
        end
    end
    LinearAlgebra.BLAS.gemm!('N', 'N', one(Float64), X, parsStart, zero(Float64), phi)# phi=New_pars*X1', if A'*B then 'T', 'N'
    return parsStart::Matrix{Float64}, phi::Matrix{Float64}
end
