function max_i(
    X::Matrix{Float64},
    sumpk_i::Vector{Float64},
    r1_i::Vector{Float64},
    pars_i::Vector{Float64},
    n_par::Int64,
    opt::NLopt.Opt,
)
    function myf(x::Vector, grad::Vector)
        n_par = size(x, 1)
        if n_par == 2
            y = X * x
        else
            y = x
        end
        if size(grad, 1) > 0
            p = r1_i - (sumpk_i ./ (1 .+ (exp_c.(.-y))))
            p = X' * p
            for i = 1:size(grad, 1)
                grad[i] = p[i]
            end
        end
        z = log1p_c.(exp_c.(y))
        return sum(r1_i .* y - (sumpk_i .* z))
    end
    opt.max_objective = myf
    opt_f = Array{Cdouble}(undef, 1)
    ccall(
        (:nlopt_optimize, NLopt.libnlopt),
        NLopt.Result,
        (NLopt._Opt, Ptr{Cdouble}, Ptr{Cdouble}),
        opt,
        pars_i,
        opt_f,
    )
    return pars_i::Vector{Float64}
end
#  function maxItems(pars_start::Array{Float64,2},r1::Matrix{Float64}, sumpk::Matrix{Float64},X::Matrix{Float64},int_opt::IntOpt, bounds::Bounds)
#     n_items=size(pars_start,2)
#     n_par=size(bounds.min_pars,1)
#     opt=NLopt.Opt(:LD_SLSQP,n_par)
#     opt.lower_bounds = bounds.min_pars
#     opt.upper_bounds = bounds.max_pars
#     opt.xtol_rel = int_opt.x_tol_rel
#     opt.maxtime = int_opt.time_limit
#     opt.ftol_rel=  int_opt.f_tol_rel
#     opt.maxeval=50
#     opt_f = Array{Cdouble}(undef,1)
#     for i=1:n_items
#         pars_i=max_i(X,sumpk[:,i],r1_i[:,i],pars_start[:,i],n_par,opt)
#         if n_par==1
#             pars_start[2,i]=copy(pars_i)
#         else
#             pars_start[:,i]=copy(pars_i)
#         end
#     end
#     return pars_start::Matrix{Float64}
# end

function maxLHMMLE(
    parsStart::Matrix{Float64},
    phi::Matrix{Float64},
    posterior::Matrix{Float64},
    iIndex::Vector{Vector{Int64}},
    design::Matrix{Float64},
    X::Matrix{Float64},
    Wk::Vector{Float64},
    responses::Matrix{Float64},
    int_opt::IntOpt,
    bounds::Bounds,
)
    n_items = size(parsStart, 2)
    N = size(iIndex, 1)
    K = size(X, 1)
    sumpk = zeros(Float64, K, n_items)
    r1 = similar(sumpk)
    posterior = posterior_simplified(posterior, N, K, n_items, iIndex, responses, Wk, phi)
    LinearAlgebra.BLAS.gemm!(
        'T',
        'T',
        one(Float64),
        posterior,
        design,
        zero(Float64),
        sumpk,
    )# sumpk KxI
    LinearAlgebra.BLAS.gemm!(
        'T',
        'T',
        one(Float64),
        posterior,
        responses,
        zero(Float64),
        r1,
    )# r1 KxI
    n_par = size(bounds.min_pars, 1)
    opt = NLopt.Opt(:LD_SLSQP, n_par)
    opt.lower_bounds = bounds.min_pars
    opt.upper_bounds = bounds.max_pars
    opt.xtol_rel = int_opt.x_tol_rel
    opt.maxtime = int_opt.time_limit
    opt.ftol_rel = int_opt.f_tol_rel
    #opt.maxeval=50
    Distributed.@sync Distributed.@distributed for i = 1:n_items
        pars_i = max_i(X, sumpk[:, i], r1[:, i], parsStart[:, i], n_par, opt)
        if n_par == 1
            parsStart[2, i] = copy(pars_i)
        else
            parsStart[:, i] = copy(pars_i)
        end
    end
    LinearAlgebra.BLAS.gemm!('N', 'N', one(Float64), X, parsStart, zero(Float64), phi)# phi=New_pars*X1', if A'*B then 'T', 'N'
    return parsStart::Matrix{Float64}, phi::Matrix{Float64}
end
