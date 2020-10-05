mutable struct Performance
    time::Float64#Dates.Millisecond
    n_iter::Int64
    Performance() = new(0.0, 0)
    Performance(time, n_iter) = new(time, n_iter)
end

mutable struct Latent
    dist::Distributions.Distribution
    metric::Vector{Float64}
    Latent() = new(Distributions.Normal(0, 1), [zero(Float64), one(Float64)])
    Latent(dist, metric) = new(dist, metric)
end

mutable struct IRT
    model::String
    n_par::Int64
    n_latent::Int64
    n_items::Int64
    N::Int64 #n test takers
    IRT() = new("2PL", 2, 1, 0, 0)
    IRT(model, n_par, n_latent, n_items, N) = new(model, n_par, n_latent, n_items, N)
end

mutable struct Data
    responses::Matrix{Float64}
    design::Matrix{Float64}
    unbalanced::Bool
    T::Int64
    n::Int64
    f::Vector{Float64}
    Data() = new(zeros(Float64, 0, 0), Matrix{Float64}(undef, 0, 0), true, 0, 0, zeros(1))
    Data(responses, design, unbalanced, T, n, f) =
        new(responses, design, unbalanced, T, n, f) #no pattern mode
end

mutable struct Block
    pars::Matrix{Float64}
    latent_values::Matrix{Float64}
    latents::Vector{Latent}
    Block() = new(Matrix{Float64}(undef, 0, 0), Matrix{Float64}(undef, 0, 0), [Latent()])
    Block(pars, latent_values, latents) = new(pars, latent_values, latents)
end

mutable struct Bootstrap
    bootstrap::Bool #true or false
    R::Int64 #number of replications
    sample_frac::Float64
    type::String #parametric or nonParametric
    n_bins::Int64
    Bootstrap() = new(false, 1, 2 / 3, "parametric", 50)
    Bootstrap(bootstrap, R, sample_frac, type, n_bins) = new(bootstrap, R, sample_frac, type, n_bins)
end

mutable struct IntOpt #if you want to specify a termination parameter you have to specify all of them
    solver::String
    x_tol_rel::Float64
    f_tol_rel::Float64
    time_limit::Int64
    IntOpt() = new("NLopt", 1e-4, 1e-5, 10)
    IntOpt(solver, x_tol_rel, f_tol_rel, time_limit) = new(solver, x_tol_rel, f_tol_rel, time_limit)
end

mutable struct ExtOpt #if you want to specify a termination parameter you have to specify all of them
    method::String# =optima weighted sum #can be: ["OWS" , "WLE" , "JML"]
    den_type::Union{Distributions.Distribution,String}
    K::Int64 #theta points
    int_W::Int64 #can be: ["1" , "1/s"  , "1/sqrt(s)"]
    min_max_W::Vector{Int64}
    first::String #can be: ["theta", "items"]
    l_tol_rel::Float64
    x_tol_rel::Float64
    time_limit::Int64
    max_iter::Int64
    ExtOpt() = new("OWS", Distributions.Normal(0, 1), 21, 3, [9, 15], "theta", 1e-20, 1e-5, 1000, 1000)
    ExtOpt(method, den_type, K, int_W, min_max_W, first, l_tol_rel, x_tol_rel, time_limit, max_iter) =
        new(method, den_type, K, int_W, min_max_W, first, l_tol_rel, x_tol_rel, time_limit, max_iter)
end

mutable struct Bounds
    min_pars::Vector{Float64}
    max_pars::Vector{Float64}
    min_latent::Vector{Float64}
    max_latent::Vector{Float64}
    Bounds() = new([0.00001, -10.0], [10.0, 10.0], [-10.0], [10.0]) #the last vectors must be the same size of the IRT , es. 2PL, size=2
    Bounds(min_pars, max_pars, min_latent, max_latent) =
        new(min_pars, max_pars, min_latent, max_latent)
end

mutable struct LatentModel
    dt::Data
    irt::IRT
    simulated_data::Block
    estimates::Block
    bootstrap::Bootstrap
    bounds::Bounds
    int_opt::IntOpt
    ext_opt::ExtOpt
    performance::Performance
    LatentModel() = new(
        Data(),
        IRT(),
        Block(),
        Block(),
        Bootstrap(),
        Bounds(),
        IntOpt(),
        ExtOpt(),
        Performance()
    )
    LatentModel(dt, irt, simulated_data, estimates, bootstrap, bounds, int_opt, ext_opt, performance) =
        new(dt, irt, simulated_data, estimates, bootstrap, bounds, int_opt, ext_opt, performance)
end
