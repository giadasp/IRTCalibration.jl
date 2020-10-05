__precompile__()
module IRTCalibration
import CSV
import DataFrames
import Distributions
import Distributed
import DelimitedFiles
import LinearAlgebra
import StatsBase
import Interpolations
import NLopt
import JLD2
import SharedArrays
using Requires

const readdlm = DelimitedFiles.readdlm
const writedlm = DelimitedFiles.writedlm
const SharedArray = SharedArrays.SharedArray
const std = StatsBase.std
const sample = StatsBase.sample

include("structs.jl")
include("utils.jl")
include("bootstrap.jl")
include("cubic_spline.jl")

function __init__()
    @require JuMP = "4076af6c-e467-56ae-b986-b466b2749572" begin
        @require CPLEX = "a076750e-1247-5638-91d2-ce28b192dca0" begin
            include("pre_test_assembly.jl")
        end
    end
end

include("simulate.jl")
include("save.jl")
include("MMLE_2PL.jl")
include("M_step.jl")

export rescaleLatentValues,
    probability,
    generate_responses,
    save_pool,
    save_latent_values,
    simulate_data,
    _overlap_pairs,
    pre_test_CPLEX,
    calibrate,
    Performance,
    Latent,
    IRT,
    Data,
    Block,
    Bootstrap,
    IntOpt,
    ExtOpt,
    Bounds,
    LatentModel,
    RMSE,
    BIAS

IRTCalibration

end # module
