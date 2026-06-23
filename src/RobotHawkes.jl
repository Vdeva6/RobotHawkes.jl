module RobotHawkes

using Lux
using Random
using NNlib
using Integrals
using ChainRulesCore

include("layers/temporal_embedding.jl")
include("layers/event_embedding.jl")
include("layers/transformer_hawkes_cell.jl")
include("models/transformer_hawkes_model.jl")
include("losses/loglikelihood.jl")
include("integration/intensity_integral.jl")
include("rules/hawkes_attention_rrule.jl")
include("rules/loss_rrules.jl")

export TemporalEmbedding
export EventEmbedding
export TransformerHawkesCell
export TransformerHawkesModel

export observed_loglikelihood
export observed_nll
export model_observed_nll

export total_intensity_integral
export full_hawkes_nll
export model_full_hawkes_nll
export quadgk_integral

end