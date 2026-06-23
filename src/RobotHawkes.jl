module RobotHawkes

using Lux
using Random

include("layers/temporal_embedding.jl")
include("layers/event_embedding.jl")

export TemporalEmbedding
export EventEmbedding

end