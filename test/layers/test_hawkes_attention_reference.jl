using Test
using Random
using RobotHawkes

function reference_hawkes_attention(qh, kh, vh, times, decay)
    size(qh) == size(kh) || throw(DimensionMismatch("qh and kh must match"))
    size(qh) == size(vh) || throw(DimensionMismatch("qh and vh must match"))

    D, T_seq, H, B = size(qh)

    size(times) == (T_seq, B) ||
        throw(DimensionMismatch("times must have shape (T, B)"))

    length(decay) == H ||
        throw(DimensionMismatch("decay length must equal H"))

    S = promote_type(eltype(qh), eltype(kh), eltype(vh), eltype(times), eltype(decay))

    y = zeros(S, D, T_seq, H, B)
    scale = inv(sqrt(S(D)))

    @inbounds for b in 1:B
        for h in 1:H
            decay_h = abs(S(decay[h]))

            for t in 1:T_seq
                scores = Vector{S}(undef, t)

                for s in 1:t
                    dot_qk = zero(S)

                    for d in 1:D
                        dot_qk += S(kh[d, s, h, b]) * S(qh[d, t, h, b])
                    end

                    Δ = max(S(times[t, b] - times[s, b]), zero(S))

                    scores[s] = dot_qk * scale - decay_h * Δ
                end

                max_score = maximum(scores)

                weights = exp.(scores .- max_score)
                weights ./= sum(weights)

                for d in 1:D
                    acc = zero(S)

                    for s in 1:t
                        acc += weights[s] * S(vh[d, s, h, b])
                    end

                    y[d, t, h, b] = acc
                end
            end
        end
    end

    return y
end

@testset "Hawkes attention reference comparison" begin
    rng = Xoshiro(2025)

    D = 4
    T_seq = 7
    H = 3
    B = 2

    qh = randn(rng, Float32, D, T_seq, H, B)
    kh = randn(rng, Float32, D, T_seq, H, B)
    vh = randn(rng, Float32, D, T_seq, H, B)

    Δt = rand(rng, Float32, T_seq, B)
    times = cumsum(Δt; dims = 1)

    decay = randn(rng, Float32, H)

    y_ref = reference_hawkes_attention(qh, kh, vh, times, decay)
    y_fast = RobotHawkes._hawkes_attention(qh, kh, vh, times, decay)

    @test size(y_fast) == size(y_ref)
    @test eltype(y_fast) == Float32
    @test all(isfinite, y_fast)

    @test y_fast ≈ y_ref rtol = 1.0f-5 atol = 1.0f-5
end