module SpikingESN
using Base:@kwdef
using Parameters: @unpack, @pack!
using LinearAlgebra
using DataStructures
using Random

@kwdef struct IzhikevichParams
    C::Float32 = 250
    b::Float32 = -2
    v_peak::Float32 = 30
    v_reset::Float32 = -65
    a::Float32 = 0.01
    d::Float32 = 200
    k::Float32 = 2.5
    v_r::Float32 = -60
    v_t::Float32 = v_r+40-(b/k); 
    Δt::Float32 = 0.04
end

function update(v, u, params::IzhikevichParams, I)
    function dvdt(v, v_r, v_t, k, u, I, C)
        return (k*(v - v_r)*(v - v_t) - u + I)/C
    end
    
    function dudt(u, v, v_r, a, b)
        return a*(b*(v - v_r) - u)
    end

    _v, _u = v, u
    v += params.Δt * dvdt(_v, params.v_r, params.v_t, params.k, _u, I, params.C)
    u += params.Δt * dudt(_u, _v, params.v_r, params.a, params.b)

    fired::Bool = v >= params.v_peak
    u = ifelse(v >= params.v_peak, u + params.d, u)
    v = ifelse(v >= params.v_peak, params.v_reset, v)

    return v, u, fired
end

@kwdef mutable struct Network
    seed::Int = 0
    rng::MersenneTwister = MersenneTwister(seed)
    
    N::Int
    M::Int

    params::IzhikevichParams = IzhikevichParams() # params for neurons
    I_bias::Float32 = 1000
    τ_R::Float32 = 2
    τ_D::Float32 = 20
    G::Float32 = 10^4
    Q::Float32 = 10^4
    p::Float32 = 0.1

    Gω_0::Matrix{Float32} = G .* randn(rng, N, N) .* (rand(rng, N, N) .< p) ./ (p*sqrt(N)) # gaussian
    ϕ::Matrix{Float32} = zeros(M, N) # W_out, BPhi
    Qη::Matrix{Float32} = Q .* 2 .* (rand(rng, M, N) .- 1) # uniform over [-1, 1], E

    P::Matrix{Float32} = Matrix(I * 2, N, N) # NxN
    cd::Vector{Float32} = zeros(N)

    v::Vector{Float32} = rand(rng, N) .* (params.v_r + (params.v_peak - params.v_r))
    u::Vector{Float32} = zeros(N)
    fired::Vector{Bool} = zeros(N)

    I::Vector{Float32} = zeros(N)
    IPSC::Vector{Float32} = zeros(N)
    h::Vector{Float32} = zeros(N)
    r::Vector{Float32} = zeros(N)
    hr::Vector{Float32} = zeros(N)
    JD::Vector{Float32} = zeros(N)
    pred::Vector{Float32} = zeros(M)
    error::Vector{Float32} = zeros(M)
end

function update!(model::Network, input::Vector, mode::Int)
    @unpack N, M, I_bias, τ_R, τ_D, Gω_0, ϕ, Qη, v, u, fired, P, cd, I, IPSC, h, r, hr, JD, pred, error = model
    @unpack Δt = model.params
    
    I .= I_bias
    for i = 1:N
        @inbounds I[i] += IPSC[i]

        for j = 1:M
            @inbounds I[i] += Qη[j, i] * pred[j] # W_fb?
        end

        # Euler integration
        @inbounds v[i], u[i], fired[i] = update(v[i], u[i], model.params, I[i])
    end

    JD .= 0
    for i = 1:N
        for j = 1:N
            @inbounds JD[i] += Gω_0[j, i] * fired[i] # W_rec
        end
    end

    # synaptic response
    for i = 1:N
        @inbounds IPSC[i] = IPSC[i] * exp(-Δt / τ_R) + h[i] * Δt
        @inbounds r[i] = r[i] * exp(-Δt / τ_R) + hr[i] * Δt
        @inbounds h[i] = h[i] * exp(-Δt / τ_D) + ifelse(reduce(+, fired)>0, JD[i]/(τ_R * τ_D), 0)
        @inbounds hr[i] = hr[i] * exp(-Δt / τ_D) + ifelse(fired[i], 1/(τ_R * τ_D), 0)
    end

    # Prediction
    pred .= 0
    for i = 1:N
        for j = 1:M
            @inbounds pred[j] += ϕ[j, i] * r[i]
        end
    end

    # Error calculation
    for i = 1:M
        @inbounds error[i] = pred[i] - input[i]
    end

    if mode == 1 # FORCE training
        cd .= 0
        for i = 1:N
            for j = 1:N
                @inbounds cd[j] += P[j, i] * r[i]
            end
        end

        dotcd = 0.0
        for i = 1:N
            @inbounds dotcd += cd[i] * r[i]
        end

        for i = 1:N
            for j = 1:N
                @inbounds P[j, i] -= cd[j] * cd[i] / (1 + dotcd)
            end
        end

        for i = 1:N
            for j = 1:M
                @inbounds ϕ[j, i] -= error[j] * cd[i]
            end
        end
    end

    @pack! model = ϕ, v, u, fired, P, I, IPSC, h, r, hr, JD, pred, error
end
end