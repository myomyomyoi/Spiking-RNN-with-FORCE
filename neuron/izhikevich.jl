using Plots
using Base:@kwdef

@kwdef struct IzhikevichParams
    C::Float32 = 250
    v_r::Float32 = -60
    v_t::Float32 = -20
    b::Float32 = -2
    v_peak::Float32 = 30
    v_reset::Float32 = -65
    a::Float32 = 0.01
    d::Float32 = 200
    k::Float32 = 2.5
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

params = IzhikevichParams(Δt=0.01)
v, u = -65, 0

T = 100000
v_arr, u_arr = zeros(T), zeros(T)

for i = 1:T
    v_arr[i], u_arr[i] = v, u

    I = ifelse(i>(T/2), 3000, 1000)
    v, u, _ = update(v, u, params, I)
end

p1 = plot(v_arr, label="v")
p2 = plot(u_arr, label="u")
plot(p1, p2,
    layout=@layout[ a{0.7h} ; b ],
    dpi=150)