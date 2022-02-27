include("../model/IzhikevichRNN.jl")
include("../util.jl")
using .IzhikevichRNN
using .Util

using Plots
using LinearAlgebra

input = Util.genereate_sinusoidal(15000, Δt=0.04, Hz=5)

N = 500
M = 1
T = length(input)

model = IzhikevichRNN.Model(seed=1, N=N, M=M)

# store array
v = zeros(N, T)
r = zeros(N, T)
ϕ = zeros(M, 20, T)
fired = zeros(N, T)
pred = zeros(M, T)
error = zeros(M, T)

w = model.ω_0 + transpose(model.η)*model.ϕ
before_evs = eigvals(w)

@time for i = 1:T
    mode = ifelse(10000/0.04 > i > 5000/0.04, 1, 0)
    IzhikevichRNN.update!(model, [input[i]], mode)

    v[:, i] = model.v
    r[:, i] = model.r
    ϕ[:, :, i] = model.ϕ[:, 1:20, :]
    fired[:, i] = model.fired
    pred[:, i] = model.pred
    error[:, i] = model.error
end

w = model.ω_0 + transpose(model.η)*model.ϕ
after_evs = eigvals(w)

 # 0~1の範囲に変換
p1 = plot(input, label="input", title="Input and Prediction")
plot!(pred[1, :], label="pred")

p4 = plot(ϕ[1, 1, 100000:end], title="Decoder")
plot!(ϕ[1, 2, 100000:end])
plot!(ϕ[1, 3, 100000:end])
plot!(ϕ[1, 4, 100000:end])
plot!(ϕ[1, 5, 100000:end])

p5 = scatter(real(before_evs), imag(before_evs), title="eigvals")
scatter!(real(after_evs), imag(after_evs))

p = plot(p1, p4, p5,
    layout=@layout[ a ; b c ],
    dpi=200,
    titlefontsize=10,
    legend=false,)

savefig(p, "./result/sin_a.png")

p2a = plot(v[1, 50000:75000], title="Before learning")
p2b = plot(v[22, 50000:75000])
p2c = plot(v[33, 50000:75000])
p2d = plot(v[495, 50000:75000])
p2 = plot(p2a, p2b, p2c, p2d, layout=@layout[grid(4, 1)])

p3a = plot(v[1, 150000:175000], title="During learning")
p3b = plot(v[22, 150000:175000])
p3c = plot(v[33, 150000:175000])
p3d = plot(v[495, 150000:175000])
p3 = plot(p3a, p3b, p3c, p3d, layout=@layout[grid(4, 1)])

pp = plot(p2, p3,
    dpi=200,
    titlefontsize=10,
    legend=false,
    xticks=false)

savefig(pp, "./result/sin_b.png")





# w = model.ω⁰ + transpose(model.η)*model.ϕ
# es = eigvals(w)