include("../util.jl")
using .Util

using Plots

mg = Util.generate_mackeyglass(1000, β=0.2, γ=0.1, τ=32, n=10, Δt=1, x0=1.2)
ss = Util.genereate_sinusoidal(5, Δt=0.01, Hz=5)
vdp_h_x, vdp_h_y = generate_vdp(50, Δt=0.001, μ=0.3, x0=0, y0=2)
vdp_r_x, vdp_r_y = generate_vdp(50, Δt=0.001, μ=5, x0=0, y0=2)
l_x, l_y, l_z = Util.generate_lorenz(50, Δt=0.001, ρ=28, σ=10, B=8/3, x0=-10, y0=-10, z0=20)

p1 = plot(ss, title="Sinsoidal (5Hz)")
p2 = plot(vdp_h_x, title="VDP (harmonic)", label="x")
plot!(vdp_h_y, label="y")
p3 = plot(vdp_r_x, title="VDP (relaxation)", label="x")
plot!(vdp_r_y, label="y")
p4 = plot(l_x, title="Lorenz", label="x")
plot!(l_y, label="y")
plot!(l_z, label="z")
p5 = plot(mg, title="Mackey-Glass (not in paper)")

p = plot(p1, p2, p3, p4, p5,
    layout=@layout[a;b c;d;e],
    legend=false,
    titlefontsize=10,
    ticks=false,
    dpi=200)

savefig(p, "./result/data.png")



