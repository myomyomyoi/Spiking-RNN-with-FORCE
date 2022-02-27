module Util

function genereate_sinusoidal(T; Δt, Hz)
    step_length::Int = round(T/Δt)

    x = zeros(step_length)
    
    for i = 1:step_length
        x[i] = sin((i-1)*Δt*2π*Hz)
    end

    return x
end

function generate_mackeyglass(T; Δt, β, γ, n, τ, x0)

    function rk4(x, delay_x, β, γ, n, Δt)

        function dxdt(x, delay_x, β, γ, n)
            return (β*delay_x)/(1+delay_x^n) - γ*x
        end
    
        k1 = Δt * dxdt(x, delay_x, β, γ, n)
        k2 = Δt * dxdt(x+k1/2, delay_x, β, γ, n)
        k3 = Δt * dxdt(x+k2/2, delay_x, β, γ, n)
        k4 = Δt * dxdt(x+k3, delay_x, β, γ, n)
    
        return x + (k1 + 2k2 + 2k3 + k4) / 6
    end

    step_length::Int = round(T/Δt)
    hist_length::Int = round(τ/Δt)
 
    x = zeros(step_length)
    x[1] = x0

    for i = 1:step_length-1
        if i > hist_length
            delay_x = x[i - hist_length]
        else
            delay_x = 0
        end

        x[i+1] = rk4(x[i], delay_x, β, γ, n, Δt)
    end

    return x
end

function generate_lorenz(T; Δt, ρ, σ, B, x0, y0, z0)

    function rk4(x, y, z, ρ, σ, B, Δt)

        function dxdt(x, y, σ)
            return σ*(y - x)
        end

        function dydt(x, y, z, ρ)
            return x*(ρ - z) - y
        end

        function dzdt(x, y, z, B)
            return x*y  - B*z
        end
    
        k1_x = Δt * dxdt(x, y, σ)
        k1_y = Δt * dydt(x, y, z, ρ)
        k1_z = Δt * dzdt(x, y, z, B)

        k2_x = Δt * dxdt(x+k1_x/2, y+k1_y/2, σ)
        k2_y = Δt * dydt(x+k1_x/2, y+k1_y/2, z+k1_z/2, ρ)
        k2_z = Δt * dzdt(x+k1_x/2, y+k1_y/2, z+k1_z/2, B)

        k3_x = Δt * dxdt(x+k2_x/2, y+k2_y/2, σ)
        k3_y = Δt * dydt(x+k2_x/2, y+k2_y/2, z+k2_z/2, ρ)
        k3_z = Δt * dzdt(x+k2_x/2, y+k2_y/2, z+k2_z/2, B)

        k4_x = Δt * dxdt(x+k3_x, y+k3_y, σ)
        k4_y = Δt * dydt(x+k3_x, y+k3_y, z+k3_z, ρ)
        k4_z = Δt * dzdt(x+k3_x, y+k3_y, z+k3_z, B)
        
        x += (k1_x + 2k2_x + 2k3_x + k4_x) / 6
        y += (k1_y + 2k2_y + 2k3_y + k4_y) / 6
        z += (k1_z + 2k2_z + 2k3_z + k4_z) / 6
    
        return x, y, z
    end

    step_length::Int = round(T/Δt)

    x = zeros(step_length)
    y = zeros(step_length)
    z = zeros(step_length)

    x[1] = x0
    y[1] = y0
    z[1] = z0

    for i = 1:step_length-1
        x[i+1], y[i+1], z[i+1] = rk4(x[i], y[i], z[i], ρ, σ, B, Δt)
    end

    return x, y, z
end

function generate_vdp(T; Δt, μ, x0, y0)

    function rk4(x, y, μ, t)

        function dxdt(y)
            return y
        end

        function dydt(x, y, μ)
            return -μ*(x^2 - 1)*y - x
        end
    
        k1_x = Δt * dxdt(y)
        k1_y = Δt * dydt(x, y, μ)

        k2_x = Δt * dxdt(y+k1_y/2)
        k2_y = Δt * dydt(x+k1_x/2, y+k1_y/2, μ)

        k3_x = Δt * dxdt(y+k2_y/2)
        k3_y = Δt * dydt(x+k2_x/2, y+k2_y/2, μ)

        k4_x = Δt * dxdt(y+k3_y)
        k4_y = Δt * dydt(x+k3_x, y+k3_y, μ)
        
        x += (k1_x + 2k2_x + 2k3_x + k4_x) / 6
        y += (k1_y + 2k2_y + 2k3_y + k4_y) / 6
    
        return x, y
    end

    step_length::Int = round(T/Δt)

    x = zeros(step_length)
    y = zeros(step_length)

    x[1] = x0
    y[1] = y0

    for i = 1:step_length-1
        x[i+1], y[i+1] = rk4(x[i], y[i], μ, Δt)
    end

    return x, y
end

end