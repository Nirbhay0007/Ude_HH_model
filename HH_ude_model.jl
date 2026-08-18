using DataFrames, DifferentialEquations, Plots, CSV, Random, Lux, Optimization
using OptimizationOptimisers: Adam
using OptimizationOptimJL: LBFGS
using Zygote, ComponentArrays, SciMLSensitivity, JLD2, StaticArrays, ForwardDiff
data_loaded = CSV.read("C:/Ude_HH_model/ude_models/synthetic_data.csv", DataFrame)
# all constants
true_n = Float32.(data_loaded[!, "n"])
true_V = Float32.(data_loaded[!, "V"])

true_m = Float32.(data_loaded[!, "m"])
true_h = Float32.(data_loaded[!, "h"])
t_steps = Float32.(data_loaded[!, "Time"])
# all constants
g_na = 120.0f0
g_k = 36.0f0
g_l = 0.3f0
c_m = 1.0f0
I_ext = 10.0f0
E_na = 50.0f0
E_k = -77.0f0
E_l = -54.4f0




# rate_functions
# rate funtions
typeof(c_m)
# --- Potassium Gating (n) ---
alpha_n(V) = abs(V + 55.0f0) < 1.0f-6 ? 0.1f0 : 0.01f0 * (V + 55.0f0) / (1.0f0 - exp(-(V + 55.0f0) / 10.0f0))
beta_n(V) = 0.125f0 * exp(-(V + 65.0f0) / 80.0f0)


# --- Sodium Activation (m) ---
alpha_m(V) = abs(V + 40.0f0) < 1.0f-6 ? 1.0f0 : 0.1f0 * (V + 40.0f0) / (1.0f0 - exp(-(V + 40.0f0) / 10.0f0))
beta_m(V) = 4.0f0 * exp(-(V + 65.0f0) / 18.0f0)

# --- Sodium Inactivation (h) ---
alpha_h(V) = 0.07f0 * exp(-(V + 65.0f0) / 20.0f0)
beta_h(V) = 1.0f0 / (1.0f0 + exp(-(V + 35.0f0) / 10.0f0))


using Statistics
rng = Random.seed!(42)
Random.seed!(42)
noise_std_V = 1.5f0     # 1.5 mV noise (or 0.05f0 * std(true_V))
noise_std_n = 0.005f0    # 2% gating probability noise
noise_std_m = 0.05f0
noise_std_h = 0.05f0
noisy_V = true_V .+ noise_std_V.* randn(Float32, size(true_V))
noisy_n= (true_n .+ noise_std_n .* randn(Float32, size(true_n)))

noisy_m = (true_V .+ noise_std_m .* randn(Float32, size(true_m)))
noisy_h = (true_V .+ noise_std_h .* randn(Float32, size(true_h)))

noise_n  = clamp.(noisy_n,0.0f0 , 1.0f0)
noise_m  = clamp.(noisy_m,0.0f0 , 1.0f0)
noise_h  = clamp.(noisy_h,0.0f0 , 1.0f0)

# ploing the noise vs ture data 

v_plot = plot(t_steps,true_V,label = " ture_V")
plot(v_plot,t_steps,noisy_V,seriestype=:scatter, markersize=2,label = "nosiy_V")

n_plot = plot(t_steps,true_n,label = " ture_n")
plot(n_plot,t_steps,noisy_n,seriestype=:scatter, markersize=2,label = "nosiy_n")

m_plot = plot(t_steps,true_m,label = " ture_m")
plot(m_plot,t_steps,noisy_m,seriestype=:scatter, markersize=2,label = "nosiy_m")

h_plot = plot(t_steps,true_h,label = " ture_h")
plot(h_plot,t_steps,noisy_h,seriestype=:scatter, markersize=2,label = "nosiy_h")



nn = Chain(Dense(1 => 32),
    Dense(32 => 32, tanh),
    Dense(32 => 1, sigmoid))

    ps, st = Lux.setup(rng, nn)

function ude_hh!(du, u, p, t)

    V, m, h, n = u
    ps = p

    pred_n, _ = nn(@SVector[n], ps, st)






    du[1] = 1 / c_m * (I_ext - g_na * m^3 * h * (V - E_na) - g_k * pred_n[1] * (V - E_k) - g_l * (V - E_l))
    du[2] = alpha_m(V) * (1 - m) - beta_m(V) * (m)
    du[3] = alpha_h(V) * (1 - h) - beta_h(V) * (h)
    du[4] = alpha_n(V) * (1 - n) - beta_n(V) * (n)


end



u_0 = [-65.0f0, 0.05f0, 0.6f0, 0.317f0]

tspan = (0.0f0, 30.0f0)
p = [g_na, g_k, g_l, c_m, I_ext, E_na, E_k, E_l, ps, st]
prob = ODEProblem(ude_hh!, u_0, tspan, p)
t_steps
function loss_function(ps, p)


    prob = ODEProblem(ude_hh!, u_0, tspan, ps)
    sol = solve(prob, TRBDF2(), reltol=1e-6, abstol=1e-6, saveat=t_steps, sensealg=ForwardDiffSensitivity())
    if sol.retcode != SciMLBase.ReturnCode.Success || length(sol.t) != length(true_V)
        return 1e5
    end
    pred_V = sol[1, :]
    loss = sum(abs2, pred_V - true_V) / length(true_V)
    return loss

end

# Optimization
size(true_V)

# a callback function
iter = 0

function callback(state, l)
    if state.iter % 5 == 0
        println("current iteration = $(state.iter)  | current_loss = $(l)")
        if state.iter % 10 == 0
            jldsave("C:/Ude_HH_model/ude_models/leaning_parameter.jld2"; p=state.u)
        end
    end
    return false
end

opt = OptimizationFunction(loss_function, AutoForwardDiff())

# Optimization
println(typeof(ps))
net = ComponentArray(ps)
opt_prob = OptimizationProblem(opt, net)

println("Lets start the training ====================================== ")

# 1st optimization loop: Adam(0.05) -- 1500 iterations
println("--- Phase 1: Adam(0.05) [1500 maxiters] ---")
res1 = solve(opt_prob, Adam(0.01), callback=callback, maxiters=500)

# 2nd optimization loop: Adam(0.01) -- 1000 iterations
println("--- Phase 2: Adam(0.01) [1000 maxiters] ---")
opt_prob2 = remake(opt_prob, u0=res1.u)
res2 = solve(opt_prob2, Adam(0.001), callback=callback, maxiters=1000)

# 3rd optimization loop: LBFGS -- 500 iterations
println("--- Phase 3: LBFGS [500 maxiters] ---")
opt_prob3 = remake(opt_prob, u0=res2.u)
res3 = solve(opt_prob3, LBFGS(), callback=callback, maxiters=1000)

# Save final parameters
jldsave("C:/Ude_HH_model/ude_models/leaning_parameter2.jld2"; p=res3.u)
println("Training completed and parameters saved to C:/Ude_HH_model/ude_models/leaning_parameter2.jld2")
# --------------------------------------------
pn = load("C:/Ude_HH_model/ude_models/leaning_parameter2.jld2", "p")
prob = ODEProblem(ude_hh!, u_0, tspan, pn)
sol = solve(prob, TRBDF2(), reltol=1e-6, abstol=1e-6, saveat=t_steps, sensealg=ForwardDiffSensitivity())
t = sol.t
V = sol[1, :]

p = plot(title="UDE model Vs HH_model", t, V, xlabel="ude_time", ylabel="ude_volatage", label="ude_model", lw=3, lc=:blue)
plot!(t_steps, true_V, label="HH_model", lw=3, lc=:red, ls=:dashdot)
# ----

