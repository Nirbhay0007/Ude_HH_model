using DataFrames, DifferentialEquations, Plots, CSV, Random, Lux, Optimization
using OptimizationOptimisers: Adam
using OptimizationOptimJL: BFGS
using Zygote, ComponentArrays, SciMLSensitivity, JLD2, StaticArrays, ForwardDiff, StaticArrays



# loading the data
data_loaded = CSV.read("C:/Ude_HH_model/ude_models/synthetic_data.csv", DataFrame)

names(data_loaded)
# all constants
true_n = Float32.(data_loaded[!, "n"])
true_V = Float32.(data_loaded[!, "V"])

true_m = Float32.(data_loaded[!, "m"])
true_h = Float32.(data_loaded[!, "h"])
t_steps = Float32.(data_loaded[!, "Time"])
# all constants
const g_na = 120.0f0
const g_k = 36.0f0
const g_l = 0.3f0
const c_m = 1.0f0
const I_ext = 10.0f0
const E_na = 50.0f0
const E_k = -77.0f0
const E_l = -54.4f0
const u_0 = Float32[-65.0f0, 0.05f0, 0.6f0, 0.317f0]



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

# Adding noise to the data

using Statistics
rng = Random.seed!(42)
Random.seed!(42)
noise_std_V = 1.5f0     # 1.5 mV noise (or 0.05f0 * std(true_V))
noise_std_n = 0.005f0    # 2% gating probability noise
noise_std_m = 0.015f0
noise_std_h = 0.015f0


noisy_V = true_V .+ noise_std_V .* randn(Float32, size(true_V))
noisy_n = (true_n .+ noise_std_n .* randn(Float32, size(true_n)))
noisy_h = (true_h .+ noise_std_n .* randn(Float32, size(true_n)))
noisy_m = (true_m .+ noise_std_m .* randn(Float32, size(true_m)))


noise_n = clamp.(noisy_n, 0.0f0, 1.0f0)
noise_m = clamp.(noisy_m, 0.0f0, 1.0f0)
noise_h = clamp.(noisy_h, 0.0f0, 1.0f0)




# ploing the noise vs ture data 

v_plot = plot(t_steps, true_V, label=" ture_V")
plot(v_plot, t_steps, noisy_V, seriestype=:scatter, markersize=2, label="nosiy_V")

n_plot = plot(t_steps, true_n, label=" ture_n")
plot(n_plot, t_steps, noisy_n, seriestype=:scatter, markersize=2, label="nosiy_n")

m_plot = plot(t_steps, true_m, label=" ture_m")
plot(m_plot, t_steps, noisy_m, seriestype=:scatter, markersize=2, label="nosiy_m")

h_plot = plot(t_steps, true_h, label=" ture_h")
plot(h_plot, t_steps, noisy_h, seriestype=:scatter, markersize=2, label="nosiy_h")

# Defing the neural network sturcture 
nn = Chain(Dense(1 => 32),
    Dense(32 => 32, tanh),
    Dense(32 => 1, sigmoid))

ps, st = Lux.setup(rng, nn)


# Definng ude funtion


function ude_hh(u, ps, t)
    V, m, h, n = u
    pred_n, _ = nn(SA[n], ps, st)

    dV = (I_ext - g_na * m^3 * h * (V - E_na) - g_k * pred_n[1] * (V - E_k) - g_l * (V - E_l)) / c_m
    dm = alpha_m(V) * (1.0f0 - m) - beta_m(V) * m
    dh = alpha_h(V) * (1.0f0 - h) - beta_h(V) * h
    dn = alpha_n(V) * (1.0f0 - n) - beta_n(V) * n

    return @SVector [dV, dm, dh, dn]
end

# Define with StaticArray initial conditions:
const u_0 = @SVector [-65.0f0, 0.05f0, 0.6f0, 0.317f0]
tspan = (0.0f0, 30.0f0)



# inital values and parameters



p = [g_na, g_k, g_l, c_m, I_ext, E_na, E_k, E_l, ps, st]
prob = ODEProblem(ude_hh, u_0, tspan, p)

# definig the loss function

function loss_function(ps, p)


    prob = ODEProblem(ude_hh, u_0, tspan, ps)
    sol = solve(prob, AutoTsit5(Rosenbrock23()), reltol=1e-4, abstol=1e-4, saveat=t_steps, sensealg=ForwardDiffSensitivity())
    if sol.retcode != SciMLBase.ReturnCode.Success || length(sol.t) != length(true_V)
        return eltype(ps)(1e5)
    end
    pred_V = sol[1, :]
    loss = sum(abs2, pred_V - true_V) / length(true_V)
    return loss

end
const loss_history = []
const iter_history = []
function callback(state, l)

    push!(loss_history, Float32(1))
    push!(iter_history, state.iter)
    if state.iter % 10 == 0
        println("current iteration = $(state.iter)  | current_loss = $(l)")
        if state.iter % 100 == 0
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

# using the previous optimized parameters

using JLD2

# load the params 
checkpoint_path = "C:/Ude_HH_model/ude_models/leaning_parameter.jld2"
saved_p = JLD2.load(checkpoint_path, "p")
p_init = ComponentArray(saved_p)

opt = OptimizationFunction(loss_function, AutoForwardDiff())
opt_prob_resumed = OptimizationProblem(opt, p_init)



println("--------------------- Resuming the trainig --------------------- ")

res_resumed = solve(opt_prob_resumed, Adam(0.001), callback=callback, maxiters=500)

































opt_prob_resumed2 = remake(opt_prob_resumed, u0=res_resumed.u)
res_resumed2 = solve(opt_prob_resumed2, LBFGS(), callback=callback, maxiters=500)

opt_prob_resumed3 = remake(opt_prob_resumed, u0=res_resumed2.u)
res_resumed3 = solve(opt_prob_resumed3, Adam(0.0025), callback=callback, maxiters=300)



# 3rd optimization loop: LBFGS -- 500 iterations
println("--- Phase 4: BFGS [500 maxiters] ---")
opt_prob_resumed4 = remake(opt_prob_resumed3, u0=res_resumed3.u)
res3 = solve(opt_prob_resumed4, BFGS(), callback=callback, maxiters=500)



# -------------------------------------------
# 1st optimization loop: Adam(0.05) -- 1500 iterations

println("--- Phase 1: Adam(0.05) [1500 maxiters] ---")
res1 = solve(opt_prob, Adam(0.005), callback=callback, maxiters=500)

# 2nd optimization loop: Adam(0.01) -- 1000 iterations
println("--- Phase 2: Adam(0.01) [1000 maxiters] ---")
opt_prob2 = remake(opt_prob, u0=res1.u)
res2 = solve(opt_prob2, Adam(0.0001), callback=callback, maxiters=2000)

# 3rd optimization loop: LBFGS -- 500 iterations
println("--- Phase 3: LBFGS [500 maxiters] ---")
opt_prob3 = remake(opt_prob, u0=res2.u)
res3 = solve(opt_prob3, LBFGS(), callback=callback, maxiters=500)

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
plot(iter_history, loss_history)
