using DataFrames, DifferentialEquations, Plots, CSV, Random, Lux, Optimization, OptimizationOptimisers, Zygote
using ComponentArrays, SciMLSensitivity, JLD2, StaticArrays, ForwardDiff
data_loaded = CSV.read("C:/Ude_HH_model/ude_models/synthetic_data.csv", DataFrame)
# all constants
true_V = Float32.(data_loaded[!, "V"])
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



rng = Random.seed!(42)


nn = Chain(Dense(1 => 32),
    Dense(32 => 32, tanh),
    Dense(32 => 1, sigmoid))
ps, st = Lux.setup(rng, nn)

function ude_hh!(du, u_0, p, t)

    V, m, h, n = u_0
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

function loss_function(ps, p)


    prob = ODEProblem(ude_hh!, u_0, tspan, ps)
    sol = solve(prob, Tsit5(), reltol=1e-6, abstol=1e-6, saveat=t_steps, sensealg=ForwardDiffSensitivity())
    pred_V = sol[1, :]
    loss = sum(abs2, pred_V - true_V) / length(true_V)
    return loss

end

# Optimization

# a callback function
iter = 0

function callback(state, l)
    if state.iter % 5 == 0
        println("current iteration = $(state.iter)  | current_loss = $(l)")
        if state.iter % 250 == 0
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

println("Lets start the trainig ====================================== ")
solve(opt_prob, Adam(0.05), callback=callback, maxiters=10)


# --------------------------------------------

