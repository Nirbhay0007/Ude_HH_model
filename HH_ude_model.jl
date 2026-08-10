using DataFrames ,DifferentialEquations, Plots,CSV,Random,Lux,Optimization,OptimizationOptimisers,Zygote
data_loaded = CSV.read("C:/Modeling_Neurons/ude_models/synthetic_data.csv",DataFrame)
true_V = Array(data_loaded[! , "V"])
# all constants

# all constants
g_na = 120.0
g_k = 36.0
g_l = 0.3
c_m = 1.0
I_ext = 10.0
E_na = 50.0
E_k = -77.0
E_l = -54.4

# rate_functions
# rate funtions

# --- Potassium Gating (n) ---
alpha_n(V) = abs(V + 55.0) < 1e-6 ? 0.1 : 0.01 * (V + 55.0) / (1.0 - exp(-(V + 55.0) / 10.0))
beta_n(V) = 0.125 * exp(-(V + 65.0) / 80.0)


# --- Sodium Activation (m) ---
alpha_m(V) = abs(V + 40.0) < 1e-6 ? 1.0 : 0.1 * (V + 40.0) / (1.0 - exp(-(V + 40.0) / 10.0))
beta_m(V) = 4.0 * exp(-(V + 65.0) / 18.0)

# --- Sodium Inactivation (h) ---
alpha_h(V) = 0.07 * exp(-(V + 65.0) / 20.0)
beta_h(V) = 1.0 / (1.0 + exp(-(V + 35.0) / 10.0))



rng = Random.seed!(42)


nn = Chain(Dense(1=>32),
          Dense(32=>32,tanh),
          Dense(32=>1,sigmoid))
ps, st = Lux.setup(rng, nn)

function ude_hh!(du,u_0,p,t)

    V, m, h ,n= u_0
    g_na, g_k, g_l, c_m, I_ext, E_na, E_k, E_l ,ps,st = p

    pred_n , _ = nn(Float32[n],ps,st)

    




    du[1] = 1 / c_m * (I_ext - g_na * m^3 * h * (V - E_na) - g_k * pred_n[1] * (V - E_k) - g_l * (V - E_l))
    du[2] = alpha_m(V) * (1 - m) - beta_m(V) * (m)
    du[3] = alpha_h(V) * (1 - h) - beta_h(V) * (h)
    du[4] = alpha_n(V) * (1 - n) - beta_n(V) * (n)

    
end



u_0 = [-65.0, 0.05, 0.6, 0.317]
tspan = (0.0, 30.0)
p = [g_na, g_k, g_l, c_m, I_ext, E_na, E_k, E_l,ps,st]
prob = ODEProblem(ude_hh!,u_0,tspan,p)


sol = solve(prob, Tsit5(), reltol=1e-6, abstol=1e-6,adaptive=false,dt = 0.01)

#  creating the loss function
println(size(pred_V))
println(size(true_V))
pred_V = Array(sol[1, :])
true_V = Array(data_loaded[! , "V"])

function loss_function(ps,p)

    updated_p = [g_na, g_k, g_l, c_m, I_ext, E_na, E_k, E_l,ps,st]
    prob = ODEProblem(ude_hh!,u_0,tspan,updated_p)
    sol = solve(prob, Tsit5(), reltol=1e-6, abstol=1e-6,adaptive=false,dt = 0.01)
    pred_V = Array(sol[1, :])
    loss = sum(abs2,pred_V - true_V)/ length(true_V)
    return (loss,pred_V)
    
end

loss_function(ps)

# Optimization

opt = OptimizationFunction(loss_function,AutoZygote())

# Optimization
opt_prob = OptimizationProblem(opt,ps)
