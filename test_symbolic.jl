# %%
# All important imports 
using JLD2, DataFrames, CSV, StaticArrays, SciMLSensitivity, Lux, LinearAlgebra, DifferentialEquations, Plots, Random, ComponentArrays

# ----------------------------------------------------
# 1. Load Data & Constants
# ----------------------------------------------------
true_data = CSV.read("C:/Ude_HH_model/ude_models/synthetic_data.csv", DataFrame)
t_steps = Float32.(true_data[!, "Time"])

# Physical constants for Hodgkin-Huxley model
g_na = 120.0f0; g_k = 36.0f0; g_l = 0.3f0; c_m = 1.0f0; I_ext = 10.0f0
E_na = 50.0f0; E_k = -77.0f0; E_l = -54.4f0

# Rate functions
alpha_n(V) = abs(V + 55.0f0) < 1.0f-6 ? 0.1f0 : 0.01f0 * (V + 55.0f0) / (1.0f0 - exp(-(V + 55.0f0) / 10.0f0))
beta_n(V) = 0.125f0 * exp(-(V + 65.0f0) / 80.0f0)
alpha_m(V) = abs(V + 40.0f0) < 1.0f-6 ? 1.0f0 : 0.1f0 * (V + 40.0f0) / (1.0f0 - exp(-(V + 40.0f0) / 10.0f0))
beta_m(V) = 4.0f0 * exp(-(V + 65.0f0) / 18.0f0)
alpha_h(V) = 0.07f0 * exp(-(V + 65.0f0) / 20.0f0)
beta_h(V) = 1.0f0 / (1.0f0 + exp(-(V + 35.0f0) / 10.0f0))

# ----------------------------------------------------
# 2. Re-create UDE Neural Network & Solve Trajectory
# ----------------------------------------------------
rng = Random.seed!(42)
nn = Chain(Dense(1 => 32),
    Dense(32 => 32, tanh),
    Dense(32 => 1, sigmoid))
ps, st = Lux.setup(rng, nn)

function ude_hh!(du, u, p, t)
    V, m, h, n = u
    pred_n, _ = nn(@SVector[n], p, st)

    du[1] = 1 / c_m * (I_ext - g_na * m^3 * h * (V - E_na) - g_k * pred_n[1] * (V - E_k) - g_l * (V - E_l))
    du[2] = alpha_m(V) * (1 - m) - beta_m(V) * (m)
    du[3] = alpha_h(V) * (1 - h) - beta_h(V) * (h)
    du[4] = alpha_n(V) * (1 - n) - beta_n(V) * (n)
end

u_0 = [-65.0f0, 0.05f0, 0.6f0, 0.317f0]
tspan = (0.0f0, 30.0f0)

pn = load("C:/Ude_HH_model/ude_models/leaning_parameter2.jld2", "p")
prob = ODEProblem(ude_hh!, u_0, tspan, pn)
sol = solve(prob, TRBDF2(), reltol=1e-6, abstol=1e-6, saveat=t_steps, sensealg=ForwardDiffSensitivity())

x = sol[4, :] # State n trajectory

# ----------------------------------------------------
# 3. Extract Neural Network Predictions (Target y_nn)
# ----------------------------------------------------
point_l = length(x)
y_nn = [nn(@SVector[x[j]], pn, st)[1][1] for j in 1:point_l]

# ----------------------------------------------------
# 4. Method A: Single-Term Candidate Comparison
# ----------------------------------------------------
println("\n=======================================================")
println("  Single-Term Candidate Evaluation (Comparing n^1 to n^8)")
println("=======================================================")
best_loss = Inf
best_k = 0
best_coeff = 0.0

for k in 1:8
    basis_k = x.^k
    coeff_k = basis_k \ y_nn
    loss_k = sum((coeff_k .* basis_k .- y_nn).^2) / point_l
    println("  Model n^$k :  coeff = ", lpad(round(coeff_k, digits=4), 7), " | MSE Loss = ", round(loss_k, digits=6))
    if loss_k < best_loss
        global best_loss = loss_k
        global best_k = k
        global best_coeff = coeff_k
    end
end
println("=======================================================")

# ----------------------------------------------------
# 5. Method B: Greedy Forward Selection (Parsimonious Model)
# ----------------------------------------------------
println("\n=======================================================")
println("  Discovered Parsimonious Model                        ")
println("=======================================================")
println("  nn(n) ≈ ", round(best_coeff, digits=4), " * n^", best_k)
println("=======================================================")

# Plot NN Predictions vs Discovered Symbolic Model
y_fit = best_coeff .* (x.^best_k)
plot(x, y_nn, label="Neural Net Output nn(n)", xlabel="n", ylabel="nn(n)", lw=3, color=:blue, title="UDE Neural Net vs Symbolic Recovery")
plot!(x, y_fit, label="Discovered Model ($(round(best_coeff, digits=2))*n^$best_k)", lw=2, linestyle=:dash, color=:red)

# ----------------------------------------------------
# 6. Method C: Convex.jl + SCS Lasso Optimization
# ----------------------------------------------------
using Convex, SCS

basis_func = [x -> 1.0, x -> x, x -> x^2, x -> x^3, x -> x^4, x -> x^5, x -> x^6, x -> x^7, x -> x^8, x -> sin(x), x -> cos(x)]
basis_names = ["1", "n", "n^2", "n^3", "n^4", "n^5", "n^6", "n^7", "n^8", "sin(n)", "cos(n)"]
basis_l = length(basis_func)
ϕ = [basis_func[i](x[j]) for j in 1:point_l, i in 1:basis_l]

# Step 1: Column Standardization (Crucial for Lasso on polynomial bases)
weights = [norm(ϕ[:, i]) for i in 1:basis_l]
ϕ_norm = ϕ ./ weights'

# Step 2: Formulate Lasso Problem in Convex.jl
λ = 0.05  # Regularization strength
β_var = Convex.Variable(basis_l)
prob = minimize(sumsquares(ϕ_norm * β_var - y_nn) + λ * norm(β_var, 1))

# Step 3: Solve with SCS Optimizer
Convex.solve!(prob, SCS.Optimizer(verbose=0))
β_raw = vec(evaluate(β_var))
β_unnorm = β_raw ./ weights

# Step 4: Debiased Re-Fit (Unregularized OLS on Lasso-selected active terms)
active_idx = findall(abs.(β_unnorm) .> 1e-3)
β_debiased = zeros(basis_l)
if !isempty(active_idx)
    β_debiased[active_idx] = ϕ[:, active_idx] \ y_nn
end

println("\n=======================================================")
println("  Discovered Model (Convex.jl Lasso + Debiased Refit)  ")
println("=======================================================")
for i in 1:basis_l
    if abs(β_debiased[i]) > 1e-4
        println("  ", round(β_debiased[i], digits=4), " * ", basis_names[i])
    end
end
println("=======================================================")

