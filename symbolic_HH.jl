# %%

# all important imports 
using JLD2, DataFrames, CSV, StaticArrays, SciMLSensitivity, Lux, LinearAlgebra, DifferentialEquations, Plots

true_data = data_loaded = CSV.read("C:/Ude_HH_model/ude_models/noisy_data.csv", DataFrame)
y = Float32.(true_data[!, "n"])
t_steps = Float32.(true_data[!, "Time"])



rng = Random.seed!(42)





nn = Chain(Dense(1 => 32),
    Dense(32 => 32, tanh),
    Dense(32 => 1, sigmoid))
ps, st = Lux.setup(rng, nn)



using CSV, DataFrames, JLD2, ComponentArrays, DifferentialEquations, SciMLSensitivity, Lux, StaticArrays, Random, LinearAlgebra
using Convex, SCS, Printf



# 1. Load synthetic data and time steps
true_data = CSV.read("C:/Ude_HH_model/ude_models/synthetic_data.csv", DataFrame)
t_steps = Float32.(true_data[!, "Time"])
ture_n = Float32.(true_data[!, "n"])

# 2. Hodgkin-Huxley Model Parameters

const g_na = 120.0f0
const g_k = 36.0f0
const g_l = 0.3f0
const c_m = 1.0f0
const I_ext = 10.0f0
const E_na = 50.0f0
const E_k = -77.0f0
const E_l = -54.4f0

# 3. Rate Functions for Gating Variables
alpha_n(V) = abs(V + 55.0f0) < 1.0f-6 ? 0.1f0 : 0.01f0 * (V + 55.0f0) / (1.0f0 - exp(-(V + 55.0f0) / 10.0f0))
beta_n(V) = 0.125f0 * exp(-(V + 65.0f0) / 80.0f0)

alpha_m(V) = abs(V + 40.0f0) < 1.0f-6 ? 1.0f0 : 0.1f0 * (V + 40.0f0) / (1.0f0 - exp(-(V + 40.0f0) / 10.0f0))
beta_m(V) = 4.0f0 * exp(-(V + 65.0f0) / 18.0f0)

alpha_h(V) = 0.07f0 * exp(-(V + 65.0f0) / 20.0f0)
beta_h(V) = 1.0f0 / (1.0f0 + exp(-(V + 35.0f0) / 10.0f0))

# 4. Neural Network Architecture & State Setup
nn = Chain(
    Dense(1 => 32),
    Dense(32 => 32, tanh),
    Dense(32 => 1, sigmoid)
)
rng = Random.seed!(42)
_, st = Lux.setup(rng, nn)

# 5. Load Trained Parameters
ps = load("C:/Ude_HH_model/ude_models/leaning_parameter2.jld2", "p")

# 6. Universal Differential Equations System

function ude_hh!(du, u, p, t)
    V, m, h, n = u
    ps = p

    pred_n, _ = nn(@SVector[n], ps, st)

    du[1] = 1 / c_m * (I_ext - g_na * m^3 * h * (V - E_na) - g_k * pred_n[1] * (V - E_k) - g_l * (V - E_l))
    du[2] = alpha_m(V) * (1 - m) - beta_m(V) * (m)
    du[3] = alpha_h(V) * (1 - h) - beta_h(V) * (h)
    du[4] = alpha_n(V) * (1 - n) - beta_n(V) * (n)
end

# 7. Initial Conditions & ODE Solution
const u_0 = [-65.0f0, 0.05f0, 0.6f0, 0.317f0]
tspan = (0.0f0, 30.0f0)

prob = ODEProblem(ude_hh!, u_0, tspan, ps)
sol = solve(prob, Rosenbrock23(), reltol=1e-6, abstol=1e-6, saveat=t_steps, sensealg=ForwardDiffSensitivity())

sol.t
n = sol[4, :]
y = Float64[nn(@SVector[n[j]], ps, st)[1][1] for j in 1:length(n)]

# 9. Diverse Candidate Basis Functions Library (28 Functions)
basis_func = [
    n -> 1.0,
    n -> n,
    n -> n^2,
    n -> n^3,
    n -> n^4,
    n -> n^5,
    n -> n^6,
    n -> n^7,
    n -> n^8,
    n -> sqrt(max(0.0, n)),
    n -> sin(n),
    n -> cos(n),
    n -> sin(2 * n),
    n -> cos(2 * n),
    n -> sin(3 * n),
    n -> cos(3 * n),
    n -> sin(pi * n),
    n -> cos(pi * n),
    n -> exp(n),
    n -> exp(-n),
    n -> exp(2 * n),
    n -> exp(-2 * n),
    n -> sinh(n),
    n -> cosh(n),
    n -> tanh(n),
    n -> log(1.0 + max(0.0, n)),
    n -> 1.0 / (1.0 + n),
    n -> n / (1.0 + n)
]

basis_names = [
    "1", "n", "n^2", "n^3", "n^4", "n^5", "n^6", "n^7", "n^8",
    "sqrt(n)",
    "sin(n)", "cos(n)", "sin(2n)", "cos(2n)", "sin(3n)", "cos(3n)", "sin(πn)", "cos(πn)",
    "exp(n)", "exp(-n)", "exp(2n)", "exp(-2n)", "sinh(n)", "cosh(n)", "tanh(n)",
    "log(1+n)", "1/(1+n)", "n/(1+n)"
]

# 10. Construct Design Matrix (ϕ)
basis_l = length(basis_func)
point_l = length(n)
ϕ = [basis_func[i](n[j]) for j in 1:point_l, i in 1:basis_l]
y = Float64[nn(@SVector[n[j]], ps, st)[1][1] for j in 1:length(n)]
num_basis = size(ϕ, 2)
β = ϕ \ y



# using lasse regresser
λ = 0.1 # lasso strength

# normalizing the desing matrix to that low value do get squezied by λ each element in ϕ
col_norms = [norm(ϕ[:, i]) for i in 1:size(ϕ, 2)]
ϕ_std = ϕ ./ col_norms'


using Variable
β_std = Variable(4)
problem = minimize(sumsquares(ϕ_std * β_std - y) + λ * norm(β_std, 1))

# Solve using SCS solver
Convex.solve!(problem, SCS.Optimizer; silent=true)

β_lasso = vec(evaluate(β_std)) ./ col_norms

active_mask = abs.(β_lasso) .> 0.1
β_opt = zeros(num_basis)
if sum(active_mask) > 0
    β_opt[active_mask] = ϕ[:, active_mask] \ y
end


println("\n" * "="^45)
println("      SYMBOLIC REGRESSION RESULTS             ")
println("="^45)

println("\nCalculated Basis Coefficients (β):")
for (name, val) in zip(basis_names, β_opt)
    @printf("  %-5s : %10.5f\n", name, val)
end


pn = load("C:/Ude_HH_model/ude_models/leaning_parameter2.jld2", "p")
prob = ODEProblem(ude_hh!, u_0, tspan, pn)
sol = solve(prob, Rosenbrock23(), reltol=1e-6, abstol=1e-6, saveat=t_steps, sensealg=ForwardDiffSensitivity())
sol.t

# Filter out small numerical noise (< 0.01 threshold)
threshold = 1e-2
active_terms = String[]


for (name, val) in zip(basis_names, β_opt)
    if abs(val) > threshold
        push!(active_terms, @sprintf("%.4f * %s", val, name))
    end
end

recovered_eq = isempty(active_terms) ? "0.0" : join(active_terms, " + ")


# Defining our basis funtions
basis_func =
    [x -> 1.0,
        x -> x,
        x -> x^2,
        x -> x^3,
        x -> x^4,
        x -> x^5,
        x -> x^6,
        x -> x^7,
        x -> x^8,
        x -> sin(x),
        x -> cos(x)]

basis_names = ["1", "n", "n^2", "n^3", "n^4", "n^5", "n^6", "n -> n^7", "n^8", "sin(n)", "cos(n)"]

# %% 
## constructing the design matrix (No. fo points) X  (basis funtions)
basis_l = length(basis_func)
point_l = length(x)
ϕ = [basis_func[i](x[j]) for j in 1:point_l, i in 1:basis_l]
size(ϕ)

β = ϕ \ y
for i in 1:basis_l

    println(" ", round(β[i], digits=4), " *", " ", basis_names[i])
end

# Compute fitted curve
y_fit = ϕ * β
plot(t_steps, y, label="Original Data (y)", xlabel="Time", ylabel="n", lw=2)
plot!(t_steps, y_fit, label="Fitted Model (y_fit)", lw=2, linestyle=:dash)

println("\n" * "-"^45)
println("Recovered Function for pred_n:")
println("  f(n) ≈ ", recovered_eq)
println("  True HH Potassium Gating Term: n^4")
println("-"^45)

