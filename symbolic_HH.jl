
# %%
# all important imports 
using JLD2, DataFrames, CSV, StaticArrays, SciMLSensitivity, Lux, LinearAlgebra, DifferentialEquations, Plots

true_data = data_loaded = CSV.read("C:/Ude_HH_model/ude_models/synthetic_data.csv", DataFrame)
y = Float32.(true_data[!, "n"])
t_steps = Float32.(true_data[!, "Time"])



rng = Random.seed!(42)


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

pn = load("C:/Ude_HH_model/ude_models/leaning_parameter2.jld2", "p")
prob = ODEProblem(ude_hh!, u_0, tspan, pn)
sol = solve(prob, TRBDF2(), reltol=1e-6, abstol=1e-6, saveat=t_steps, sensealg=ForwardDiffSensitivity())
sol.t

x = sol[4, :]



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