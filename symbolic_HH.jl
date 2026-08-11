
# %%
# all important imports 
using JLD2, DataFrames, CSV




# %%
# Data [X] and [Y]
x = []
for i in 1:3001
    push!(x, i^3 / i^2)
end
(x)

true_data = data_loaded = CSV.read("C:/Ude_HH_model/ude_models/synthetic_data.csv", DataFrame)
true_V = Float32.(true_data[!, "V"])


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

basis_names = ["1", "x", "x^2", "x^3", "x^4", "x^5", "x^6", "x -> x^7", "x^8", "sin(x)", "cos(x)"]

# %% 
## constructing the design matrix (No. fo points) X  (basis funtions)
basis_l = length(basis_func)
point_l = length(x)
ϕ = [basis_func[i](x[j]) for j in 1:point_l, i in 1:basis_l]
size(ϕ)


# %% 
## Define the regression problem

