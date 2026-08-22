# Hodgkin-Huxley UDE & Symbolic Discovery

Discovering unknown biophysical equations from neural data using **Universal Differential Equations (UDE)** and **Sparse Symbolic Regression (Lasso)** in Julia.

---

## ⚡ Visual Pipeline

```
+---------------------------+
|    1. DATA GENERATION     |  Simulate 4D Hodgkin-Huxley neuron dynamics
|  (Time, Voltage, m, h, n) |
+-------------+-------------+
              |
              v
+-------------+-------------+
|    2. HYBRID UDE MODEL    |  Replace unknown K+ gating with Neural Network:
|   dV/dt = ... - g_K * NN  |  dV/dt = (I_ext - I_Na - g_K * NN(n) * (V - E_K) - I_L) / C_m
+-------------+-------------+
              |
              v
+-------------+-------------+
|   3. SPARSE REGRESSION    |  Test 28 candidate functions (1, n, n^2, ..., sin, exp, tanh)
|      (Lasso / SINDy)      |  Shrink all irrelevant coefficients to 0
+-------------+-------------+
              |
              v
+-------------+-------------+
|      DISCOVERED LAW       |  
|           n^4             |  Successfully recovered exact biophysical power law!
+---------------------------+
```

---

## 🔬 Step-by-Step Overview

### Step 1: Generate Neuron Data (`ude_models/Hodgkin-Huxley_model.ipynb`)
Simulates action potentials (*V*, *m*, *h*, *n*) over 30 ms and saves synthetic voltage recordings.

```
Voltage (mV)
  30 |        /\
   0 |       /  \
 -30 |      /    \
 -65 +----+/      \+--------  Time (ms)
```

---

### Step 2: Train Neural ODE (`Hodgkin-Huxley_UDE_model.ipynb`)
Combines known biophysical equations with a neural network to learn the missing potassium dynamics.

```
       [ Known Physics: Sodium & Leak currents ]
                          +
       [ Machine Learning: g_K * NN(n) * (V - E_K) ]
                          |
                          v
               [ Output: Voltage Trajectory ]
```

---

### Step 3: Extract the Symbolic Equation (`symbolic_HH.ipynb`)
Uses **Lasso (L1 Regularization)** on a 28-candidate basis library to isolate the true underlying formula.

```
Candidate Library (28 terms)           Lasso Selection            Final Formula
----------------------------           ---------------            -------------
1, n, n^2, n^3, n^4, ..., sin, exp  -->  [ 0, 0, 0, 0, 1.0, ..., 0 ]  -->      n^4
```

---

## 📁 Notebook Execution Order

1. `ude_models/Hodgkin-Huxley_model.ipynb` — Generate data
2. `Hodgkin-Huxley_UDE_model.ipynb` — Train UDE
3. `symbolic_HH.ipynb` — Recover symbolic formula
