import csv, json
from statistics import mean

# ── Load data ──────────────────────────────────────────────────────────────
rows = []
with open("delivery_time.csv") as f:
    for r in csv.DictReader(f):
        rows.append(r)

# ── Feature engineering (same one-hot encoding as sklearn OHE drop-first) ──
# Numeric: Distance_km, Num_Items
# Time_of_Day: Afternoon, Evening, Morning, Night  → drop Afternoon as baseline
# Weather:     Cloudy, Rainy, Sunny               → drop Cloudy as baseline
# Peak_Hour:   Yes=1, No=0

def encode(r):
    d   = float(r["Distance_km"])
    ni  = float(r["Num_Items"])
    tod = r["Time_of_Day"]
    wx  = r["Weather"]
    pk  = 1.0 if r["Peak_Hour"] == "Yes" else 0.0
    y   = float(r["Delivery_Time_min"])
    # Time_of_Day dummies (baseline = Afternoon)
    tod_eve  = 1.0 if tod == "Evening"   else 0.0
    tod_morn = 1.0 if tod == "Morning"   else 0.0
    tod_ngt  = 1.0 if tod == "Night"     else 0.0
    # Weather dummies (baseline = Cloudy)
    wx_rainy = 1.0 if wx == "Rainy" else 0.0
    wx_sunny = 1.0 if wx == "Sunny" else 0.0
    # Feature vector (intercept handled separately)
    x = [d, ni, tod_eve, tod_morn, tod_ngt, wx_rainy, wx_sunny, pk]
    return x, y

X, Y = [], []
for r in rows:
    x, y = encode(r)
    X.append(x)
    Y.append(y)

n  = len(Y)
nf = len(X[0])

# ── OLS via normal equations  (X'X)^-1 X'y  ─────────────────────────────
# Augment X with bias column (intercept)
Xa = [[1.0] + x for x in X]
p  = nf + 1   # number of parameters incl. intercept

# X'X
XtX = [[sum(Xa[i][a] * Xa[i][b] for i in range(n)) for b in range(p)] for a in range(p)]
# X'y
Xty = [sum(Xa[i][a] * Y[i] for i in range(n)) for a in range(p)]

# Gaussian elimination to solve XtX * beta = Xty
def solve(A, b):
    import copy
    n = len(b)
    M = [A[i][:] + [b[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[pivot] = M[pivot], M[col]
        pv = M[col][col]
        M[col] = [v / pv for v in M[col]]
        for r in range(n):
            if r != col:
                f = M[r][col]
                M[r] = [M[r][c] - f * M[col][c] for c in range(n + 1)]
    return [M[i][-1] for i in range(n)]

beta = solve(XtX, Xty)

# ── R² ───────────────────────────────────────────────────────────────────
y_mean = mean(Y)
y_pred = [sum(beta[j] * Xa[i][j] for j in range(p)) for i in range(n)]
ss_res = sum((Y[i] - y_pred[i]) ** 2 for i in range(n))
ss_tot = sum((Y[i] - y_mean)     ** 2 for i in range(n))
r2     = 1 - ss_res / ss_tot

# ── Output ───────────────────────────────────────────────────────────────
names = ["intercept", "Distance_km", "Num_Items",
         "Time_Evening", "Time_Morning", "Time_Night",
         "Weather_Rainy", "Weather_Sunny", "Peak_Hour_Yes"]

result = {
    "r2": round(r2, 6),
    "coefficients": {names[i]: round(beta[i], 6) for i in range(p)}
}
print(json.dumps(result, indent=2))
