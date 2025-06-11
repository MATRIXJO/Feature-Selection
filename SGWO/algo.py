import numpy as np
import scipy.io
import time
import pandas as pd
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score

# —————————————————————————————————————————————
# 1. Algorithm parameters (Sec. 5)
# —————————————————————————————————————————————
POP_SIZE = 10           # N (Eq. 24) :contentReference[oaicite:0]{index=0}
MAX_ITER = 50           # Maxit (Eq. 21) :contentReference[oaicite:1]{index=1}
K = 5                   # KNN neighbors
α = 0.5                 # trade-off parameter (Eq. 26) :contentReference[oaicite:2]{index=2}
β = 1 - α
Sp_speed = 343          # sound speed in medium (Eq. 1) :contentReference[oaicite:3]{index=3}
g = 9.81                # gravity constant (Eq. 4) :contentReference[oaicite:4]{index=4}

# —————————————————————————————————————————————
# 2. S-shaped transfer function (Eq. 22–23) :contentReference[oaicite:5]{index=5}
# —————————————————————————————————————————————
def s_shaped(x):
    return 1.0 / (1.0 + np.exp(-x))

# —————————————————————————————————————————————
# 3. Initialization (Eq. 24) :contentReference[oaicite:6]{index=6}
#    Xi,j = (UBj–LBj)*rand + LBj, here LBj=0, UBj=1
# —————————————————————————————————————————————
def init_population(n_feat):
    return np.random.rand(POP_SIZE, n_feat)

# —————————————————————————————————————————————
# 4. FOX update (Eqs. 1–9) 
# —————————————————————————————————————————————
def fox_update(pop, best_cont, t):
    n, d = pop.shape
    # track all Time_S for MinT calc
    time_S_all = np.zeros((n, d))
    new_pop = np.zeros_like(pop)

    for i in range(n):
        p = np.random.rand()
        if p >= 0.5:  # Exploitation branch :contentReference[oaicite:7]{index=7}
            # Eqs. 1–3: sound-travel distance & fox-prey distance
            Time_S = np.random.rand(d)
            Dist_S = (Sp_speed * Time_S)  # Eq. 1
            Dist_FP = Dist_S * 0.5        # Eq. 3
            # Eq. 4: jump height
            Jump = 0.5 * g * (Time_S**2)  # Eq. 4
            # Eq. 5–6: direction jump
            c1 = np.random.uniform(0, 0.18, size=d)
            c2 = np.random.uniform(0.19, 1.0, size=d)
            if np.random.rand() > 0.18:
                new_pop[i] = Dist_FP * Jump * c1  # Eq. 5
            else:
                new_pop[i] = Dist_FP * Jump * c2  # Eq. 6
            time_S_all[i] = Time_S
        else:
            # Exploration (Eq. 7–9) :contentReference[oaicite:8]{index=8}
            # compute MinT
            Time_S = np.random.rand(d)
            tt = Time_S.sum() / d            # Eq. 7 numerator/dimension
            MinT = tt                        # Eq. 7
            a = 2 * (1 - t/MAX_ITER)         # Eq. 8
            new_pop[i] = best_cont * np.random.rand(d) * MinT * a  # Eq. 9
            time_S_all[i] = Time_S

    return new_pop, time_S_all

# —————————————————————————————————————————————
# 5. GWO update (Eqs. 10–21) 
# —————————————————————————————————————————————
def gwo_update(pop, fitness, t):
    # identify alpha, beta, delta (three best) continuous positions
    idx = np.argsort(fitness)
    Xα, Xβ, Xδ = pop[idx[0]], pop[idx[1]], pop[idx[2]]
    a_vec = 2 * (1 - t/MAX_ITER)          # Eq. 21

    new_pop = np.zeros_like(pop)
    for i, X in enumerate(pop):
        # coefficient vectors for each leader
        r1, r2 = np.random.rand(), np.random.rand()
        A1 = 2 * a_vec * r1 - a_vec      # Eq. 12
        C1 = 2 * np.random.rand()        # Eq. 13
        Dα = np.abs(C1 * Xα - X)         # Eq. 11
        x1 = Xα - A1 * Dα                # Eq. 15

        A2 = 2 * a_vec * np.random.rand() - a_vec
        C2 = 2 * np.random.rand()
        Dβ = np.abs(C2 * Xβ - X)
        x2 = Xβ - A2 * Dβ                # Eq. 16

        A3 = 2 * a_vec * np.random.rand() - a_vec
        C3 = 2 * np.random.rand()
        Dδ = np.abs(C3 * Xδ - X)
        x3 = Xδ - A3 * Dδ                # Eq. 17

        new_pop[i] = (x1 + x2 + x3) / 3   # Eq. 14

    return new_pop

# —————————————————————————————————————————————
# 6. Fitness (Eq. 26) & binary conversion (Eq. 25) :contentReference[oaicite:9]{index=9}
# —————————————————————————————————————————————
def evaluate(pop_cont, X_tr, y_tr, X_te, y_te):
    # binary mask
    pop_bin = (pop_cont >= 0.5).astype(int)  # Eq. 25
    fitness = np.zeros(POP_SIZE)
    for i, mask in enumerate(pop_bin):
        sel = np.where(mask == 1)[0]
        if sel.size == 0:
            fitness[i] = np.inf
            continue
        clf = KNeighborsClassifier(n_neighbors=K)
        clf.fit(X_tr[:, sel], y_tr)
        y_pred = clf.predict(X_te[:, sel])
        err = 1 - accuracy_score(y_te, y_pred)      # error rate
        ratio = sel.size / X_tr.shape[1]
        fitness[i] = α * err + β * ratio           # Eq. 26
    return pop_bin, fitness

# —————————————————————————————————————————————
# 7. Full FOX-GWO loop
# —————————————————————————————————————————————
def fox_gwo(X_tr, y_tr, X_te, y_te):
    n_feat = X_tr.shape[1]
    pop = init_population(n_feat)
    best_idx = None
    best_cont = None
    best_bin = None
    best_fit = np.inf

    # initial eval
    pop_bin, fit = evaluate(pop, X_tr, y_tr, X_te, y_te)
    best_idx = np.argmin(fit)
    best_cont = pop[best_idx].copy()
    best_bin = pop_bin[best_idx].copy()
    best_fit = fit[best_idx]

    for t in range(1, MAX_ITER + 1):
        # FOX update
        fox_pop, _ = fox_update(pop, best_cont, t)
        # GWO update
        gwo_pop = gwo_update(fox_pop, fit, t)
        pop = gwo_pop

        # evaluate
        pop_bin, fit = evaluate(pop, X_tr, y_tr, X_te, y_te)
        cur = np.argmin(fit)
        if fit[cur] < best_fit:
            best_idx, best_fit = cur, fit[cur]
            best_cont = pop[cur].copy()
            best_bin = pop_bin[cur].copy()

    return best_bin

# —————————————————————————————————————————————
# 8. Running on multiple .mat datasets and saving to CSV
# —————————————————————————————————————————————

results = []

datasets = [
    ("../Dataset/Leukemia_1.mat",      "Leukemia1"),
    ("../Dataset/DLBCL.mat",           "DLBCL"),
    ("../Dataset/Brain_Tumor_1.mat",   "Brain_Tumor_1"),
    ("../Dataset/Prostate_Tumor_1.mat","Prostate_Tumor_1"),
    ("../Dataset/nci9.mat",            "nci9"),
    ("../Dataset/Leukemia_3.mat",      "Leukemia_3"),
    ("../Dataset/CLL_SUB_111.mat",     "CLL_SUB_111"),
    ("../Dataset/Lung_Cancer.mat",     "Lung_Cancer"),
    ("../Dataset/SMK_CAN_187.mat",     "SMK_CAN_187"),
    ("../Dataset/GLI_85.mat",          "GLI_85")
]

for path, name in datasets:
    # 1. Load
    data = scipy.io.loadmat(path)
    X, y = data['X'], data['Y'].ravel()
    # 2. Split
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

    # 3. Run exact FOX-GWO
    start = time.time()
    mask = fox_gwo(X_tr, y_tr, X_te, y_te)
    elapsed = time.time() - start

    # 4. Metrics
    sel_count = mask.sum()
    tot_count = X.shape[1]

    clf = KNeighborsClassifier(n_neighbors=K)
    clf.fit(X_tr[:, mask==1], y_tr)
    y_pred = clf.predict(X_te[:, mask==1])
    acc  = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, average='weighted', zero_division=0)

    results.append({
        'dataset': name,
        'accuracy': acc,
        'precision': prec,
        'time_sec': elapsed,
        'features_selected': sel_count,
        'total_features': tot_count
    })
    print(name);
    print('Accuracy:', acc)
    print('Precision:', prec)
    print('Time:', elapsed)
    print('Features Selected:', sel_count)
    print('Total Features:', tot_count,"\n\n")

# 5. Save all results
df = pd.DataFrame(results)

csv_path = "../results/fox_gwo_results.csv"
file_exists = os.path.isfile(csv_path)

df.to_csv(
    csv_path,
    mode='a',               # append mode
    header=not file_exists, # write header only if file didn’t exist
    index=False
)
