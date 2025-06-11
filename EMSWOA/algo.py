# EMSWOA: Exact implementation based on the paper (Dec 2024)
# Includes: dynamic multi-swarm, elite tuning, LSD mechanism, full WOA behavior

import numpy as np
import csv
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score
from sklearn.preprocessing import MinMaxScaler
from scipy.io import loadmat
from time import time

def sigmoid(x):
    return 1 / (1 + np.exp(-2 * x))

def binary_conversion(position, delta_position):
    prob = sigmoid(delta_position)
    return np.where(np.random.rand(*prob.shape) < prob, 1, 0)

def fitness_function(X, y, feature_subset, alpha=0.99):
    if np.sum(feature_subset) == 0:
        return 1, 0, 0

    selected_X = X[:, feature_subset == 1]
    skf = StratifiedKFold(n_splits=5)
    accs, precs = [], []

    for train_idx, test_idx in skf.split(selected_X, y):
        clf = SVC(kernel='linear')
        clf.fit(selected_X[train_idx], y[train_idx])
        y_pred = clf.predict(selected_X[test_idx])
        accs.append(accuracy_score(y[test_idx], y_pred))
        precs.append(precision_score(y[test_idx], y_pred, average='macro', zero_division=0))

    error = 1 - np.mean(accs)
    reduction = np.sum(feature_subset) / X.shape[1]
    return alpha * error + (1 - alpha) * reduction, np.mean(accs), np.mean(precs)

def local_sparse_density(fitness_values):
    sorted_idx = np.argsort(fitness_values)
    n = len(fitness_values)
    l1 = np.zeros(n)
    l2 = np.zeros(n)
    L = np.ptp(fitness_values)
    L = np.ptp(fitness_values)
    if L == 0:
        return np.zeros_like(fitness_values)

    for i in range(1, n-1):
        l1[i] = fitness_values[sorted_idx[i]] - fitness_values[sorted_idx[i-1]]
        l2[i] = fitness_values[sorted_idx[i+1]] - fitness_values[sorted_idx[i]]

    con = (l1 + l2) / L
    dis = np.minimum(l1, l2) / (np.maximum(l1, l2) + 1e-6)
    con = con / (np.max(con) + 1e-6)
    dis = dis / (np.max(dis) + 1e-6)
    lsd = con / (dis + 1e-6)
    lsd[0] = lsd[-1] = 0
    return lsd

def EMSWOA(X, y, max_iter=100, pop_size=30, alpha=0.99, name="Dataset"):
    dim = X.shape[1]
    b = 1
    group_min, group_max, R = 1, 20, 10
    lambda_thr = 0.5
    a_linear = lambda t: 2 - t * (2 / max_iter)

    position = np.random.randint(0, 2, (pop_size, dim))
    fitness = np.zeros(pop_size)
    accs, precs = np.zeros(pop_size), np.zeros(pop_size)

    for i in range(pop_size):
        fitness[i], accs[i], precs[i] = fitness_function(X, y, position[i], alpha)

    best_idx = np.argmin(fitness)
    best_position = position[best_idx].copy()
    best_fitness = fitness[best_idx]
    best_acc = accs[best_idx]
    best_prec = precs[best_idx]
    prev_best_position = best_position.copy()

    start = time()

    for t in range(max_iter):
        a = a_linear(t)
        r = np.random.rand(pop_size, dim)
        A = 2 * a * r - a
        C = 2 * r

        if t % R == 0:
            k = int(group_max - (group_max - group_min) * (t / max_iter))
            swarm_indices = np.array_split(np.random.permutation(pop_size), k)
        else:
            swarm_indices = [np.arange(pop_size)]

        for swarm in swarm_indices:
            local_positions = position[swarm]
            local_fitness = fitness[swarm]
            local_best_idx = np.argmin(local_fitness)
            local_best = local_positions[local_best_idx]
            local_fitness_values = fitness[swarm]
            lsd = local_sparse_density(local_fitness_values)

            for idx, i in enumerate(swarm):
                p = np.random.rand()
                if p < 0.5:
                    if np.abs(A[i][0]) < 1:
                        D = np.abs(C[i] * local_best - position[i])
                        new_pos = local_best - A[i] * D
                    else:
                        rand_idx = np.random.randint(len(swarm))
                        rand_pos = position[swarm[rand_idx]]
                        D = np.abs(C[i] * rand_pos - position[i])
                        new_pos = rand_pos - A[i] * D
                else:
                    D = np.abs(local_best - position[i])
                    l = np.random.uniform(-1, 1)
                    new_pos = D * np.exp(b * l) * np.cos(2 * np.pi * l) + local_best

                delta = new_pos - position[i]
                position[i] = binary_conversion(position[i], delta)
        
        for i in range(pop_size):
            for j in range(dim):
                if best_position[j] == prev_best_position[j]:
                    if best_position[j] == 1 and position[i][j] == 0:
                        position[i][j] = 1 if np.random.rand() > 0.45 else 0
                    elif best_position[j] == 0 and position[i][j] != 0:
                        reduction_ratio = np.sum(position[i]) / dim
                        if np.random.rand() > (0.2 + reduction_ratio):
                            position[i][j] = 0
                else:
                    if best_position[j] == 1 and position[i][j] == 0:
                        position[i][j] = 1 if np.random.rand() < 0.5 else 0

        for i in range(pop_size):
            fitness[i], accs[i], precs[i] = fitness_function(X, y, position[i], alpha)

        cur_best_idx = np.argmin(fitness)
        if fitness[cur_best_idx] < best_fitness:
            prev_best_position = best_position.copy()
            best_position = position[cur_best_idx].copy()
            best_fitness = fitness[cur_best_idx]
            best_acc = accs[cur_best_idx]
            best_prec = precs[cur_best_idx]

    end = time()

    result = {
        "accuracy": best_acc,
        "precision": best_prec,
        "time_taken": end - start,
        "selected_features": int(np.sum(best_position)),
        "total_features": dim
    }

    print(f"Results for {name}:")
    print(f"  Accuracy        : {result['accuracy']*100:.2f}%")
    print(f"  Precision       : {result['precision']*100:.2f}%")
    print(f"  Time Taken      : {result['time_taken']:.2f} seconds")
    print(f"  Features Selected: {result['selected_features']} / {result['total_features']}")

    result["selected_indices"] = np.where(best_position == 1)[0].tolist()
    return result

import warnings

if __name__ == "__main__":
    datasets = [
        ("../Dataset/Leukemia_1.mat", "Leukemia1"),
        ("../Dataset/DLBCL.mat", "DLBCL"),
        ("../Dataset/Brain_Tumor_1.mat", "Brain_Tumor_1"),
        ("../Dataset/Prostate_Tumor_1.mat", "Prostate_Tumor_1"),
        ("../Dataset/nci9.mat", "nci9"),
        ("../Dataset/Leukemia_3.mat", "Leukemia_3"),
        ("../Dataset/CLL_SUB_111.mat", "CLL_SUB_111"),
        ("../Dataset/Lung_Cancer.mat", "Lung_Cancer"),
        ("../Dataset/SMK_CAN_187.mat", "SMK_CAN_187"),
        ("../Dataset/GLI_85.mat", "GLI_85")
    ]

    csv_file = "../results/emswoa.csv"
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Dataset", "Accuracy", "Precision", "Time Taken (s)", "Features Selected", "Total Features"])

        for filepath, name in datasets:
            print(f"\n\nExecuting {name}")
            data = loadmat(filepath)
            X = data["X"]
            y = data["Y"].ravel()

            X = MinMaxScaler().fit_transform(X)

            warning_issued = False
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always", category=UserWarning)
                result = EMSWOA(X, y, max_iter=50, pop_size=30, alpha=0.99, name=name)
                for warning in w:
                    if "The least populated class in y has only" in str(warning.message):
                        print(f"⚠️  Warning in {name}: class size too small for StratifiedKFold (n_splits=10)")
                        warning_issued = True
                        break

            writer.writerow([
                name,
                result["accuracy"] * 100,
                result["precision"] * 100,
                result["time_taken"],
                result["selected_features"],
                result["total_features"]
            ])

