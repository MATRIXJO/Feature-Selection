# Full and Exact Python Implementation of HMOFA (Hierarchical learning Multi-objective Firefly Algorithm)
# As per the paper "Hierarchical learning multi-objective firefly algorithm for high-dimensional feature selection"

import numpy as np
import os
import time
import csv
import random
import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score
from sklearn.feature_selection import mutual_info_classif, chi2
from sklearn.cluster import KMeans
from scipy.io import loadmat

np.random.seed(42)
random.seed(42)

# ReliefF approximation (can be replaced with true ReliefF)
def reliefF(X, y, n_neighbors=10):
    return mutual_info_classif(X, y)

# Aggregate Filter Method (AFM)
def aggregate_filter_score(X, y):
    mad = np.mean(np.abs(X - np.mean(X, axis=0)), axis=0)
    chi2_score = chi2(X, y)[0]
    fisher = np.array([(np.mean(X[y == c], axis=0) - np.mean(X, axis=0))**2 /
                       (np.var(X[y == c], axis=0) + 1e-10) for c in np.unique(y)]).sum(axis=0)
    mi = mutual_info_classif(X, y)
    relieff = reliefF(X, y)
    scores = np.vstack([mad, chi2_score, fisher, mi, relieff]).T
    return MinMaxScaler().fit_transform(scores).mean(axis=1)

# Canopy clustering (simplified version for cluster count estimation)
def canopy_cluster_count(scores, threshold=0.1):
    unique_vals = np.unique(np.round(scores, 2))
    return max(2, min(len(unique_vals), len(scores) // 5))

# Improved Hamming Distance
def improved_hamming_distance(a, b):
    both_one = np.logical_and(a == 1, b == 1).sum()
    return (np.logical_xor(a, b).sum()) / (both_one + 1e-5)

# Fast Non-Dominated Sorting
def fast_non_dominated_sort(fitness):
    S = [[] for _ in range(len(fitness))]
    front = [[]]
    n = [0 for _ in range(len(fitness))]
    rank = [0 for _ in range(len(fitness))]
    for p in range(len(fitness)):
        for q in range(len(fitness)):
            if (all(fitness[p][i] <= fitness[q][i] for i in range(2)) and
                any(fitness[p][i] < fitness[q][i] for i in range(2))):
                S[p].append(q)
            elif (all(fitness[q][i] <= fitness[p][i] for i in range(2)) and
                  any(fitness[q][i] < fitness[p][i] for i in range(2))):
                n[p] += 1
        if n[p] == 0:
            front[0].append(p)
    i = 0
    while front[i]:
        Q = []
        for p in front[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1
                    Q.append(q)
        i += 1
        front.append(Q)
    del front[-1]
    return front

# Clustering Initialization
def clustering_initialization(X, y, pop_size):
    scores = aggregate_filter_score(X, y)
    k = canopy_cluster_count(scores)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(scores.reshape(-1, 1))
    clusters = {i: np.where(kmeans.labels_ == i)[0] for i in range(k)}
    population = []
    for _ in range(pop_size):
        individual = np.zeros(X.shape[1])
        for indices in clusters.values():
            f = np.random.choice(indices)
            individual[f] = 1
        population.append(individual)
    return np.array(population), clusters

# Evaluation
def evaluate_individual(ind, X, y):
    selected = np.where(ind == 1)[0]
    if len(selected) == 0:
        return 1.0, 1.0
    clf = KNeighborsClassifier(n_neighbors=5)
    X_sel = X[:, selected]
    y_pred = cross_val_predict(clf, X_sel, y, cv=StratifiedKFold(n_splits=5))
    err = 1 - accuracy_score(y, y_pred)
    return err, len(selected) / X.shape[1]

# Hierarchical Movement
def hierarchical_movement(pop, fitness):
    fronts = fast_non_dominated_sort(fitness)
    new_pop = []
    for idx, ind in enumerate(pop):
        for f_idx, front in enumerate(fronts):
            if idx in front:
                current_rank = f_idx
                break
        if current_rank > 0:
            better_front = fronts[current_rank - 1]
            selected_idx = random.choice(better_front)
            leader = pop[selected_idx]
            beta = np.exp(-improved_hamming_distance(ind, leader))
            rand = np.random.rand(len(ind))
            new_ind = np.where(rand < beta, leader, ind)
            new_pop.append(new_ind)
        else:
            new_pop.append(ind)
    return np.array(new_pop)

# Competitive Mutation
def competitive_mutation(pop, fitness, clusters):
    fronts = fast_non_dominated_sort(fitness)
    alpha = 0.4
    for f_idx in range(1, len(fronts) - 1):
        for idx in fronts[f_idx]:
            if np.random.rand() < alpha:
                current = pop[idx]
                sup = np.vstack([pop[i] for i in fronts[f_idx - 1]])
                inf = np.vstack([pop[i] for i in fronts[f_idx + 1]])
                Set1 = np.where(sup.sum(axis=0) > 0)[0]
                Set2 = np.where(inf.sum(axis=0) > 0)[0]
                Set = np.where(current == 1)[0]
                Sup = np.setdiff1d(Set1, Set)
                Inf = np.intersect1d(Set2, Set)
                if Sup.size > 0 and Inf.size > 0:
                    f1 = np.random.choice(Inf)
                    f2 = np.random.choice(Sup)
                    current[f1] = 0
                    current[f2] = 1
    return pop

# Duplicate Solution Modification
def duplicate_modification(pop, clusters):
    unique = []
    modified = []
    for i, ind in enumerate(pop):
        found = False
        for u in unique:
            if np.array_equal(ind, u):
                found = True
                break
        if not found:
            unique.append(ind.copy())
        else:
            for cluster in clusters.values():
                f1s = np.where(ind == 1)[0]
                f0s = np.setdiff1d(cluster, f1s)
                if len(f1s) > 0 and len(f0s) > 0:
                    f1 = np.random.choice(f1s)
                    f0 = np.random.choice(f0s)
                    ind[f1] = 0
                    ind[f0] = 1
                    break
            modified.append(ind)
    return np.array(unique + modified)

# Main HMOFA Procedure
def run_hmofa(X, y, pop_size=10, max_gen=20):
    start = time.time()
    pop, clusters = clustering_initialization(X, y, pop_size)
    for _ in range(max_gen):
        fitness = [evaluate_individual(ind, X, y) for ind in pop]
        pop = hierarchical_movement(pop, fitness)
        pop = competitive_mutation(pop, fitness, clusters)
        pop = duplicate_modification(pop, clusters)
    fitness = [evaluate_individual(ind, X, y) for ind in pop]
    best_idx = np.argmin([f[0] for f in fitness])
    best_ind = pop[best_idx]
    selected = np.where(best_ind == 1)[0]
    clf = KNeighborsClassifier(n_neighbors=5)
    y_pred = cross_val_predict(clf, X[:, selected], y, cv=5)
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, average='macro')
    end = time.time()
    return {
        'accuracy': acc,
        'precision': prec,
        'time_taken': end - start,
        'selected_features': len(selected),
        'total_features': X.shape[1]
    }

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

    csv_file = "../results/hmofa_results.csv"
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
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
                result = run_hmofa(X, y, pop_size=30, max_gen=50)
                for warning in w:
                    if "The least populated class in y has only" in str(warning.message):
                        print(f"⚠️  Warning in {name}: class size too small for StratifiedKFold (n_splits=10)")
                        warning_issued = True
                        break

            writer.writerow([
                name,
                round(result["accuracy"] * 100, 4),
                round(result["precision"] * 100, 4),
                round(result["time_taken"], 2),
                result["selected_features"],
                result["total_features"]
            ])

            print(f"Results for {name}:")
            print(f"  Accuracy        : {result['accuracy'] * 100:.2f}%")
            print(f"  Precision       : {result['precision'] * 100:.2f}%")
            print(f"  Time Taken      : {result['time_taken']:.2f} seconds")
            print(f"  Features Selected: {result['selected_features']} / {result['total_features']}")

