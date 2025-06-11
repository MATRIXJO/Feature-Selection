import numpy as np
import time
import csv
import os
import warnings
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.exceptions import UndefinedMetricWarning

# Symmetric Uncertainty
from sklearn.metrics import mutual_info_score

def entropy(vec):
    _, counts = np.unique(vec, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs + 1e-10))

def symmetric_uncertainty(x, y):
    h_x = entropy(x)
    h_y = entropy(y)
    ig = mutual_info_score(x, y)
    return 2.0 * ig / (h_x + h_y + 1e-10)

def init_population(X, y, pop_size, sigma_min, sigma_max):
    n_features = X.shape[1]
    igr = mutual_info_classif(X, y)
    su = np.array([symmetric_uncertainty(X[:, i], y) for i in range(n_features)])
    igr_norm = (igr - igr.min()) / (igr.max() - igr.min() + 1e-10)
    su_norm = (su - su.min()) / (su.max() - su.min() + 1e-10)

    population = []
    for i in range(pop_size):
        probs = igr_norm if i < pop_size // 2 else su_norm
        mask = np.random.rand(n_features) < probs
        sigma = np.random.uniform(sigma_min, sigma_max)
        antibody = np.concatenate([mask.astype(int), [int(sigma * 10)]])
        population.append(antibody)
    return np.array(population)

def affinity(antibody, X, y):
    mask = antibody[:-1].astype(bool)
    sigma = antibody[-1] / 10.0
    if np.sum(mask) == 0:
        return 0
    X_sel = X[:, mask]
    clf = KNeighborsClassifier(n_neighbors=3)
    accs = []
    skf = StratifiedKFold(n_splits=5)
    for train_idx, test_idx in skf.split(X_sel, y):
        clf.fit(X_sel[train_idx], y[train_idx])
        preds = clf.predict(X_sel[test_idx])
        accs.append(accuracy_score(y[test_idx], preds))
    return np.mean(accs)

def su_local_search(antibody, X, y, alpha=0.2, gmax=5):
    D = X.shape[1]
    mask = antibody[:-1].copy()
    for _ in range(gmax):
        indices = np.where(mask == 1)[0]
        n_flip = int(len(indices) * alpha)
        if n_flip < 1:
            continue
        selected = np.random.choice(indices, n_flip, replace=False)
        su_values = [symmetric_uncertainty(X[:, i], y) for i in selected]
        sorted_indices = [i for _, i in sorted(zip(su_values, selected), reverse=True)]

        for i in sorted_indices:
            for j in selected:
                if i != j and symmetric_uncertainty(X[:, i], X[:, j]) > symmetric_uncertainty(X[:, j], y):
                    mask[j] = 0
        non_selected = np.where(mask == 0)[0]
        candidates = np.random.choice(non_selected, n_flip, replace=False)
        avg_su = np.mean([symmetric_uncertainty(X[:, i], y) for i in selected])
        for i in candidates:
            if symmetric_uncertainty(X[:, i], y) > avg_su:
                mask[i] = 1
    new_antibody = np.concatenate([mask, [antibody[-1]]])
    return new_antibody if affinity(new_antibody, X, y) > affinity(antibody, X, y) else antibody

def run_ia_flfs(X, y, pop_size=10, max_iter=20):
    start_time = time.time()
    sigma_min, sigma_max = 0.1, 1.0
    population = init_population(X, y, pop_size, sigma_min, sigma_max)
    affinities = np.array([affinity(ind, X, y) for ind in population])
    best = population[np.argmax(affinities)]
    best_aff = max(affinities)

    for _ in range(max_iter):
        clones = []
        for i in range(pop_size):
            clone = population[i].copy()
            flip_idx = np.random.choice(len(clone) - 1, size=np.random.randint(1, 5), replace=False)
            clone[flip_idx] = 1 - clone[flip_idx]
            clone[-1] = np.clip(clone[-1] + np.random.normal(0, 1), 1, 10)
            clone = su_local_search(clone, X, y)
            clones.append(clone)

        for i in range(pop_size):
            if affinity(clones[i], X, y) > affinities[i]:
                population[i] = clones[i]
                affinities[i] = affinity(clones[i], X, y)

        new_best = population[np.argmax(affinities)]
        new_best_aff = max(affinities)
        if new_best_aff > best_aff:
            best = new_best
            best_aff = new_best_aff
        print(_)

    sel_features = int(np.sum(best[:-1]))
    total_features = X.shape[1]
    time_taken = time.time() - start_time

    final_X = X[:, best[:-1].astype(bool)]
    clf = KNeighborsClassifier(n_neighbors=3)
    accs, precs = [], []
    skf = StratifiedKFold(n_splits=5)
    for train_idx, test_idx in skf.split(final_X, y):
        clf.fit(final_X[train_idx], y[train_idx])
        preds = clf.predict(final_X[test_idx])
        accs.append(accuracy_score(y[test_idx], preds))
        precs.append(precision_score(y[test_idx], preds, average='macro', zero_division=0))

    return {
        "accuracy": np.mean(accs),
        "precision": np.mean(precs),
        "time_taken": time_taken,
        "selected_features": sel_features,
        "total_features": total_features
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

    csv_file = "../results/ia_flfs_results.csv"
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
                print("check")
                result = run_ia_flfs(X, y, pop_size=10, max_iter=20)
                for warning in w:
                    if "The least populated class in y has only" in str(warning.message):
                        print(f"⚠️  Warning in {name}: class size too small for StratifiedKFold")
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

