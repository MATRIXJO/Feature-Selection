import numpy as np
import scipy.io
import time
import csv
import math
import warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import mutual_info_classif


def sigmoid(x):
    z = -10 * (x - 0.5)
    z = np.clip(z, -500, 500)  # Prevent overflow in exp
    return 1 / (1 + np.exp(z))


def reliefF_importance(X, y):
    return mutual_info_classif(X, y)



def binary_transfer(x):
    return 1 if sigmoid(x) > np.random.rand() else 0


def fitness_function(solution, X, y, w=0.99):
    selected_indices = np.where(solution == 1)[0]
    if len(selected_indices) == 0:
        return 1.0  # worst if nothing selected

    X_selected = X[:, selected_indices]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []

    for train_idx, test_idx in skf.split(X_selected, y):
        X_train, X_test = X_selected[train_idx], X_selected[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = KNeighborsClassifier(n_neighbors=5)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))

    acc = np.mean(accuracies)
    error_rate = 1 - acc
    return (1 - w) * (len(selected_indices) / X.shape[1]) + w * error_rate


def differential_evolution(x_alpha, beta, delta, omega1, omega2, F=0.5):
    return x_alpha + F * (beta - delta) + F * (omega1 - omega2)


def levy_flight(dim, beta=1.5):
    sigma_u = np.power((math.gamma(1 + beta) * np.sin(np.pi * beta / 2)) /
                       (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)), 1 / beta)
    u = np.random.normal(0, sigma_u, size=dim)
    v = np.random.normal(0, 1, size=dim)
    return u / np.power(np.abs(v), 1 / beta)


def MIGWO(X, y, dataset="dataset", num_agents=20, max_iter=50):
    dim = X.shape[1]
    positions = np.zeros((num_agents, dim))
    alpha_pos, beta_pos, delta_pos = np.zeros(dim), np.zeros(dim), np.zeros(dim)
    alpha_score = beta_score = delta_score = float("inf")

    importance = reliefF_importance(X, y)
    top_features = np.argsort(importance)[-int(0.3 * dim):]
    for i in range(num_agents):
        mask = np.zeros(dim)
        mask[np.random.choice(top_features, size=len(top_features)//2, replace=False)] = 1
        positions[i] = mask

    start_time = time.time()

    for t in range(max_iter):
        a = 2 - t * (2 / max_iter)

        for i in range(num_agents):
            fitness = fitness_function(positions[i], X, y)
            if fitness < alpha_score:
                delta_score, delta_pos = beta_score, beta_pos.copy()
                beta_score, beta_pos = alpha_score, alpha_pos.copy()
                alpha_score, alpha_pos = fitness, positions[i].copy()
            elif fitness < beta_score:
                delta_score, delta_pos = beta_score, beta_pos.copy()
                beta_score, beta_pos = fitness, positions[i].copy()
            elif fitness < delta_score:
                delta_score, delta_pos = fitness, positions[i].copy()

        fitness_vals = np.array([1/alpha_score, 1/beta_score, 1/delta_score])
        w_fitness = fitness_vals / np.sum(fitness_vals)
        improvements = np.array([
            np.abs(alpha_score - fitness_function(alpha_pos, X, y)),
            np.abs(beta_score - fitness_function(beta_pos, X, y)),
            np.abs(delta_score - fitness_function(delta_pos, X, y))
        ])
        w_advance = improvements / (np.sum(improvements) + 1e-10)
        weights = (w_fitness + w_advance) / np.sum(w_fitness + w_advance)

        for i in range(num_agents):
            for j in range(dim):
                r1, r2 = np.random.rand(), np.random.rand()
                A1, C1 = 2 * a * r1 - a, 2 * r2
                D_alpha = abs(C1 * alpha_pos[j] - positions[i][j])
                X1 = alpha_pos[j] - A1 * D_alpha

                r1, r2 = np.random.rand(), np.random.rand()
                A2, C2 = 2 * a * r1 - a, 2 * r2
                D_beta = abs(C2 * beta_pos[j] - positions[i][j])
                X2 = beta_pos[j] - A2 * D_beta

                r1, r2 = np.random.rand(), np.random.rand()
                A3, C3 = 2 * a * r1 - a, 2 * r2
                D_delta = abs(C3 * delta_pos[j] - positions[i][j])
                X3 = delta_pos[j] - A3 * D_delta

                X_avg = weights[0] * X1 + weights[1] * X2 + weights[2] * X3
                positions[i][j] = binary_transfer(X_avg)

        for leader in [alpha_pos, beta_pos, delta_pos]:
            idxs = np.random.choice(np.arange(num_agents), 2, replace=False)
            v = differential_evolution(alpha_pos, beta_pos, delta_pos, positions[idxs[0]], positions[idxs[1]])
            u = np.array([v[j] if np.random.rand() < 0.5 else alpha_pos[j] for j in range(dim)])
            step = levy_flight(dim)
            for j in range(dim):
                if np.random.rand() < 0.8:
                    u[j] = u[j] + step[j]
                leader[j] = binary_transfer(u[j])

    exec_time = time.time() - start_time
    selected_features = np.where(alpha_pos == 1)[0]
    X_selected = X[:, selected_features]

    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X_selected, y)
    y_pred = clf.predict(X_selected)

    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='macro')
    csv_file = "../results/migwo_res.csv"
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(["Dataset", "Accuracy", "Precision", "Time Taken (s)", "Features Selected", "Total Features"])
        writer.writerow([
            dataset,
            accuracy * 100,
            precision * 100,
            exec_time,
            len(selected_features),
            X.shape[1]
        ])

    return {
        "accuracy": accuracy,
        "precision": precision,
        "time_taken": exec_time,
        "selected_features": len(selected_features),
        "total_features": dim
    }


def run_on_mat(filepath, X_var="X", y_var="Y", dataset="dataset"):
    data = scipy.io.loadmat(filepath)
    X = data[X_var]
    y = data[y_var].ravel()
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    warning_issued = False
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", category=UserWarning)
        result = MIGWO(X, y, dataset)

        for warning in w:
            if "The least populated class in y has only" in str(warning.message):
                print(f"⚠️  Warning in {dataset}: class size too small for StratifiedKFold (n_splits=10)")
                warning_issued = True
                break

    return result


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

    for filepath, name in datasets:
        print(f"\n\nExecuting {name}")
        result = run_on_mat(filepath, X_var="X", y_var="Y", dataset=name)
        
        print(f"Results for {name}:")
        print(f"  Accuracy        : {result['accuracy']*100:.2f}%")
        print(f"  Precision       : {result['precision']*100:.2f}%")
        print(f"  Time Taken      : {result['time_taken']:.2f} seconds")
        print(f"  Features Selected: {result['selected_features']} / {result['total_features']}")


