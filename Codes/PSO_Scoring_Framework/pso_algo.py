import numpy as np
import random
import time
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.metrics import precision_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import mutual_info_classif

# Load .mat dataset
def load_mat_dataset(filepath, feature_key='X', label_key='Y'):
    data = loadmat(filepath)
    X = data[feature_key]
    y = data[label_key].flatten()
    return X, y

# Symmetric Uncertainty
def symmetric_uncertainty(X, y):
    mi = mutual_info_classif(X, y)
    H_y = -np.sum([p * np.log2(p) for p in np.bincount(y) / len(y) if p > 0])
    su = []
    for i in range(X.shape[1]):
        xi = X[:, i]
        hist, bins = np.histogram(xi, bins='auto', density=True)
        probs = hist * np.diff(bins)
        H_x = -np.sum([p * np.log2(p) for p in probs if p > 0])
        su_i = 2 * mi[i] / (H_x + H_y) if (H_x + H_y) != 0 else 0
        su.append(su_i)
    return np.array(su)

# Piecewise Initialization
def piecewise_initialization(X, y, N):
    D = X.shape[1]
    SU = symmetric_uncertainty(X, y)
    ranked_features = np.argsort(-SU)
    M = int(0.1 * D)
    population = np.zeros((N, D))
    for i in range(N):
        L_Pi = max(M, round(D / N) * (i + 1))
        selected = ranked_features[:L_Pi]
        for j in selected:
            if np.where(ranked_features == j)[0][0] < M:
                population[i, j] = 0.4 * random.random() + 0.6
            else:
                population[i, j] = random.random()
    return population, ranked_features, SU

# Binarize
def binarize(position, threshold=0.6):
    return np.array(position > threshold, dtype=int)

# Fitness function
def fitness_function(X, y, selected_features, beta=0.9):
    if np.sum(selected_features) == 0:
        return 1.0
    clf = KNeighborsClassifier(n_neighbors=1)
    acc = cross_val_score(clf, X[:, selected_features == 1], y, cv=5).mean()
    error = 1 - acc
    return beta * error + (1 - beta) * (np.sum(selected_features) / len(selected_features))

# Comprehensive Scoring
def comprehensive_score(population_bin, SU, lambda_=0.5):
    freq = np.sum(population_bin, axis=0)
    rank = np.argsort(-SU)
    rank_dict = {f: i for i, f in enumerate(rank)}
    scores = lambda_ * (freq / population_bin.shape[0]) + (1 - lambda_) * np.array([1 - rank_dict[i] / len(SU) for i in range(len(SU))])
    return scores

# Scaling factor
def get_scaling_factor(D):
    r = random.random()
    if D <= 1000:
        return 0.5 + (1.0 - 0.5) * r
    elif 1000 < D <= 5000:
        return 0.1 + (0.5 - 0.1) * r
    else:
        return 0.05 + (0.1 - 0.05) * r

# PSO-CSM
def pso_csm(X, y, n_particles=30, max_iter=50, beta=0.9):
    population, ranked_features, SU = piecewise_initialization(X, y, n_particles)
    D = X.shape[1]
    velocity = np.zeros((n_particles, D))
    position = np.copy(population)
    personal_best = np.copy(position)
    personal_best_fitness = np.array([fitness_function(X, y, binarize(p), beta) for p in position])
    global_best_index = np.argmin(personal_best_fitness)
    global_best = personal_best[global_best_index]
    feature_space = np.arange(D)

    for t in range(max_iter):
        population_bin = np.array([binarize(p) for p in position])
        for i in range(n_particles):
            r1, r2 = np.random.rand(D), np.random.rand(D)
            velocity[i] = 0.9 * velocity[i] + 1.49445 * r1 * (personal_best[i] - position[i]) + 1.49445 * r2 * (global_best - position[i])
            position[i] += velocity[i]
            position[i] = np.clip(position[i], 0, 1)

            fitness = fitness_function(X, y, binarize(position[i]), beta)
            if fitness < personal_best_fitness[i]:
                personal_best[i] = position[i]
                personal_best_fitness[i] = fitness

        if t % (max_iter // 5) == 0 and t > 0:
            scores = comprehensive_score(population_bin, SU)
            alpha = get_scaling_factor(len(feature_space))
            top_k = int(alpha * D)
            top_features = np.argsort(-scores)[:top_k]
            gbest_selected = binarize(global_best).astype(bool)
            new_space = np.unique(np.concatenate([top_features, np.where(gbest_selected)[0]]))
            X = X[:, new_space]
            position = position[:, new_space]
            velocity = velocity[:, new_space]
            personal_best = personal_best[:, new_space]
            global_best = global_best[new_space]
            D = X.shape[1]
            SU = symmetric_uncertainty(X, y)
            feature_space = new_space

        global_best_index = np.argmin(personal_best_fitness)
        global_best = personal_best[global_best_index]

    selected = binarize(global_best)
    final_features = feature_space[selected == 1]
    final_fitness = fitness_function(X, y, selected, beta)
    return final_features, final_fitness

# Visualization + Evaluation
def run_pso_csm_with_visuals(X, y, n_particles=30, max_iter=50, beta=0.9):
    start_time = time.time()
    selected_features, final_fitness = pso_csm(X, y, n_particles=n_particles, max_iter=max_iter, beta=beta)
    time_taken = time.time() - start_time

    X_selected = X[:, selected_features]
    clf = KNeighborsClassifier(n_neighbors=1)
    acc = cross_val_score(clf, X_selected, y, cv=5).mean()

    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    prec = precision_score(y_test, y_pred, average='macro')

    su_all = symmetric_uncertainty(X, y)
    selected_mask = np.zeros(X.shape[1], dtype=bool)
    selected_mask[selected_features] = True

    plt.figure(figsize=(10, 5))
    plt.plot(su_all, label='All Features SU', alpha=0.5)
    plt.scatter(np.where(selected_mask)[0], su_all[selected_mask], color='red', label='Selected Features', zorder=5)
    plt.title("Symmetric Uncertainty of Features")
    plt.xlabel("Feature Index")
    plt.ylabel("SU Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("\n📝 PSO-CSM Feature Selection Summary")
    print(f"⏱️  Time taken: {time_taken:.2f} seconds")
    print(f"#️⃣  Features selected: {len(selected_features)}")
    print(f"📊  Total features in dataset: {X.shape[1]}")
    print(f"✅  Accuracy: {acc:.4f}")
    print(f"🎯  Precision: {prec:.4f}")

    return {
        "time_taken": time_taken,
        "num_selected": len(selected_features),
        "total_features": X.shape[1],
        "accuracy": acc,
        "precision": prec,
    }
