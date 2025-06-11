
import numpy as np
import scipy.io
import time
import csv
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score
from sklearn.feature_selection import mutual_info_classif

try:
    from skrebate import ReliefF
    relief_available = True
except ImportError:
    relief_available = False

def load_mat_dataset(path):
    mat = scipy.io.loadmat(path)
    X = mat['X'] if 'X' in mat else mat[list(mat.keys())[-1]]
    y = mat['Y'].ravel() if 'Y' in mat else mat['y'].ravel()
    return X, y

def cubic_chaotic_map(size, rho=4):
    y = np.random.rand(*size)
    for _ in range(10):
        y = np.clip(y, 1e-5, 0.9999)
        y = rho * y * (1 - y**2)
    return y

def compute_feature_weights(X, y):
    if relief_available:
        relief = ReliefF(n_neighbors=10, n_features_to_select=X.shape[1])
        relief.fit(X, y)
        return relief.feature_importances_
    else:
        return mutual_info_classif(X, y)

def initialize_population(n_particles, n_features, weights, threshold):
    pop = np.zeros((n_particles, n_features))
    chaos = cubic_chaotic_map((n_particles, n_features))
    for i in range(n_particles):
        for j in range(n_features):
            prob = chaos[i, j]
            if weights[j] > threshold:
                pop[i, j] = 1 if prob > 0.3 else 0
            else:
                pop[i, j] = 1 if prob > 0.7 else 0
    return pop.astype(int)

def evaluate_fitness(pop, X, y, alpha=0.9):
    fitness = []
    for i in range(pop.shape[0]):
        selected = pop[i] == 1
        if np.sum(selected) == 0:
            fitness.append(1.0)
            continue
        X_selected = X[:, selected]
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3)
        clf = KNeighborsClassifier(n_neighbors=3)
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        err = 1 - acc
        fit = alpha * err + (1 - alpha) * (np.sum(selected) / X.shape[1])
        fitness.append(fit)
    return np.array(fitness)

def calculate_diversity(pop):
    N, D = pop.shape
    mean_vec = np.mean(pop, axis=0)
    diversity = np.mean(np.sqrt(np.sum((pop - mean_vec)**2, axis=1)))
    return diversity

def build_pyramid_structure(pop, fitness, layers):
    sorted_indices = np.argsort(fitness)
    pop_sorted = pop[sorted_indices]
    layered = []
    total = 0
    for n in layers:
        layered.append(pop_sorted[total:total+n])
        total += n
    return layered, sorted_indices

def adaptive_threshold(fitness, diversity_now, diversity_prev, a=30, b=10):
    index = int(a + np.floor(b * (diversity_now / (diversity_prev + 1e-9))))
    index = min(index, len(fitness)-1)
    return np.sort(fitness)[index]

def dynamic_flip(particle, weights, freq, A=0.1, B=0.1):
    flipped = particle.copy()
    norm_w = (weights - np.min(weights)) / (np.max(weights) - np.min(weights) + 1e-6)
    for i in range(len(particle)):
        P = A * (-4 * norm_w[i]**2 + 4 * norm_w[i]) + B * freq[i]
        if np.random.rand() < P:
            flipped[i] = 1 - flipped[i]
    return flipped

def run_appso(X, y, n_particles=30, n_gen=30):
    D = X.shape[1]
    weights = compute_feature_weights(X, y)
    threshold = np.mean(weights)
    pop = initialize_population(n_particles, D, weights, threshold)
    pbest = pop.copy()
    gbest = None
    pbest_fit = evaluate_fitness(pop, X, y)
    gbest_fit = np.min(pbest_fit)
    gbest = pop[np.argmin(pbest_fit)]
    diversity_prev = calculate_diversity(pop)
    layers = [5, 10, 15]

    for gen in range(n_gen):
        fitness = evaluate_fitness(pop, X, y)
        pop_layered, indices = build_pyramid_structure(pop, fitness, layers)
        freq = np.mean(pop, axis=0)
        diversity_now = calculate_diversity(pop)
        threshold_val = adaptive_threshold(fitness, diversity_now, diversity_prev)
        diversity_prev = diversity_now
        new_pop = pop.copy()

        idx = 0
        for layer in pop_layered:
            for i in range(0, len(layer)-1, 2):
                loser_idx = indices[idx + i + 1]
                winner_idx = indices[idx + i]
                winner_fit = fitness[winner_idx]

                if winner_fit >= threshold_val:
                    rand_elite = gbest
                    if np.random.rand() < 0.6:
                        new_pop[loser_idx] = rand_elite.copy()
                    else:
                        new_pop[loser_idx] = pop[loser_idx]
                else:
                    new_pop[loser_idx] = dynamic_flip(pop[winner_idx], weights, freq)
            idx += len(layer)

        pop = new_pop
        pbest_fit = evaluate_fitness(pop, X, y)
        for i in range(n_particles):
            if pbest_fit[i] < gbest_fit:
                gbest_fit = pbest_fit[i]
                gbest = pop[i]

    return gbest

def evaluate_solution(best_sol, X, y):
    selected = best_sol == 1
    if np.sum(selected) == 0:
        return 0, 0, 0, 0, X.shape[1]
    X_selected = X[:, selected]
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3)
    clf = KNeighborsClassifier(n_neighbors=3)
    start = time.time()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    end = time.time()
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro')
    return acc, prec, end - start, np.sum(selected), X.shape[1]

def run_appso_on_mat_file(file_path,dataset):
    X, y = load_mat_dataset(file_path)
    print(f"Running APPSO on: {file_path} | Shape: {X.shape}")
    start = time.time()
    best_sol = run_appso(X, y)
    acc, prec, clf_time, n_selected, n_total = evaluate_solution(best_sol, X, y)
    total_time = time.time() - start
    print(f"Accuracy          : {acc * 100:.2f}%")
    print(f"Precision         : {prec * 100:.2f}%")
    print(f"Classifier Time   : {clf_time:.4f}s")
    print(f"Selected Features : {n_selected}/{n_total}")
    print(f"Total Time        : {total_time:.2f}s")
    
    csv_file = "../results/appso_res.csv"
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
    
    # Write header only if file is empty
        if file.tell() == 0:
            writer.writerow(["Dataset","Accuracy", "Precision", "Time Taken (s)", "Features Selected", "Total Features"])
    
        writer.writerow([
        dataset,
        acc * 100,
	prec * 100,
        total_time,
        np.sum(n_selected),
        X.shape[1]
       ])
    
if __name__ == "__main__":
    print("\n\nExecuting Leukemia1")
    run_appso_on_mat_file("../Dataset/Leukemia_1.mat","Leukemia1")
    
    print("\n\nExecuting DLBCL")
    run_appso_on_mat_file("../Dataset/DLBCL.mat","DLBCL")
    
    print("\n\nExecuting Brain_Tumor_1")
    run_appso_on_mat_file("../Dataset/Brain_Tumor_1.mat","Brain_Tumor_1")
    
    print("\n\nExecuting Prostate_Tumor_1")
    run_appso_on_mat_file("../Dataset/Prostate_Tumor_1.mat","Prostate_Tumor_1")
    
    print("\n\nExecuting nci9")
    run_appso_on_mat_file("../Dataset/nci9.mat","nci9")
    
    print("\n\nExecuting Leukemia_3")
    run_appso_on_mat_file("../Dataset/Leukemia_3.mat","nci9")
   
    print("\n\nExecuting CLL_SUB_111")
    run_appso_on_mat_file("../Dataset/CLL_SUB_111.mat","CLL_SUB_111")
    
    print("\n\nExecuting Lung_Cancer")
    run_appso_on_mat_file("../Dataset/Lung_Cancer.mat","Lung_Cancer")
    
    print("\n\nExecuting SMK_CAN_187")
    run_appso_on_mat_file("../Dataset/SMK_CAN_187.mat","SMK_CAN_187")
    
    print("\n\nExecuting GLI_85")
    run_appso_on_mat_file("../Dataset/GLI_85.mat","GLI_85")
