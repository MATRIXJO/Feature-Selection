import numpy as np
import scipy.io
import time
import csv
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore")

# ------------------------------
# Load your dataset
# ------------------------------
def load_mat_dataset(path, x_key='X', y_key='Y'):
    data = scipy.io.loadmat(path)
    X = data[x_key]
    y = data[y_key].ravel()
    return X, y

# ------------------------------
# 1. PSTD Decomposition
# ------------------------------
def compute_vip(pls_model, X):
    t = pls_model.x_scores_
    w = pls_model.x_weights_
    q = pls_model.y_loadings_
    p, h = w.shape
    s = np.sum(np.square(t), axis=0) * np.sum(np.square(q), axis=0)
    total_s = np.sum(s)
    vip = np.zeros((p,))
    for i in range(p):
        weight = np.array([(w[i, j] / np.linalg.norm(w[:, j])) ** 2 for j in range(h)])
        vip[i] = np.sqrt(p * np.sum(s * weight) / total_s)
    return vip

def pstd_decomposition(X, y, c_max=10, q_max=0.5, s_max=10):
    # 1. Find optimal number of PLS components (knee method)
    scores = []
    for c in range(1, c_max + 1):
        pls = PLSRegression(n_components=c)
        pls.fit(X, y)
        scores.append(pls.score(X, y))
    optimal_c = np.argmax(scores) + 1

    # 2. Apply PLS
    pls = PLSRegression(n_components=optimal_c)
    pls.fit(X, y)
    W = pls.x_weights_
    P = pls.x_loadings_

    # 3. Compute VIP and filter
    vip = compute_vip(pls, X)
    quantile_range = np.arange(0.01, q_max + 0.01, 0.01)
    best_acc = 0
    best_q = 0.25
    for q in quantile_range:
        threshold = np.quantile(vip, q)
        keep = vip >= threshold
        if np.sum(keep) < 2: continue
        acc = cross_val_accuracy(X[:, keep], y)
        if acc > best_acc:
            best_acc = acc
            best_q = q
    threshold = np.quantile(vip, best_q)
    selected_idx = np.where(vip >= threshold)[0]
    X_reduced = X[:, selected_idx]

    # 4. Rebuild PLS on reduced set
    pls = PLSRegression(n_components=optimal_c)
    pls.fit(X_reduced, y)
    P = pls.x_loadings_

    # 5. Clustering using HAC (Ward + Euclidean)
    best_s = 2
    best_score = -1
    for s in range(2, min(s_max + 1, len(selected_idx))):
        clustering = AgglomerativeClustering(n_clusters=s, linkage='ward')
        labels = clustering.fit_predict(P)
        sil_score = silhouette_score(P, labels)
        if sil_score > best_score:
            best_score = sil_score
            best_s = s
            best_labels = labels
    clusters = [selected_idx[best_labels == i] for i in range(best_s)]
    return clusters, selected_idx

# ------------------------------
# 2. Binary Genetic Algorithm
# ------------------------------
def binary_genetic_algorithm(X, y, clusters, max_gen=100, stagnation_limit=10):
    num_clusters = len(clusters)
    pop_size = 30
    mutation_rate = 0.05
    best_solution = None
    best_fitness = 0
    no_improve = 0
    population = [[np.random.randint(0, 2, len(cluster)) for cluster in clusters] for _ in range(pop_size)]
    
    for gen in range(max_gen):
        fitnesses = []
        for ind in population:
            selected = []
            for i, bitstring in enumerate(ind):
                selected.extend(clusters[i][bitstring == 1])
            if len(selected) == 0:
                fitnesses.append(0)
                continue
            score = eval_fitness(X[:, selected], y)
            fitnesses.append(score)
            if score > best_fitness:
                best_fitness = score
                best_solution = selected
                no_improve = 0
        if no_improve >= stagnation_limit:
            break
        no_improve += 1
        # Selection + Crossover
        indices = np.argsort(fitnesses)[-pop_size:]
        new_pop = []
        for _ in range(pop_size):
            idx1, idx2 = np.random.choice(indices, 2, replace=False)
            p1, p2 = population[idx1], population[idx2]
            child = [np.copy(p1[i]) if np.random.rand() < 0.5 else np.copy(p2[i]) for i in range(num_clusters)]
            for i in range(num_clusters):
                for j in range(len(child[i])):
                    if np.random.rand() < mutation_rate:
                        child[i][j] = 1 - child[i][j]
            new_pop.append(child)
        population = new_pop
    return best_solution

# ------------------------------
# 3. Evaluation
# ------------------------------
def eval_fitness(X_subset, y):
    X_train, X_test, y_train, y_test = train_test_split(X_subset, y, stratify=y, test_size=0.3)
    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(X_train, y_train)
    return clf.score(X_test, y_test)

def cross_val_accuracy(X, y):
    skf = StratifiedKFold(n_splits=5)
    scores = []
    for train_idx, test_idx in skf.split(X, y):
        clf = KNeighborsClassifier(n_neighbors=1)
        clf.fit(X[train_idx], y[train_idx])
        scores.append(clf.score(X[test_idx], y[test_idx]))
    return np.mean(scores)

# ------------------------------
# 4. Full Pipeline
# ------------------------------
def run_pipeline(mat_path, x_key='X', y_key='Y',dataset=""):
    start = time.time()
    X, y = load_mat_dataset(mat_path, x_key, y_key)
    total_features = X.shape[1]
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    clusters, selected_idx = pstd_decomposition(X, y)
    best_features = binary_genetic_algorithm(X, y, clusters)
    
    if not best_features:
        print("No features selected!")
        return
    
    # Evaluation
    X_sel = X[:, best_features]
    X_train, X_test, y_train, y_test = train_test_split(X_sel, y, stratify=y, test_size=0.3)
    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro')
    duration = time.time() - start

    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Time taken (s):", duration)
    print("Selected features:", len(best_features))
    print("Total features:", total_features)

    csv_file="../results/PSTD_CCEA.csv"
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(["Dataset", "Accuracy", "Precision", "Time Taken (s)", "Features Selected", "Total Features"])
        writer.writerow([
            dataset,
            acc * 100,
            prec * 100,
            duration,
            len(best_features),
            total_features
            ])

# Example:
#run_pipeline('your_dataset.mat', x_key='X', y_key='Y')

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
        print("\n\n",name)
        run_pipeline(filepath, x_key='X', y_key='Y',dataset=name)
