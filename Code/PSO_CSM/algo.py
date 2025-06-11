import numpy as np
import scipy.io
import time
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore")

# === Symmetric Uncertainty ===
def entropy(v):
    _, counts = np.unique(v, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs + 1e-10))


def symmetric_uncertainty(X, y):
    X_disc = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform').fit_transform(X)
    mi = mutual_info_classif(X_disc, y, discrete_features=True)
    H_y = entropy(y)
    su = []
    for i in range(X.shape[1]):
        H_x = entropy(X_disc[:, i])
        su.append(2 * mi[i] / (H_x + H_y) if (H_x + H_y) > 0 else 0)
    return np.array(su)


# === PSO-CSM ===
class PSO_CSM:
    def __init__(self, X, y, n_particles=30, max_iter=100, beta=0.9):
        self.X = X
        self.y = y
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.beta = beta
        self.dim = X.shape[1]
        self.population = np.zeros((n_particles, self.dim))
        self.velocity = np.zeros_like(self.population)
        self.su = symmetric_uncertainty(X, y)
        self.pbest = None
        self.gbest = None
        self.gbest_score = np.inf
        self.history = []

    def initialize(self):
        sorted_idx = np.argsort(-self.su)
        for i in range(self.n_particles):
            L = max(int((self.dim // self.n_particles) * (i + 1)), int(0.1 * self.dim))
            for d in range(self.dim):
                if d < int(0.1 * self.dim):
                    self.population[i, sorted_idx[d]] = 0.6 + 0.4 * np.random.rand()
                elif d < L:
                    self.population[i, sorted_idx[d]] = np.random.rand()
                else:
                    self.population[i, sorted_idx[d]] = 0

    def fitness(self, particle):
        mask = particle > 0.6
        if np.sum(mask) == 0:
            return 1.0
        X_sel = self.X[:, mask]
        acc = cross_val_score(KNeighborsClassifier(n_neighbors=1), X_sel, self.y,
                              cv=StratifiedKFold(n_splits=5)).mean()
        feat_ratio = np.sum(mask) / self.dim
        return self.beta * (1 - acc) + (1 - self.beta) * feat_ratio

    def comprehensive_score(self, population, lambda_=0.5):
        occurrence = np.sum(population > 0.6, axis=0)
        ranks = np.argsort(np.argsort(-self.su)) + 1
        scores = lambda_ * (occurrence / self.n_particles) + (1 - lambda_) * (1 - ranks / self.dim)
        return scores

    def _alpha_scaling(self):
        D = self.dim
        if D <= 1000:
            return 0.5 + (1.0 - 0.5) * np.random.rand()
        elif D <= 5000:
            return 0.1 + (0.5 - 0.1) * np.random.rand()
        else:
            return 0.05 + (0.1 - 0.05) * np.random.rand()

    def run(self):
        self.initialize()
        self.pbest = self.population.copy()
        pbest_scores = np.array([self.fitness(p) for p in self.pbest])
        self.gbest = self.pbest[np.argmin(pbest_scores)].copy()
        self.gbest_score = np.min(pbest_scores)

        for t in range(self.max_iter):
            w = 0.9 - 0.5 * (t / self.max_iter)
            c1 = c2 = 1.49445
            r1 = np.random.rand(*self.population.shape)
            r2 = np.random.rand(*self.population.shape)

            self.velocity = w * self.velocity + \
                            c1 * r1 * (self.pbest - self.population) + \
                            c2 * r2 * (self.gbest - self.population)
            self.population += self.velocity

            fitness_scores = []
            for i, particle in enumerate(self.population):
                score = self.fitness(particle)
                fitness_scores.append(score)
                if score < pbest_scores[i]:
                    self.pbest[i] = particle.copy()
                    pbest_scores[i] = score
                    if score < self.gbest_score:
                        self.gbest = particle.copy()
                        self.gbest_score = score

            if t > 0 and t % (self.max_iter // 5) == 0:
                scores = self.comprehensive_score(self.population)
                alpha = self._alpha_scaling()
                top_k = int(alpha * self.dim)
                top_features = np.argsort(-scores)[:top_k]
                gbest_selected = np.where(self.gbest > 0.6)[0]
                final_feats = np.unique(np.concatenate((top_features, gbest_selected)))

                for i in range(self.n_particles):
                    mask = np.zeros(self.dim)
                    mask[final_feats] = 1
                    self.population[i] *= mask

            self.history.append(self.gbest_score)

        return self.gbest > 0.6, self.gbest_score


# === Load and Run on .mat Dataset ===
def run_on_mat(mat_path, X_var, y_var,algo):
    print(f"Loading dataset from {mat_path} ...")
    data = scipy.io.loadmat(mat_path)
    X = data[X_var]
    y = data[y_var].ravel()

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

    start = time.time()
    pso = PSO_CSM(X_train, y_train, n_particles=30, max_iter=50)
    selected_mask, _ = pso.run()
    elapsed = time.time() - start

    X_train_sel = X_train[:, selected_mask]
    X_test_sel = X_test[:, selected_mask]

    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(X_train_sel, y_train)
    y_pred = model.predict(X_test_sel)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')

    print("\n=== Results ===")
    print("Accuracy        :", round(acc * 100, 2), "%")
    print("Precision       :", round(prec * 100, 2), "%")
    print("Time taken      :", round(elapsed, 2), "seconds")
    print("Features selected:", np.sum(selected_mask))
    print("Total features   :", X.shape[1])
    
    csv_file = "../results/PSO_CSM_res.csv"
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
    
    # Write header only if file is empty
        if file.tell() == 0:
            writer.writerow(["Algorithm","Accuracy", "Precision", "Time Taken (s)", "Features Selected", "Total Features"])
    
        writer.writerow([
        algo,
        round(acc * 100),
        round(prec * 100),
        round(elapsed),
        int(np.sum(selected_mask)),
        X.shape[1]
       ])
    
# === MAIN ===
if __name__ == "__main__":
    print("\n\nExecuting Leukemia1")
    run_on_mat("../Dataset/Leukemia_1.mat", "X", "Y","Leukemia1")
    
    print("\n\nExecuting DLBCL")
    run_on_mat("../Dataset/DLBCL.mat", "X", "Y","DLBCL")
    
    print("\n\nExecuting Brain_Tumor_1")
    run_on_mat("../Dataset/Brain_Tumor_1.mat", "X", "Y","Brain_Tumor_1")
    
    print("\n\nExecuting Prostate_Tumor_1")
    run_on_mat("../Dataset/Prostate_Tumor_1.mat", "X", "Y","Prostate_Tumor_1")
    
    print("\n\nExecuting nci9")
    run_on_mat("../Dataset/nci9.mat", "X", "Y","nci9")
    
    print("\n\nExecuting Leukemia_3")
    run_on_mat("../Dataset/Leukemia_3.mat", "X", "Y","Leukemia_3")
    
    print("\n\nExecuting CLL_SUB_111")
    run_on_mat("../Dataset/CLL_SUB_111.mat", "X", "Y","CLL_SUB_111")
    
    print("\n\nExecuting Lung_Cancer")
    run_on_mat("../Dataset/Lung_Cancer.mat", "X", "Y","Lung_Cancer")
    
    print("\n\nExecuting SMK_CAN_187")
    run_on_mat("../Dataset/SMK_CAN_187.mat", "X", "Y","SMK_CAN_187")
    
    print("\n\nExecuting GLI_85")
    run_on_mat("../Dataset/GLI_85.mat", "X", "Y","GLI_85")

