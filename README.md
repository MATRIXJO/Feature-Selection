# 🔍 Comparative Study of Optimization Algorithms for High-Dimensional Feature Selection

This repository presents a **comparative study of nature-inspired and evolutionary optimization algorithms** for **feature selection** in high-dimensional datasets. The algorithms analyzed are based on cutting-edge research published between 2024 and 2025. Feature selection plays a crucial role in reducing dimensionality, improving model accuracy, and minimizing computational costs, especially for domains like bioinformatics, intrusion detection, and image processing.

## 📚 Included Algorithms

This study examines and compares the following feature selection algorithms:

| Algorithm | Paper Reference | Key Innovations |
|----------|------------------|-----------------|
| **HMOFA** (Hierarchical Learning Multi-Objective Firefly Algorithm) | Zhao et al., 2024, Applied Soft Computing | Hierarchical learning, clustering-based initialization, duplicate solution modification |
| **EMSWOA** (Dynamic Multi-Swarm Whale Optimization with Elite Tuning) | Miao et al., 2024, Applied Soft Computing | Centroid-based dynamic swarms, elite tuning to reduce invalid flips |
| **APPSO** (Adaptive Pyramid Particle Swarm Optimization) | Jin et al., 2024, Expert Systems with Applications | Pyramid-based swarm structure, dynamic flip strategy |
| **MIGWO** (Multi-Strategy Improved Grey Wolf Optimizer) | Huang et al., 2025, Information Sciences | ReliefF-based initialization, DE and Lévy flight strategies |
| **FOX-GWO** (S-Shaped Grey Wolf FOX Optimizer) | Feda et al., 2024, Heliyon | FOX algorithm improved with S-shaped transfer and GWO integration |
| **IA-FLFS** (Immune Algorithm-Based Filter Local Feature Selection) | Wang et al., 2025, Applied Soft Computing | Local feature subset generation using adaptive immune strategy |
| **ST-CCEA** (Self-Tuning Cooperative Co-Evolutionary Algorithm) | Venâncio et al., 2025, Knowledge-Based Systems | Self-tuning decomposition of subproblems using domain knowledge |
| **PSO-CSM** (PSO with Comprehensive Scoring Mechanism) | Wei et al., 2025, Swarm & Evol. Comp. | Piecewise initialization, scoring-based dimension control |

## 🧪 Experimentation

- **Datasets:** The algorithms were evaluated on multiple high-dimensional datasets
  ![Screenshot from 2025-05-06 20-14-12](https://github.com/user-attachments/assets/f6de2088-598c-4a25-97fb-5ced5f9df7e8)
- **Metrics:**
  - Classification Accuracy
  - Number of Selected Features
  - Reduction Ratio
  - Computational Time
- **Classifier Used:** Typically, SVM or KNN was used in the wrapper-based evaluations.
- **Cross-validation:** 5-fold CV for robustness.

## 🏆 Key Takeaways

- **HMOFA** and **EMSWOA** showed superior balance between accuracy and feature reduction.
- **MIGWO** achieved high accuracy with compact subsets using hybridized exploration strategies.
- **FOX-GWO** displayed strong exploration in binary search space with fewer selected features.
- **IA-FLFS** was particularly effective for microarray datasets where local feature relevance is crucial.

## 📁 Repository Structure

```
.
├── Datasets/                 # .mat or .svm formatted high-dimensional datasets
├── Code/               # Implementation of all optimization algorithms
│   ├── hmofa/
│   ├── emswoa/
│   ├── appsso/
│   ├── migwo/
│   ├── foxgwo/
│   ├── ia_flfs/
│   ├── stccea/
│   └── psocsm/
└── README.md
```

## 🛠️ Tools & Libraries

- `Python 3.8+`
- `numpy`, `scipy`, `sklearn`, `matplotlib`, `pandas`
- `MATLAB` (for .mat dataset preprocessing if needed)
- `joblib` or `multiprocessing` for parallelism

## 🔬 Future Work

- Extend study to include deep learning-based feature selection.
- Benchmark on real-time streaming datasets.
- Integrate results with autoML pipelines.

