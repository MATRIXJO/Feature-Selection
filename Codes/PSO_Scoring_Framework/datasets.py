from pso_algo import load_mat_dataset, run_pso_csm_with_visuals
import csv
import os


def store(result, algo):
    csv_file = '../results/PSO_Score_Frame.csv'
    row = {"Algorithm": algo} | result
    headers = row.keys()
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        if os.stat(csv_file).st_size == 0:
            writer.writeheader()
        writer.writerow(row)


#Leukemia1
print("\n\nExecuting Leukemia1")
X, Y = load_mat_dataset("../Dataset/Leukemia_1.mat", feature_key='X', label_key='Y')
results = run_pso_csm_with_visuals(X, Y, n_particles=30, max_iter=50)
store(results,"Leukemia1");

#DLBCL
print("\n\nExecuting DLBCL")
X, Y = load_mat_dataset("../Dataset/DLBCL.mat", feature_key='X', label_key='Y')
results = run_pso_csm_with_visuals(X, Y, n_particles=30, max_iter=50)
store(results,"DLBCL");


#Brain1
print("\n\nExecuting Brain1")
X, Y = load_mat_dataset("../Dataset/Brain_Tumor_1.mat", feature_key='X', label_key='Y')
results = run_pso_csm_with_visuals(X, Y, n_particles=30, max_iter=50)
store(results,"Brain1");


#Prostate
print("\n\nExecuting Prostate")
X, Y = load_mat_dataset("../Dataset/Prostate_Tumor_1.mat", feature_key='X', label_key='Y')
results = run_pso_csm_with_visuals(X, Y, n_particles=30, max_iter=50)
store(results,"Prostate");

#Nic9
print("\n\nExecuting Nic9")
X, Y = load_mat_dataset("../Dataset/nci9.mat", feature_key='X', label_key='Y')
results = run_pso_csm_with_visuals(X, Y, n_particles=30, max_iter=50)
store(results,"Nic9");


#Leukemia3
print("\n\nExecuting Leukemia3")
X, Y = load_mat_dataset("../Dataset/Leukemia_3.mat", feature_key='X', label_key='Y')
results = run_pso_csm_with_visuals(X, Y, n_particles=30, max_iter=50)
store(results,"Leukemia3");


#CLL_SUB
print("\n\nExecuting CLL_SUB")
X, Y = load_mat_dataset("../Dataset/CLL_SUB_111.mat", feature_key='X', label_key='Y')
results = run_pso_csm_with_visuals(X, Y, n_particles=30, max_iter=50)
store(results,"CLL_SUB");


#Lung-Cancer
print("\n\nExecuting Lung-Cancer")
X, Y = load_mat_dataset("../Dataset/Lung_Cancer.mat", feature_key='X', label_key='Y')
results = run_pso_csm_with_visuals(X, Y, n_particles=30, max_iter=50)
store(results,"Lung-Cancer");


#SMK-CAN-187
print("\n\nExecuting SMK-CAN-187")
X, Y = load_mat_dataset("../Dataset/SMK_CAN_187.mat", feature_key='X', label_key='Y')
results = run_pso_csm_with_visuals(X, Y, n_particles=30, max_iter=50)
store(results,"SMK-CAN-187");


#GLI-85
print("\n\nExecuting GLI-85")
X, Y = load_mat_dataset("../Dataset/GLI_85.mat", feature_key='X', label_key='Y')
results = run_pso_csm_with_visuals(X, Y, n_particles=30, max_iter=50)
store(results,"GLI-85");
