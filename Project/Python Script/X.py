import pickle
import numpy as np
from collections import Counter
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, roc_curve
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import os

# Set up the output folder
script_dir = os.path.dirname(os.path.abspath(__file__))  
output_folder = os.path.join(script_dir, "results")      
os.makedirs(output_folder, exist_ok=True)

# --- Data Processing ---
print("Loading the pickle file...")
pickle_path = os.path.join(os.path.dirname(__file__), "mini_gm_public_v0.1.p")
with open(pickle_path, "rb") as file:
    raw_data = pickle.load(file)

print("Flattening the data...")
images = []
syndromes = []
subjects = []
image_ids = []

for syndrome in raw_data:
    for subject in raw_data[syndrome]:
        for image_id, embedding in raw_data[syndrome][subject].items():
            images.append(embedding)
            syndromes.append(syndrome)
            subjects.append(subject)
            image_ids.append(image_id)

print("Checking data integrity...")
if any(np.isnan(img).any() for img in images):
    print("Found missing values! Filtering them out...")
    valid_indices = [i for i, img in enumerate(images) if not np.isnan(img).any()]
    images = [images[i] for i in valid_indices]
    syndromes = [syndromes[i] for i in valid_indices]
    subjects = [subjects[i] for i in valid_indices]
    image_ids = [image_ids[i] for i in valid_indices]
else:
    print("No missing values found.")

if len(set(len(img) for img in images)) > 1:
    print("Inconsistent embedding sizes! Keeping only 320D ones...")
    valid_indices = [i for i, img in enumerate(images) if len(img) == 320]
    images = [images[i] for i in valid_indices]
    syndromes = [syndromes[i] for i in valid_indices]
    subjects = [subjects[i] for i in valid_indices]
    image_ids = [image_ids[i] for i in valid_indices]
else:
    print("All embeddings are 320D, good to go!")

flattened_data = {"images": images, "syndromes": syndromes, "subjects": subjects, "image_ids": image_ids}
with open(os.path.join(output_folder, "flattened_data.pkl"), "wb") as file:
    pickle.dump(flattened_data, file)
print(f"Flattened data saved to {output_folder}flattened_data.pkl")

# --- Exploratory Data Analysis ---
print("\nDigging deep into the data:")
total_images = len(images)
unique_syndromes = len(set(syndromes))
unique_subjects = len(set(subjects))
print(f"Total images: {total_images}")
print(f"Unique syndromes: {unique_syndromes}")
print(f"Unique subjects: {unique_subjects}")

images_per_syndrome = Counter(syndromes)
print("\nImages per syndrome:")
for syndrome, count in sorted(images_per_syndrome.items()):
    print(f"{syndrome}: {count}")

images_per_subject = Counter(subjects)
print("\nImages per subject breakdown:")
subject_counts = list(images_per_subject.values())
print(f"Average: {np.mean(subject_counts):.2f}")
print(f"Median: {np.median(subject_counts):.2f}")
print(f"Min: {min(subject_counts)}")
print(f"Max: {max(subject_counts)}")
print(f"Standard deviation: {np.std(subject_counts):.2f}")

subjects_per_syndrome = {}
for syndrome in set(syndromes):
    subjects_per_syndrome[syndrome] = len(set([subjects[i] for i, s in enumerate(syndromes) if s == syndrome]))
print("\nSubjects per syndrome:")
for syndrome, count in sorted(subjects_per_syndrome.items()):
    print(f"{syndrome}: {count}")

print("\nAverage images per subject within each syndrome:")
avg_images_per_subject_by_syndrome = {}
for syndrome in set(syndromes):
    syndrome_subjects = [subjects[i] for i, s in enumerate(syndromes) if s == syndrome]
    subject_counts_per_syndrome = Counter(syndrome_subjects)
    avg_images_per_subject_by_syndrome[syndrome] = np.mean(list(subject_counts_per_syndrome.values()))
for syndrome, avg in sorted(avg_images_per_subject_by_syndrome.items()):
    print(f"{syndrome}: {avg:.2f}")

print("\nChecking the embeddings:")
embedding_means = np.mean(images, axis=0)
embedding_vars = np.var(images, axis=0)
print(f"Average across all embeddings (first 5): {embedding_means[:5]}")
print(f"Variance across all embeddings (first 5): {embedding_vars[:5]}")
low_variance_dims = sum(embedding_vars < 0.01)
print(f"Dimensions with low variance (< 0.01): {low_variance_dims}")

imbalance_ratio = max(images_per_syndrome.values()) / min(images_per_syndrome.values())
print(f"\nImbalance ratio (max/min images per syndrome): {imbalance_ratio:.2f}")
if imbalance_ratio > 2:
    print("Significant imbalance detected!")
else:
    print("Fairly balanced distribution.")

with open(os.path.join(output_folder, "eda_summary.txt"), "w") as file:
    file.write(f"Total images: {total_images}\n")
    file.write(f"Unique syndromes: {unique_syndromes}\n")
    file.write(f"Unique subjects: {unique_subjects}\n")
    file.write("\nImages per syndrome:\n")
    for s, c in sorted(images_per_syndrome.items()):
        file.write(f"{s}: {c}\n")
    file.write("\nImages per subject stats:\n")
    file.write(f"Average: {np.mean(subject_counts):.2f}\n")
    file.write(f"Median: {np.median(subject_counts):.2f}\n")
    file.write(f"Min: {min(subject_counts)}\n")
    file.write(f"Max: {max(subject_counts)}\n")
    file.write(f"Std Dev: {np.std(subject_counts):.2f}\n")
    file.write("\nSubjects per syndrome:\n")
    for s, c in sorted(subjects_per_syndrome.items()):
        file.write(f"{s}: {c}\n")
    file.write("\nAverage images per subject by syndrome:\n")
    for s, avg in sorted(avg_images_per_subject_by_syndrome.items()):
        file.write(f"{s}: {avg:.2f}\n")
    file.write("\nEmbedding stats:\n")
    file.write(f"Mean (first 5): {embedding_means[:5]}\n")
    file.write(f"Variance (first 5): {embedding_vars[:5]}\n")
    file.write(f"Low variance dims: {low_variance_dims}\n")
    file.write(f"\nImbalance ratio: {imbalance_ratio:.2f}\n")
print(f"EDA summary saved to {output_folder}eda_summary.txt")

# --- Data Visualization ---
print("\nSquishing the 320D embeddings to 2D with t-SNE...")
tsne = TSNE(n_components=2, random_state=42)
images_2d = tsne.fit_transform(np.array(images))

tsne_data = {"x": images_2d[:, 0], "y": images_2d[:, 1], "syndromes": syndromes}
with open(os.path.join(output_folder, "tsne_2d_data.pkl"), "wb") as file:
    pickle.dump(tsne_data, file)
print(f"2D t-SNE data saved to {output_folder}tsne_2d_data.pkl")

print("Drawing a picture of the 2D embeddings...")
plt.figure(figsize=(12, 8))
unique_syndromes = sorted(set(syndromes))
colors = plt.cm.get_cmap("tab10", len(unique_syndromes))
for i, syndrome in enumerate(unique_syndromes):
    indices = [j for j, s in enumerate(syndromes) if s == syndrome]
    plt.scatter(images_2d[indices, 0], images_2d[indices, 1], 
                color=colors(i), label=syndrome, alpha=0.6)
plt.legend(title="Syndromes")
plt.title("t-SNE of Image Embeddings")
plt.xlabel("X")
plt.ylabel("Y")
plt.savefig(os.path.join(output_folder, "tsne_plot.png"))
plt.close()
print(f"Plot saved to {output_folder}tsne_plot.png")

observations = []
if len(unique_syndromes) <= 10:
    observations.append("The plot uses different colors for each syndrome. Look for clusters!")
distances = np.sqrt((images_2d[:, 0] - images_2d[:, 0].mean())**2 + 
                    (images_2d[:, 1] - images_2d[:, 1].mean())**2)
spread = np.std(distances)
print(f"Spread of points (std dev of distances): {spread:.2f}")
if spread < 10:
    observations.append("Points are pretty tight—might be hard to separate syndromes.")
else:
    observations.append("Points are spread out—could show distinct groups.")

with open(os.path.join(output_folder, "tsne_observations.txt"), "w") as file:
    file.write("t-SNE Patterns and Thoughts:\n")
    for obs in observations:
        file.write(f"- {obs}\n")
    file.write("\nHow this helps classification:\n")
    file.write("- If clusters match syndromes, classification should work well.\n")
    file.write("- Overlap means some syndromes might get mixed up.\n")
    file.write("- Tight spread could mean embeddings need tweaking for better separation.\n")
print(f"Observations saved to {output_folder}tsne_observations.txt")

# --- Classification and Metrics ---
X = np.array(images)
y = np.array(syndromes)
unique_classes = sorted(set(y))
kf = KFold(n_splits=10, shuffle=True, random_state=42)

distances = {"cosine": 11, "euclidean": 15}
metrics = {"cosine": {"fpr": [], "tpr": [], "auc": [], "f1": [], "top3": []}, 
           "euclidean": {"fpr": [], "tpr": [], "auc": [], "f1": [], "top3": []}}

for distance, best_k in distances.items():
    print(f"\nRunning KNN with {distance} distance (k={best_k})...")
    knn = KNeighborsClassifier(n_neighbors=best_k, metric=distance)
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        y_test_bin = label_binarize(y_test, classes=unique_classes)
        y_pred_proba = knn.predict_proba(X_test)
        auc = roc_auc_score(y_test_bin, y_pred_proba, multi_class="ovr", average="macro")
        metrics[distance]["auc"].append(auc)
        fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_pred_proba.ravel())
        metrics[distance]["fpr"].append(fpr)
        metrics[distance]["tpr"].append(tpr)
        f1 = f1_score(y_test, y_pred, average="macro")
        metrics[distance]["f1"].append(f1)
        y_pred_proba_sorted = np.argsort(y_pred_proba, axis=1)[:, -3:]
        top3_correct = sum(y_test[i] in knn.classes_[y_pred_proba_sorted[i]] for i in range(len(y_test)))
        top3_acc = top3_correct / len(y_test)
        metrics[distance]["top3"].append(top3_acc)

print("\nAveraging ROC curves across folds...")
plt.figure(figsize=(10, 8))
for distance in distances:
    mean_fpr = np.linspace(0, 1, 100)
    tprs = [np.interp(mean_fpr, fpr, tpr) for fpr, tpr in zip(metrics[distance]["fpr"], metrics[distance]["tpr"])]
    mean_tpr = np.mean(tprs, axis=0)
    plt.plot(mean_fpr, mean_tpr, label=f"{distance.capitalize()} (AUC = {np.mean(metrics[distance]['auc']):.3f})")
plt.plot([0, 1], [0, 1], "k--")
plt.title("Average ROC Curves for KNN")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.savefig(os.path.join(output_folder, "roc_curves.png"))
plt.close()
print(f"ROC curves plot saved to {output_folder}roc_curves.png")

summary_lines = ["KNN Performance Metrics Across 10 Folds:\n"]
for distance in distances:
    summary_lines.append(f"\n{distance.capitalize()} Distance (k={distances[distance]}):")
    summary_lines.append(f"Average AUC: {np.mean(metrics[distance]['auc']):.3f}")
    summary_lines.append(f"Average F1-Score: {np.mean(metrics[distance]['f1']):.3f}")
    summary_lines.append(f"Average Top-3 Accuracy: {np.mean(metrics[distance]['top3']):.3f}")

with open(os.path.join(output_folder, "knn_metrics_summary.txt"), "w") as file:
    file.writelines("\n".join(summary_lines))
print(f"Metrics summary saved to {output_folder}knn_metrics_summary.txt")