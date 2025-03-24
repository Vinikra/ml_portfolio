# README

## Overview
This script processes, analyzes, and classifies image embeddings stored in a pickle file. It performs exploratory data analysis (EDA), visualizes embeddings using t-SNE, and classifies images using k-Nearest Neighbors (k-NN) with different distance metrics.

## Features
- **Data Processing:** Loads and flattens embeddings, removes invalid data, and ensures consistency.
- **Exploratory Data Analysis (EDA):** Provides insights into dataset distribution, syndromes, subjects, and embedding characteristics.
- **t-SNE Visualization:** Projects high-dimensional embeddings into 2D space for visualization.
- **Classification:** Uses k-NN with cosine and Euclidean distance metrics to classify embeddings and evaluate performance using AUC, F1-score, and Top-3 accuracy.

## Dependencies
Ensure you have the following Python packages installed:
```bash
pip install numpy scikit-learn matplotlib
```

## Usage
Run the script in a Python environment:
```bash
python X.py
```

## Outputs
The script generates the following results, stored in the `results/` folder:
- **Flattened Data:** `flattened_data.pkl`
- **EDA Summary:** `eda_summary.txt`
- **t-SNE Data:** `tsne_2d_data.pkl`
- **t-SNE Plot:** `tsne_plot.png`
- **t-SNE Observations:** `tsne_observations.txt`
- **ROC Curves Plot:** `roc_curves.png`
- **KNN Metrics Summary:** `knn_metrics_summary.txt`

## Notes
- Ensure the `mini_gm_public_v0.1.p` pickle file is in the same directory as the script.
- The script automatically creates the `results/` directory if it does not exist.
- The classification model uses 10-fold cross-validation for evaluation.

## Contact
For any questions or issues, please reach out to the repository maintainer.

