# Syndrome Classification from Image Embeddings

This project uses machine learning to classify syndromes based on 320-dimensional image embeddings from a dataset (`mini_gm_public_v0.1.p`). It processes the data, explores it, visualizes it with t-SNE, runs KNN classification with Cosine and Euclidean distances, evaluates performance, and writes a detailed report—all in one Python script, `X.py`.

## What’s Inside
- **Data Processing**: Loads and flattens the embeddings into lists, checks for issues, and saves them.
- **Exploratory Data Analysis (EDA)**: Counts images, syndromes, subjects, and spots imbalances.
- **Data Visualization**: Uses t-SNE to make a 2D plot colored by syndrome.
- **Classification**: Tests KNN with Cosine (k=11) and Euclidean (k=15) distances via 10-fold cross-validation.
- **Metrics**: Plots ROC curves and summarizes AUC, F1-Score, and Top-3 Accuracy.
- **Report**: Pulls it all together in `report.md`.

## Files
- **`X.py`**: The main script that does everything.
- **`requirements.txt`**: Lists the Python packages you need.
- **`mini_gm_public_v0.1.p`**: The input pickle file with embeddings (not included here, assumed present).
- **`results/` Folder**: Where outputs are saved:
  - `flattened_data.pkl`: Processed embeddings and labels.
  - `big_eda_summary.txt`: EDA stats.
  - `tsne_2d_data.pkl`: 2D t-SNE coordinates.
  - `tsne_plot.png`: t-SNE plot.
  - `roc_curves.png`: ROC curves for Cosine and Euclidean.
  - `knn_metrics_summary.txt`: KNN performance table.

## How to Set It Up
1. **Get the Files**:
   - Put `X.py`, `requirements.txt`, and `mini_gm_public_v0.1.p` in a folder (e.g., `Project/`).
   - Make sure the `results/` folder can be created there.

2. **Install Python Stuff**:
   - You need Python 3 (e.g., 3.11). Check with:
   python3 --version
   - Install the packages:
   pip install -r requirements.txt
   (Needs `numpy`, `scikit-learn`, `matplotlib`—see `requirements.txt`.)
   
3. **Run It**:
- From the folder, type:
python3 X.py
- It’ll take a few minutes (t-SNE and KNN are slow) and fill the `results/` folder.

## What You’ll Get
- **Stats**: Total images (1116), 10 syndromes, 941 subjects, plus breakdowns (e.g., 210 images for syndrome 300000034).
- **Plot**: `tsne_plot.png` shows embeddings in 2D—tight clusters mean overlap.
- **Results**: Cosine KNN (79% accuracy) beats Euclidean (74.3%), with AUCs near 0.95-0.96.
- **Report**: `report.md` explains it all—methods, results, and tips.

## Notes
- Cosine worked better—direction matters more than distance here.
- Check `report.md` for the full story and next steps!

## Need Help?
If something breaks, check the error message and make sure `mini_gm_public_v0.1.p` is in the right spot. Ping me if you’re stuck!