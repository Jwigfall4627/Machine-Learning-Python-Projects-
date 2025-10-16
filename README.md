# Machine-Learning-Python-Notebooks-

A curated companion README for the DS4510 workspace. This repository contains lecture notebooks that introduce ML concepts and a Capstone folder with practical notebooks applying those concepts to real data (NBA analytics). Use this README as a quick guide to what each notebook covers and how the pieces fit together.

---

## What is Machine Learning (short & practical)

Machine learning (ML) is a set of methods that let computers learn patterns from data and make predictions or decisions without being explicitly programmed for the task. In practice, ML workflows usually include:

- Problem framing (classification, regression, clustering)
- Data collection and cleaning
- Feature engineering and transformation
- Model selection and training
- Evaluation and validation
- Deployment and monitoring

This repository contains lecture notebooks that walk through these steps, followed by capstone notebooks that apply them to an NBA dataset as concrete examples.

## Contents & How the files map to ML use-cases

Top-level lecture notebooks (concepts, examples, exercises):

- `Lec01_Introduction_to_Python_P1.ipynb`, `Lec02 - Introduction to Python - P2.ipynb`
  - Python basics used throughout the course: data types, control flow, functions — the foundation.
- `Lec03_Data_Transformation_P1.ipynb`, `Lec04_Data_Transformation_P2.ipynb`
  - Data cleaning and transformation: handling missing values, merging, reshaping — essential preprocessing steps.
- `Lec05_Data_visualization_P1.ipynb`, `Lec06_Data_visualization_P2.ipynb`
  - Exploratory Data Analysis (EDA): visualizing distributions, relationships, and spotting outliers.
- `Lec07_Statistical_Tests_P1.ipynb`, `Lec08_Statistical_Tests_P2.ipynb`
  - Statistical tests and inference: hypothesis testing and p-values to support data-driven decisions.
- `Lec09_Multiple_Linear_Regression.ipynb`
  - Multiple linear regression: modeling relationships for continuous outcomes and interpreting coefficients.
- `Lec10 - Dimensionality Reduction.ipynb`
  - PCA and techniques for reducing feature dimensionality while preserving signal.
- `Lec11_Regression_and_Feature_Selection.ipynb`
  - Feature selection strategies and regression refinements.
- `Lec12_kNN_Regression.ipynb`
  - k-Nearest Neighbors for regression problems; distance-based prediction.
- `Lec13_Logistic_Regression.ipynb`
  - Logistic regression for binary classification and probability estimation.
- `Lec14_Decision_Trees.ipynb`
  - Decision tree models, interpretability, and overfitting trade-offs.
- `Lec15_Support_Vector_Machine.ipynb`
  - SVMs for classification with kernels and margin concepts.
- `Lec16_Neural_Networks.ipynb`
  - Intro to neural networks: architectures, training basics, and simple examples.

Capstone — `Capstone Project 4510` (practical applications / case studies):

- `classification_models_nba_won_game.ipynb`
  - Classification models aimed at predicting whether an NBA team wins a game. Demonstrates feature engineering, model training (likely logistic regression, trees, or other classifiers), and evaluation (accuracy, ROC/AUC, confusion matrix).
- `LogisticRegression.ipynb`, `LogisticRegressionArchive.ipynb`
  - Focused examples on logistic regression: model building, interpretation of coefficients, and performance on classification tasks.
- `mlr-model.ipynb`
  - Multiple linear regression modeling for continuous target variables — useful for predicting box-score stats or team performance metrics.
- `NBA_Data_Regression_Analysis-Modified_Copy.ipynb`, `NBA_Data_Regression_Analysis_Archive.ipynb`, `NBA_Data_Regression_Analysis_Archive02.ipynb`, `Copy of NBA_Data_Regression_Analysis.ipynb`
  - Regression analyses over NBA datasets. These notebooks walk through data prep, modeling, diagnostics, and comparisons between model variants.
- `Final Presentation/client-pres-3-notebook (1).ipynb` and `client-pres-3-notebookArchive.ipynb`
  - Presentation-ready notebooks (slimmed outputs / slides) summarizing the capstone findings and visualizations for non-technical audiences.

Data

- `Data/` — a folder for datasets used by the notebooks (may be empty locally). Put CSVs or other raw data here and update notebook paths if needed.

## Suggested learning / run order

1. Start with `Lec01` & `Lec02` if you need Python basics.
2. Move to `Lec03`–`Lec06` for data cleaning & visualization fundamentals.
3. Study `Lec07`–`Lec11` to learn statistical foundations, regression, and feature selection.
4. Explore `Lec12`–`Lec16` for algorithm-specific notebooks (kNN, logistic, trees, SVM, neural nets).
5. Work through the Capstone folder notebooks to see applied examples and experiments on NBA data.

## Quick setup (Windows cmd)

Recommended Python environment: Python 3.8+ and Jupyter Notebook / Jupyter Lab.

Create and activate a virtual environment (Windows cmd):

```cmd
python -m venv .venv
.venv\Scripts\activate
```

Install common dependencies (suggested):

```cmd
pip install --upgrade pip
pip install jupyterlab jupyter pandas numpy scikit-learn matplotlib seaborn statsmodels notebook
```

Open Jupyter Lab / Notebook:

```cmd
jupyter lab
```

Notes:
- If a `requirements.txt` exists in the future, prefer `pip install -r requirements.txt`.
- Some notebooks may use heavier libraries (TensorFlow / PyTorch). Install them only if needed.

## How to use the notebooks

- Open a notebook in Jupyter and run cells in order. If a cell expects a dataset path, ensure your CSV files are in the `Data/` folder or update the path at the top of the notebook.
- For reproducibility, set a random seed where relevant (many notebooks already do this in examples).
- Use small slices of data while experimenting to speed iteration; switch to full data for final runs.

## Reproducibility checklist

- Confirm Python version and package versions.
- Verify data paths and presence of required CSVs in `Data/`.
- Re-run EDA cells before training if datasets were modified.

## Contributing

If you want to improve these notebooks:

1. Make a branch.
2. Add or update a notebook or dataset.
3. Add a `requirements.txt` if you add new packages.
4. Open a pull request with a short description of changes.
