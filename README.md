# - Movie Recommendation System using Machine Learning

A content-based movie recommendation system that suggests similar movies/TV shows based on genres, descriptions, cast, and directors — built and analyzed using the Amazon Prime movies and credits datasets.

## - Project Overview

- **Project Type:** Unsupervised Learning (Content-Based Filtering)
- **Contribution:** Individual
- **Author:** Pavitra J Haromuchadi

With the rapid growth of streaming platforms, users often struggle to choose from thousands of titles. This project builds a recommendation engine that analyzes movie/show features — genres, descriptions, cast, and directors — to recommend similar content, without relying on user ratings or watch history.

## - Dataset

Two datasets are merged on a common `id` column:

| Dataset | Description |
|---|---|
| `titles.csv` | Title, type, description, release year, genres, runtime, IMDb/TMDb scores, production countries |
| `credits.csv` | Person ID, name, character, role (actor/director) |

## - Tech Stack

- **Language:** Python (Google Colab)
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, NLTK, SciPy, Joblib

## - Workflow

1. **Data Preprocessing** — handling missing values, duplicates, merging datasets
2. **Exploratory Data Analysis** — 15+ univariate, bivariate, and multivariate visualizations (content type distribution, genre trends, rating distributions, correlation heatmaps, top actors, etc.)
3. **Hypothesis Testing** — t-tests and Pearson correlation to validate relationships (e.g., Movie vs. Show ratings, pre/post-2015 ratings, IMDb votes vs. TMDb popularity)
4. **Feature Engineering & NLP Preprocessing** — text cleaning, tokenization, lemmatization, TF-IDF vectorization on descriptions
5. **Feature Scaling & Dimensionality Reduction** — StandardScaler, PCA
6. **Content-Based Recommendation Engine** — CountVectorizer + Cosine Similarity to recommend similar titles
7. **Predictive Modeling (IMDb Score Prediction)** — Linear Regression, Decision Tree Regressor, and Random Forest Regressor, with hyperparameter tuning via RandomizedSearchCV
8. **Model Deployment Prep** — best model (Random Forest Regressor) saved via `joblib`

## - Model Performance

| Model | MAE | MSE | RMSE | R² Score |
|---|---|---|---|---|
| Decision Tree Regressor | 0.0146 | 0.0363 | 0.1904 | 0.9753 |
| Random Forest Regressor | 0.0172 | 0.0243 | 0.1560 | **0.9835** |

**Random Forest Regressor** was selected as the final model based on the best evaluation metrics and its ability to capture non-linear relationships.

## - Key Insights

- Drama and Comedy dominate the genre distribution.
- IMDb ratings differ significantly between Movies and TV Shows (statistically confirmed).
- IMDb votes and TMDb popularity show a weak-to-moderate positive correlation.
- `imdb_votes`, `tmdb_score`, `tmdb_popularity`, `runtime`, and `release_year` are the most important predictors of IMDb score.

## - How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/pavitra-j22/Book-Recommendation-System.git
   cd Book-Recommendation-System
   ```
2. Open the notebook in Google Colab or Jupyter.
3. Upload `titles.csv` and `credits.csv` to the working directory.
4. Run all cells sequentially.

## - Repository Structure

```
├── notebook.ipynb          # Main analysis & modeling notebook
├── best_model.joblib        # Saved Random Forest model
├── titles.csv               # Movie/show metadata
├── credits.csv               # Cast and crew data
└── README.md
```

## - Future Work

- Deploy the recommendation engine as a web app (Streamlit/Flask)
- Incorporate collaborative filtering using user ratings
- Expand feature set with sentiment analysis on descriptions

## - License

This project is for educational purposes as part of a Machine Learning capstone project.
