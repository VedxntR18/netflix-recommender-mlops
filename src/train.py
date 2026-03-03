"""
train.py - Trains the Netflix recommendation model and logs everything to MLflow.

What this script does:
1. Loads the cleaned Netflix data
2. Creates a TF-IDF matrix (converts text to numbers)
3. Saves model artifacts (vectorizer, matrix, titles, genres)
4. Logs all parameters, metrics, and artifacts to MLflow
"""

import pandas as pd
import numpy as np
import yaml
import os
import json
import joblib
import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_params():
    """Read configuration parameters from params.yaml."""
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    return params


def train_model():
    """
    Main training function.
    
    Steps:
    1. Load cleaned data
    2. Configure and fit TF-IDF vectorizer
    3. Test with a sample recommendation
    4. Save model artifacts (including genre data for evaluation)
    5. Log everything to MLflow
    """
    params = load_params()
    train_params = params["train"]

    # Step 1: Load cleaned data
    cleaned_data_path = os.path.join("data", "netflix_cleaned.csv")

    if not os.path.exists(cleaned_data_path):
        print("ERROR: Cleaned data not found. Run preprocess.py first!")
        print("Command: python src/preprocess.py")
        return

    df = pd.read_csv(cleaned_data_path)
    print(f"Loaded {len(df)} cleaned entries")

# Step 2: Set up MLflow experiment
    # We use the client to check if the experiment is in a 'deleted' state
    client = mlflow.tracking.MlflowClient()
    exp_name = "netflix-recommendation-experiment"
    
    experiment = client.get_experiment_by_name(exp_name)
    
    # If the experiment exists but was deleted, restore it
    if experiment and experiment.lifecycle_stage == "deleted":
        print(f"Restoring deleted experiment: {exp_name}")
        client.restore_experiment(experiment.experiment_id)
    
    mlflow.set_experiment(exp_name)

    with mlflow.start_run(run_name="tfidf-content-based-v1") as run:
        # Using run.info.run_id here will turn the 'mlflow' import bright
        print(f"MLflow Run ID: {run.info.run_id}")

        # Step 3: Log parameters
        mlflow.log_param("model_type", "content_based_tfidf")
        mlflow.log_param("max_features", train_params["max_features"])
        mlflow.log_param("ngram_range_min", train_params["ngram_range_min"])
        mlflow.log_param("ngram_range_max", train_params["ngram_range_max"])
        mlflow.log_param("top_n", train_params["top_n_recommendations"])
        mlflow.log_param("dataset_size", len(df))

        # Step 4: Create and fit TF-IDF Vectorizer
        tfidf = TfidfVectorizer(
            max_features=train_params["max_features"],
            ngram_range=(
                train_params["ngram_range_min"],
                train_params["ngram_range_max"]
            ),
            stop_words="english"
        )

        tfidf_matrix = tfidf.fit_transform(df["tags"])

        print(f"TF-IDF Matrix shape: {tfidf_matrix.shape}")

        # Step 5: Compute basic training metrics
        test_title = params["api"]["default_title"]
        titles_list = df["title"].tolist()

        np.random.seed(42)
        sample_size = min(100, len(df))
        sample_indices = np.random.choice(len(df), sample_size, replace=False)

        avg_similarity_scores = []
        for idx in sample_indices:
            sim_scores = cosine_similarity(
                tfidf_matrix[idx:idx + 1], tfidf_matrix
            ).flatten()
            top_n = train_params["top_n_recommendations"]
            sorted_scores = np.sort(sim_scores)[::-1]
            top_scores = sorted_scores[1:top_n + 1]
            avg_similarity_scores.append(np.mean(top_scores))

        mean_avg_similarity = float(np.mean(avg_similarity_scores))
        vocabulary_size = len(tfidf.vocabulary_)
        matrix_density = float(
            (tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1])) * 100
        )

        print(f"Mean Average Similarity Score: {mean_avg_similarity:.4f}")
        print(f"Vocabulary Size: {vocabulary_size}")
        print(f"Matrix Density: {matrix_density:.2f}%")

        # Step 6: Log metrics to MLflow
        mlflow.log_metric("mean_avg_similarity", mean_avg_similarity)
        mlflow.log_metric("vocabulary_size", vocabulary_size)
        mlflow.log_metric("matrix_density_percent", matrix_density)
        mlflow.log_metric("num_shows", len(df))

        # Step 7: Save model artifacts
        os.makedirs("models", exist_ok=True)

        joblib.dump(tfidf, os.path.join("models", "tfidf_vectorizer.pkl"))
        joblib.dump(tfidf_matrix, os.path.join("models", "tfidf_matrix.pkl"))
        joblib.dump(titles_list, os.path.join("models", "movie_titles.pkl"))

        # Save genre data and cleaned DataFrame for the evaluation step
        genres_list = df["listed_in"].tolist()
        joblib.dump(genres_list, os.path.join("models", "movie_genres.pkl"))
        df.to_csv(os.path.join("models", "evaluation_data.csv"), index=False)

        print("Model artifacts saved to models/ folder")

        # Step 8: Log artifacts to MLflow
        mlflow.log_artifacts("models", artifact_path="model_artifacts")

        # Step 9: Register model in MLflow Model Registry
        mlflow.sklearn.log_model(
            sk_model=tfidf,
            artifact_path="tfidf_model",
            registered_model_name="netflix-recommender-tfidf"
        )
        print("Model registered in MLflow Model Registry")

        # Step 10: Save training metrics to JSON
        metrics = {
            "mean_avg_similarity": round(mean_avg_similarity, 4),
            "vocabulary_size": vocabulary_size,
            "matrix_density_percent": round(matrix_density, 2),
            "num_shows": len(df),
            "model_type": "content_based_tfidf"
        }

        with open("metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        mlflow.log_artifact("metrics.json")
        print("Metrics saved to metrics.json")

        # Demo: Show sample recommendations
        if test_title in titles_list:
            test_idx = titles_list.index(test_title)
            sim_scores = cosine_similarity(
                tfidf_matrix[test_idx:test_idx + 1], tfidf_matrix
            ).flatten()
            top_indices = sim_scores.argsort()[::-1][1:6]

            print(f"\n Sample Recommendations for '{test_title}':")
            for i, idx in enumerate(top_indices, 1):
                print(f"   {i}. {titles_list[idx]} "
                      f"(similarity: {sim_scores[idx]:.3f})")
        else:
            print(f"\nNote: '{test_title}' not found in dataset.")

    print("\n Training complete! Run 'mlflow ui' to see the dashboard.")


if __name__ == "__main__":
    train_model()