import pandas as pd
import numpy as np
import yaml
import os
import json
import joblib
import mlflow
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
matplotlib.use('Agg')

# ================================================================
# SECTION 1: HELPER FUNCTIONS
# ================================================================

def load_params():
    """Read configuration parameters from params.yaml."""
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    return params


def parse_genres(genre_string):
    """
    Convert "Dramas, Comedies, International Movies"
    into a set: {"dramas", "comedies", "international movies"}
    """
    if pd.isna(genre_string) or str(genre_string).strip() == "":
        return set()
    return set(g.strip().lower() for g in str(genre_string).split(","))


def is_relevant(genres_query, genres_candidate):
    """
    A recommendation is RELEVANT if it shares at least one genre
    with the query show.
    """
    if len(genres_query) == 0 or len(genres_candidate) == 0:
        return False
    return len(genres_query.intersection(genres_candidate)) > 0


# ================================================================
# SECTION 2: METRIC COMPUTATION FUNCTIONS
# ================================================================

def precision_at_k(relevant_flags, k):
    """
    Precision@K = (relevant items in top K) / K
    Example: flags = [1, 0, 1, 1, 0], K=5 -> 3/5 = 0.60
    """
    if k == 0:
        return 0.0
    top_k = relevant_flags[:k]
    return sum(top_k) / k


def recall_at_k(relevant_flags, k, total_relevant):
    """
    Recall@K = (relevant items in top K) / (total relevant in database)
    """
    if total_relevant == 0:
        return 0.0
    top_k = relevant_flags[:k]
    return sum(top_k) / total_relevant


def ndcg_at_k(relevant_flags, k):
    if k == 0:
        return 0.0

    top_k = relevant_flags[:k]

    # DCG: sum of relevance / log2(position + 1)
    dcg = 0.0
    for i, rel in enumerate(top_k):
        dcg += rel / np.log2(i + 2)

    # Ideal DCG: best possible ordering (all relevant items first)
    ideal_flags = sorted(top_k, reverse=True)
    idcg = 0.0
    for i, rel in enumerate(ideal_flags):
        idcg += rel / np.log2(i + 2)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def average_precision(relevant_flags):
    """
    AP: Average of precision values at each relevant position.
    
    Captures both precision and ranking quality in one number.
    """
    if not any(relevant_flags):
        return 0.0

    precisions = []
    num_relevant = 0

    for i, is_rel in enumerate(relevant_flags):
        if is_rel:
            num_relevant += 1
            precisions.append(num_relevant / (i + 1))

    return float(np.mean(precisions)) if precisions else 0.0


def hit_rate(relevant_flags, k):
    """
    Hit Rate@K: 1.0 if at least one relevant item in top K, else 0.0.
    """
    return 1.0 if sum(relevant_flags[:k]) > 0 else 0.0


def intra_list_diversity(pairwise_similarities):
    """
    Diversity = 1 - average(pairwise similarity among recommendations).
    Higher = more diverse recommendations.
    """
    if len(pairwise_similarities) == 0:
        return 0.0
    return 1.0 - float(np.mean(pairwise_similarities))


# ================================================================
# SECTION 3: BASELINE RECOMMENDERS (For Comparison)
# ================================================================

class RandomRecommender:
    """
    Baseline 1: Recommends random shows.
    Our model MUST beat this to be considered useful.
    """

    def __init__(self, num_items, seed=42):
        self.num_items = num_items
        self.rng = np.random.RandomState(seed)

    def recommend(self, query_idx, k):
        """Return K random indices excluding the query."""
        candidates = list(range(self.num_items))
        candidates.remove(query_idx)
        return self.rng.choice(
            candidates,
            size=min(k, len(candidates)),
            replace=False
        )


class PopularityRecommender:
    """
    Baseline 2: Recommends shows from the most common genres.
    Smarter than random but ignores the specific query.
    """

    def __init__(self, genres_list, seed=42):
        self.rng = np.random.RandomState(seed)

        # Count genre frequency
        all_genres = []
        for genre_str in genres_list:
            if pd.notna(genre_str) and str(genre_str).strip():
                all_genres.extend(
                    [g.strip().lower() for g in str(genre_str).split(",")]
                )

        self.genre_counts = Counter(all_genres)

        # Score each show by popularity of its genres
        self.popularity_scores = []
        for genre_str in genres_list:
            genres = parse_genres(genre_str)
            score = sum(self.genre_counts.get(g, 0) for g in genres)
            self.popularity_scores.append(score)

        self.popularity_scores = np.array(self.popularity_scores)

    def recommend(self, query_idx, k):
        """Return K shows with highest popularity scores."""
        scores = self.popularity_scores.copy()
        scores[query_idx] = -1
        return scores.argsort()[::-1][:k]


# ================================================================
# SECTION 4: MAIN EVALUATION FUNCTION
# ================================================================

def evaluate_model():
    """
    Main evaluation function.
    Runs the model on test queries, measures performance,
    compares against baselines, generates charts and reports.
    """
    print("=" * 70)
    print("  NETFLIX RECOMMENDATION MODEL - EVALUATION PIPELINE")
    print("=" * 70)

    # Load parameters
    params = load_params()
    eval_params = params["evaluate"]

    k_values = eval_params["k_values"]
    num_test_samples = eval_params["num_test_samples"]
    seed = eval_params["random_seed"]

    # ── Load model artifacts ──
    print("\n Loading model artifacts...")

    required_files = [
        "models/tfidf_matrix.pkl",
        "models/movie_titles.pkl",
        "models/movie_genres.pkl"
    ]

    for f in required_files:
        if not os.path.exists(f):
            print(f"ERROR: {f} not found. Run training first: python src/train.py")
            return

    tfidf_matrix = joblib.load("models/tfidf_matrix.pkl")
    movie_titles = joblib.load("models/movie_titles.pkl")
    movie_genres = joblib.load("models/movie_genres.pkl")

    num_items = len(movie_titles)
    print(f"  Loaded {num_items} shows/movies")
    print(f"  TF-IDF Matrix shape: {tfidf_matrix.shape}")

    # ── Parse genres ──
    print("\n  Parsing genre information...")
    all_genre_sets = [parse_genres(g) for g in movie_genres]

    shows_with_genres = sum(1 for g in all_genre_sets if len(g) > 0)
    print(f"  Shows with genre labels: {shows_with_genres}/{num_items} "
          f"({shows_with_genres / num_items * 100:.1f}%)")

    # ── Select test samples ──
    np.random.seed(seed)
    valid_indices = [i for i in range(num_items) if len(all_genre_sets[i]) > 0]
    test_indices = np.random.choice(
        valid_indices,
        size=min(num_test_samples, len(valid_indices)),
        replace=False
    )
    print(f"  Selected {len(test_indices)} test queries")

    # ── Initialize baselines ──
    random_baseline = RandomRecommender(num_items, seed=seed)
    popularity_baseline = PopularityRecommender(movie_genres, seed=seed)

    # ── Set up MLflow ──
    mlflow.set_experiment("netflix-recommendation-evaluation")

    with mlflow.start_run(run_name="evaluation-v1") as run:

        print(f"\n  MLflow Run ID: {run.info.run_id}")

        # Log evaluation parameters
        mlflow.log_param("num_test_samples", len(test_indices))
        mlflow.log_param("k_values", str(k_values))
        mlflow.log_param("model_type", "content_based_tfidf")
        mlflow.log_param("num_total_items", num_items)
        mlflow.log_param("min_precision_threshold", eval_params["min_precision_at_5"])
        mlflow.log_param("min_hit_rate_threshold", eval_params["min_hit_rate"])

        # ════════════════════════════════════════════════
        # EVALUATE: TF-IDF Model
        # ════════════════════════════════════════════════

        print("\n" + "-" * 70)
        print("  Evaluating TF-IDF Content-Based Model...")
        print("-" * 70)

        max_k = max(k_values)

        model_metrics = {
            k: {"precisions": [], "recalls": [], "ndcgs": [], "hits": []}
            for k in k_values
        }
        model_aps = []
        model_diversities = []

        for i, query_idx in enumerate(test_indices):
            query_genres = all_genre_sets[query_idx]

            # Total relevant items in database (for recall)
            total_relevant = sum(
                1 for j in range(num_items)
                if j != query_idx and is_relevant(query_genres, all_genre_sets[j])
            )

            # Get model recommendations
            sim_scores = cosine_similarity(
                tfidf_matrix[query_idx:query_idx + 1],
                tfidf_matrix
            ).flatten()

            sorted_indices = sim_scores.argsort()[::-1]
            rec_indices = [idx for idx in sorted_indices if idx != query_idx][:max_k]

            # Determine relevance of each recommendation
            relevant_flags = [
                1 if is_relevant(query_genres, all_genre_sets[rec_idx]) else 0
                for rec_idx in rec_indices
            ]

            # Compute metrics at each K
            for k in k_values:
                model_metrics[k]["precisions"].append(
                    precision_at_k(relevant_flags, k)
                )
                model_metrics[k]["recalls"].append(
                    recall_at_k(relevant_flags, k, total_relevant)
                )
                model_metrics[k]["ndcgs"].append(
                    ndcg_at_k(relevant_flags, k)
                )
                model_metrics[k]["hits"].append(
                    hit_rate(relevant_flags, k)
                )

            # Average Precision
            model_aps.append(average_precision(relevant_flags[:max_k]))

            # Diversity among top-5
            top5_indices = rec_indices[:5]
            if len(top5_indices) >= 2:
                rec_vectors = tfidf_matrix[top5_indices]
                pairwise_sims = cosine_similarity(rec_vectors)
                upper_tri = pairwise_sims[
                    np.triu_indices_from(pairwise_sims, k=1)
                ]
                model_diversities.append(intra_list_diversity(upper_tri))

            # Progress
            if (i + 1) % 50 == 0:
                print(f"  Evaluated {i + 1}/{len(test_indices)} queries...")

        # ════════════════════════════════════════════════
        # EVALUATE: Random Baseline
        # ════════════════════════════════════════════════

        print("\n  Evaluating Random Baseline...")

        random_metrics = {
            k: {"precisions": [], "hits": []} for k in k_values
        }

        for query_idx in test_indices:
            query_genres = all_genre_sets[query_idx]
            rec_indices = random_baseline.recommend(query_idx, max_k)

            relevant_flags = [
                1 if is_relevant(query_genres, all_genre_sets[rec_idx]) else 0
                for rec_idx in rec_indices
            ]

            for k in k_values:
                random_metrics[k]["precisions"].append(
                    precision_at_k(relevant_flags, k)
                )
                random_metrics[k]["hits"].append(
                    hit_rate(relevant_flags, k)
                )

        # ════════════════════════════════════════════════
        # EVALUATE: Popularity Baseline
        # ════════════════════════════════════════════════

        print("  Evaluating Popularity Baseline...")

        popularity_metrics = {
            k: {"precisions": [], "hits": []} for k in k_values
        }

        for query_idx in test_indices:
            query_genres = all_genre_sets[query_idx]
            rec_indices = popularity_baseline.recommend(query_idx, max_k)

            relevant_flags = [
                1 if is_relevant(query_genres, all_genre_sets[rec_idx]) else 0
                for rec_idx in rec_indices
            ]

            for k in k_values:
                popularity_metrics[k]["precisions"].append(
                    precision_at_k(relevant_flags, k)
                )
                popularity_metrics[k]["hits"].append(
                    hit_rate(relevant_flags, k)
                )

        # ════════════════════════════════════════════════
        # AGGREGATE RESULTS
        # ════════════════════════════════════════════════

        print("\n" + "=" * 70)
        print("  EVALUATION RESULTS")
        print("=" * 70)

        results = {
            "model": "content_based_tfidf",
            "num_test_queries": int(len(test_indices)),
            "num_total_items": int(num_items),
            "mean_average_precision": round(float(np.mean(model_aps)), 4),
            "mean_diversity_at_5": round(
                float(np.mean(model_diversities)), 4
            ) if model_diversities else 0.0,
            "metrics_by_k": {},
            "baselines": {"random": {}, "popularity": {}},
            "quality_gates": {}
        }

        header = (f"{'K':>4} | {'Prec@K':>8} | {'Recall@K':>9} | "
                  f"{'NDCG@K':>8} | {'HitRate':>8} | "
                  f"{'Rand_P@K':>9} | {'Pop_P@K':>8}")
        print(f"\n{header}")
        print("-" * 80)

        for k in k_values:
            # Our model
            avg_prec = float(np.mean(model_metrics[k]["precisions"]))
            avg_rec = float(np.mean(model_metrics[k]["recalls"]))
            avg_ndcg = float(np.mean(model_metrics[k]["ndcgs"]))
            avg_hit = float(np.mean(model_metrics[k]["hits"]))

            # Baselines
            avg_rand_prec = float(np.mean(random_metrics[k]["precisions"]))
            avg_pop_prec = float(np.mean(popularity_metrics[k]["precisions"]))
            avg_rand_hit = float(np.mean(random_metrics[k]["hits"]))
            avg_pop_hit = float(np.mean(popularity_metrics[k]["hits"]))

            print(f"{k:>4} | {avg_prec:>8.4f} | {avg_rec:>9.4f} | "
                  f"{avg_ndcg:>8.4f} | {avg_hit:>8.4f} | "
                  f"{avg_rand_prec:>9.4f} | {avg_pop_prec:>8.4f}")

            # Store results
            results["metrics_by_k"][str(k)] = {
                "precision": round(avg_prec, 4),
                "recall": round(avg_rec, 4),
                "ndcg": round(avg_ndcg, 4),
                "hit_rate": round(avg_hit, 4)
            }

            results["baselines"]["random"][str(k)] = {
                "precision": round(avg_rand_prec, 4),
                "hit_rate": round(avg_rand_hit, 4)
            }

            results["baselines"]["popularity"][str(k)] = {
                "precision": round(avg_pop_prec, 4),
                "hit_rate": round(avg_pop_hit, 4)
            }

            # Log to MLflow
            mlflow.log_metric(f"precision_at_{k}", avg_prec)
            mlflow.log_metric(f"recall_at_{k}", avg_rec)
            mlflow.log_metric(f"ndcg_at_{k}", avg_ndcg)
            mlflow.log_metric(f"hit_rate_at_{k}", avg_hit)
            mlflow.log_metric(f"random_precision_at_{k}", avg_rand_prec)
            mlflow.log_metric(f"popularity_precision_at_{k}", avg_pop_prec)

        # Aggregate metrics
        map_score = results["mean_average_precision"]
        diversity = results["mean_diversity_at_5"]
        mlflow.log_metric("MAP", map_score)
        mlflow.log_metric("mean_diversity_at_5", diversity)

        print(f"\n  Mean Average Precision (MAP): {map_score:.4f}")
        print(f"  Mean Diversity@5:             {diversity:.4f}")

        # ════════════════════════════════════════════════
        # IMPROVEMENT OVER BASELINES
        # ════════════════════════════════════════════════

        print("\n  IMPROVEMENT OVER BASELINES (at K=5):")

        model_p5 = results["metrics_by_k"]["5"]["precision"]
        random_p5 = results["baselines"]["random"]["5"]["precision"]
        pop_p5 = results["baselines"]["popularity"]["5"]["precision"]

        imp_random = (
            ((model_p5 - random_p5) / random_p5 * 100)
            if random_p5 > 0 else float('inf')
        )
        imp_pop = (
            ((model_p5 - pop_p5) / pop_p5 * 100)
            if pop_p5 > 0 else float('inf')
        )

        print(f"  vs Random:     +{imp_random:.1f}%")
        print(f"  vs Popularity: +{imp_pop:.1f}%")

        mlflow.log_metric("improvement_over_random_pct", round(imp_random, 2))
        mlflow.log_metric("improvement_over_popularity_pct", round(imp_pop, 2))

        # ════════════════════════════════════════════════
        # COVERAGE ANALYSIS
        # ════════════════════════════════════════════════

        print("\n  COVERAGE ANALYSIS:")

        all_recommended = set()
        for query_idx in test_indices[:100]:
            sim_scores = cosine_similarity(
                tfidf_matrix[query_idx:query_idx + 1], tfidf_matrix
            ).flatten()
            top_indices = sim_scores.argsort()[::-1][1:6]
            all_recommended.update(top_indices.tolist())

        coverage = len(all_recommended) / num_items
        print(f"  Catalog Coverage: {coverage:.4f} "
              f"({len(all_recommended)}/{num_items} items)")
        results["coverage"] = round(coverage, 4)
        mlflow.log_metric("catalog_coverage", coverage)

        # ════════════════════════════════════════════════
        # QUALITY GATES (PASS / FAIL)
        # ════════════════════════════════════════════════

        print("\n  QUALITY GATES:")

        min_prec = eval_params["min_precision_at_5"]
        min_hit = eval_params["min_hit_rate"]

        prec_pass = model_p5 >= min_prec
        hit_pass = results["metrics_by_k"]["5"]["hit_rate"] >= min_hit
        beats_random = model_p5 > random_p5
        beats_popularity = model_p5 > pop_p5
        overall_pass = prec_pass and hit_pass and beats_random

        results["quality_gates"] = {
            "precision_at_5_threshold": min_prec,
            "precision_at_5_actual": model_p5,
            "precision_at_5_passed": prec_pass,
            "hit_rate_threshold": min_hit,
            "hit_rate_actual": results["metrics_by_k"]["5"]["hit_rate"],
            "hit_rate_passed": hit_pass,
            "beats_random_baseline": beats_random,
            "beats_popularity_baseline": beats_popularity,
            "overall_pass": overall_pass
        }

        gate_status = "PASS" if overall_pass else "FAIL"

        print(f"  Precision@5 >= {min_prec}:    "
              f"{'PASS' if prec_pass else 'FAIL'} "
              f"(actual: {model_p5:.4f})")
        print(f"  Hit Rate@5  >= {min_hit}:    "
              f"{'PASS' if hit_pass else 'FAIL'} "
              f"(actual: {results['metrics_by_k']['5']['hit_rate']:.4f})")
        print(f"  Beats Random:          "
              f"{'PASS' if beats_random else 'FAIL'}")
        print(f"  Beats Popularity:      "
              f"{'PASS' if beats_popularity else 'FAIL'}")
        print(f"\n  OVERALL: {gate_status}")

        mlflow.log_metric(
            "quality_gate_passed",
            1.0 if overall_pass else 0.0
        )

        # ════════════════════════════════════════════════
        # GENERATE VISUALIZATIONS
        # ════════════════════════════════════════════════

        print("\n  Generating evaluation charts...")
        os.makedirs("reports", exist_ok=True)

        # ── Chart 1: Three-Panel Line Charts ──
        # Precision@K, NDCG@K, Hit Rate@K across K values
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Extract values for plotting
        model_precs = [
            results["metrics_by_k"][str(k)]["precision"] for k in k_values
        ]
        random_precs = [
            results["baselines"]["random"][str(k)]["precision"] for k in k_values
        ]
        pop_precs = [
            results["baselines"]["popularity"][str(k)]["precision"]
            for k in k_values
        ]
        model_ndcgs = [
            results["metrics_by_k"][str(k)]["ndcg"] for k in k_values
        ]
        model_hits = [
            results["metrics_by_k"][str(k)]["hit_rate"] for k in k_values
        ]
        random_hits = [
            results["baselines"]["random"][str(k)]["hit_rate"] for k in k_values
        ]
        pop_hits = [
            results["baselines"]["popularity"][str(k)]["hit_rate"]
            for k in k_values
        ]

        # Panel 1: Precision@K
        axes[0].plot(
            k_values, model_precs, 'b-o',
            linewidth=2, markersize=8, label='TF-IDF Model'
        )
        axes[0].plot(
            k_values, random_precs, 'r--^',
            linewidth=2, markersize=8, label='Random Baseline'
        )
        axes[0].plot(
            k_values, pop_precs, 'g-.s',
            linewidth=2, markersize=8, label='Popularity Baseline'
        )
        axes[0].set_xlabel('K (Number of Recommendations)', fontsize=12)
        axes[0].set_ylabel('Precision@K', fontsize=12)
        axes[0].set_title('Precision@K Comparison', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(0, 1.0)

        # Panel 2: NDCG@K
        axes[1].plot(
            k_values, model_ndcgs, 'b-o',
            linewidth=2, markersize=8, label='TF-IDF Model'
        )
        axes[1].set_xlabel('K (Number of Recommendations)', fontsize=12)
        axes[1].set_ylabel('NDCG@K', fontsize=12)
        axes[1].set_title('NDCG@K (Ranking Quality)', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0, 1.0)

        # Panel 3: Hit Rate@K
        axes[2].plot(
            k_values, model_hits, 'b-o',
            linewidth=2, markersize=8, label='TF-IDF Model'
        )
        axes[2].plot(
            k_values, random_hits, 'r--^',
            linewidth=2, markersize=8, label='Random Baseline'
        )
        axes[2].plot(
            k_values, pop_hits, 'g-.s',
            linewidth=2, markersize=8, label='Popularity Baseline'
        )
        axes[2].set_xlabel('K (Number of Recommendations)', fontsize=12)
        axes[2].set_ylabel('Hit Rate@K', fontsize=12)
        axes[2].set_title('Hit Rate@K Comparison', fontsize=14, fontweight='bold')
        axes[2].legend(fontsize=10)
        axes[2].grid(True, alpha=0.3)
        axes[2].set_ylim(0, 1.05)

        plt.tight_layout()
        chart1_path = os.path.join("reports", "precision_recall_curves.png")
        plt.savefig(chart1_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {chart1_path}")

        # ── Chart 2: Bar Chart — Model vs Baselines at K=5 ──
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        model_names = ['TF-IDF\nModel', 'Popularity\nBaseline', 'Random\nBaseline']
        colors = ['#2196F3', '#4CAF50', '#f44336']

        # Precision@5 bars
        prec_values = [model_p5, pop_p5, random_p5]
        bars1 = axes[0].bar(
            model_names, prec_values, color=colors,
            width=0.6, edgecolor='black', linewidth=0.5
        )
        axes[0].set_ylabel('Precision@5', fontsize=12)
        axes[0].set_title(
            'Model vs Baselines (Precision@5)',
            fontsize=14, fontweight='bold'
        )
        axes[0].set_ylim(0, max(prec_values) * 1.3)
        axes[0].axhline(
            y=min_prec, color='orange', linestyle='--',
            linewidth=2, label=f'Min Threshold ({min_prec})'
        )
        axes[0].legend(fontsize=10)

        for bar, val in zip(bars1, prec_values):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2.,
                bar.get_height() + 0.01,
                f'{val:.4f}',
                ha='center', va='bottom', fontweight='bold', fontsize=11
            )

        # Hit Rate@5 bars
        hit_values = [
            results["metrics_by_k"]["5"]["hit_rate"],
            results["baselines"]["popularity"]["5"]["hit_rate"],
            results["baselines"]["random"]["5"]["hit_rate"]
        ]
        bars2 = axes[1].bar(
            model_names, hit_values, color=colors,
            width=0.6, edgecolor='black', linewidth=0.5
        )
        axes[1].set_ylabel('Hit Rate@5', fontsize=12)
        axes[1].set_title(
            'Model vs Baselines (Hit Rate@5)',
            fontsize=14, fontweight='bold'
        )
        axes[1].set_ylim(0, 1.15)
        axes[1].axhline(
            y=min_hit, color='orange', linestyle='--',
            linewidth=2, label=f'Min Threshold ({min_hit})'
        )
        axes[1].legend(fontsize=10)

        for bar, val in zip(bars2, hit_values):
            axes[1].text(
                bar.get_x() + bar.get_width() / 2.,
                bar.get_height() + 0.01,
                f'{val:.4f}',
                ha='center', va='bottom', fontweight='bold', fontsize=11
            )

        plt.tight_layout()
        chart2_path = os.path.join("reports", "metric_comparison.png")
        plt.savefig(chart2_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {chart2_path}")

        # ── Chart 3: Heatmap of All Metrics ──
        fig, ax = plt.subplots(figsize=(10, 6))

        metrics_matrix = []
        metric_names = ['Precision', 'Recall', 'NDCG', 'Hit Rate']

        for k in k_values:
            row = [
                results["metrics_by_k"][str(k)]["precision"],
                results["metrics_by_k"][str(k)]["recall"],
                results["metrics_by_k"][str(k)]["ndcg"],
                results["metrics_by_k"][str(k)]["hit_rate"]
            ]
            metrics_matrix.append(row)

        metrics_df = pd.DataFrame(
            metrics_matrix,
            index=[f"K={k}" for k in k_values],
            columns=metric_names
        )

        sns.heatmap(
            metrics_df, annot=True, fmt='.4f', cmap='YlOrRd',
            linewidths=0.5, ax=ax, vmin=0, vmax=1,
            annot_kws={"size": 12, "fontweight": "bold"}
        )
        ax.set_title(
            'Evaluation Metrics Heatmap (TF-IDF Model)',
            fontsize=14, fontweight='bold'
        )
        ax.set_ylabel('Top-K', fontsize=12)
        ax.set_xlabel('Metric', fontsize=12)

        plt.tight_layout()
        chart3_path = os.path.join("reports", "metrics_heatmap.png")
        plt.savefig(chart3_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {chart3_path}")

        # ── Chart 4: Improvement Over Baselines Bar Chart ──
        fig, ax = plt.subplots(figsize=(8, 5))

        improvement_labels = ['vs Random\nBaseline', 'vs Popularity\nBaseline']
        improvement_values = [imp_random, imp_pop]
        imp_colors = ['#FF5722', '#FF9800']

        bars = ax.bar(
            improvement_labels, improvement_values, color=imp_colors,
            width=0.5, edgecolor='black', linewidth=0.5
        )
        ax.set_ylabel('Improvement (%)', fontsize=12)
        ax.set_title(
            'TF-IDF Model Improvement Over Baselines (Precision@5)',
            fontsize=14, fontweight='bold'
        )
        ax.axhline(
            y=0, color='black', linestyle='-', linewidth=0.5
        )

        for bar, val in zip(bars, improvement_values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                bar.get_height() + 0.5,
                f'+{val:.1f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=13
            )

        plt.tight_layout()
        chart4_path = os.path.join("reports", "improvement_chart.png")
        plt.savefig(chart4_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {chart4_path}")

        # ── Chart 5: Quality Gate Dashboard ──
        fig, ax = plt.subplots(figsize=(10, 4))

        gate_names = [
            f'Precision@5\n>= {min_prec}',
            f'Hit Rate@5\n>= {min_hit}',
            'Beats\nRandom',
            'Beats\nPopularity'
        ]
        gate_passed = [prec_pass, hit_pass, beats_random, beats_popularity]
        gate_colors = ['#4CAF50' if p else '#f44336' for p in gate_passed]
        gate_labels = ['PASS' if p else 'FAIL' for p in gate_passed]

        bars = ax.barh(
            gate_names, [1] * len(gate_names), color=gate_colors,
            height=0.6, edgecolor='black', linewidth=0.5
        )

        for i, (bar, label) in enumerate(zip(bars, gate_labels)):
            ax.text(
                0.5, bar.get_y() + bar.get_height() / 2.,
                label,
                ha='center', va='center', fontweight='bold',
                fontsize=16, color='white'
            )

        ax.set_xlim(0, 1)
        ax.set_xticks([])
        overall_color = '#4CAF50' if overall_pass else '#f44336'
        ax.set_title(
            f'Quality Gate Dashboard — Overall: {gate_status}',
            fontsize=14, fontweight='bold', color=overall_color
        )

        plt.tight_layout()
        chart5_path = os.path.join("reports", "quality_gates_dashboard.png")
        plt.savefig(chart5_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {chart5_path}")

        # ── Log all charts to MLflow ──
        mlflow.log_artifacts("reports", artifact_path="evaluation_charts")

        # ── Save full evaluation report as JSON ──
        report_path = os.path.join("reports", "evaluation_report.json")
        with open(report_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved: {report_path}")

        mlflow.log_artifact(report_path)

        # ── Save flat metrics for DVC tracking ──
        eval_metrics_flat = {
            "precision_at_5": model_p5,
            "recall_at_5": results["metrics_by_k"]["5"]["recall"],
            "ndcg_at_5": results["metrics_by_k"]["5"]["ndcg"],
            "hit_rate_at_5": results["metrics_by_k"]["5"]["hit_rate"],
            "MAP": map_score,
            "diversity_at_5": diversity,
            "coverage": results["coverage"],
            "improvement_over_random_pct": round(imp_random, 2),
            "improvement_over_popularity_pct": round(imp_pop, 2),
            "quality_gate_passed": overall_pass,
            "beats_random": beats_random,
            "beats_popularity": beats_popularity
        }

        with open("eval_metrics.json", "w") as f:
            json.dump(eval_metrics_flat, f, indent=2)

        mlflow.log_artifact("eval_metrics.json")

    # ════════════════════════════════════════════════
    # FINAL SUMMARY
    # ════════════════════════════════════════════════

    print("\n" + "=" * 70)
    print("  EVALUATION COMPLETE")
    print("=" * 70)
    print("\n  Full Report:     reports/evaluation_report.json")
    print("  Charts:          reports/precision_recall_curves.png")
    print("                   reports/metric_comparison.png")
    print("                   reports/metrics_heatmap.png")
    print("                   reports/improvement_chart.png")
    print("                   reports/quality_gates_dashboard.png")
    print("  MLflow:          Run 'mlflow ui' to see metrics and charts")
    print("\n  Overall Quality Gate: {gate_status}")

    if not overall_pass:
        print("\n  The model did not pass all quality gates.")
        print("  Consider adjusting parameters in params.yaml and retraining.")
    else:
        print("\n  The model passed all quality gates!")
        print("  It is ready for deployment.")

    return results


if __name__ == "__main__":
    evaluate_model()