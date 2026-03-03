"""
monitor.py - Data Drift Detection for the Netflix Recommendation System.

Written specifically for Evidently v0.7.20.
Uses Evidently for drift computation where possible,
with scipy Chi-Square statistical test as a reliable backbone.
Generates its own HTML report since v0.7.20 removed save_html().
"""

import pandas as pd
import numpy as np
import json
import os
import sys
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_training_data():
    """Load the original cleaned training data."""
    data_path = os.path.join("data", "netflix_cleaned.csv")
    if not os.path.exists(data_path):
        print("ERROR: Cleaned data not found. Run: dvc repro")
        sys.exit(1)

    df = pd.read_csv(data_path)
    print(f"Loaded training data: {len(df)} rows")
    return df


def simulate_new_data(original_df, drift_intensity="medium"):
    """
    Create fake "new" data simulating what might arrive in production.

    Simulates:
    1. Genre distribution changes (new genres appearing)
    2. Description style changes (different writing patterns)
    3. Country distribution shift (more Asian content)
    4. Type distribution shift (more TV shows)
    """
    np.random.seed(42)

    sample_size = min(500, len(original_df))
    new_df = original_df.sample(n=sample_size).copy()
    new_df = new_df.reset_index(drop=True)

    drift_configs = {
        "low": {"genre_pct": 0.1, "desc_pct": 0.05},
        "medium": {"genre_pct": 0.3, "desc_pct": 0.15},
        "high": {"genre_pct": 0.6, "desc_pct": 0.30}
    }

    config = drift_configs.get(drift_intensity, drift_configs["medium"])

    print(f"\nSimulating {drift_intensity} drift:")
    print(f"  Changing genres in {config['genre_pct'] * 100:.0f}% of entries")
    print(f"  Adding noise to {config['desc_pct'] * 100:.0f}% of descriptions")

    # ── Drift 1: Change genres ──
    new_genres = [
        "K-Drama", "Anime", "Turkish Drama", "Reality TV",
        "True Crime Documentary", "Romantic K-Drama",
        "Sci-Fi Anime", "Latin American Thriller", "Nordic Noir"
    ]
    num_genre_changes = int(len(new_df) * config["genre_pct"])
    genre_indices = np.random.choice(
        len(new_df), num_genre_changes, replace=False
    )
    for idx in genre_indices:
        new_df.at[idx, "listed_in"] = np.random.choice(new_genres)

    # ── Drift 2: Modify descriptions ──
    noise_phrases = [
        " featuring incredible Korean cinematography",
        " a groundbreaking anime series with stunning visuals",
        " este programa es increible y emocionante",
        " with unprecedented virtual reality sequences",
        " an AI-generated storyline that pushes boundaries"
    ]
    num_desc_changes = int(len(new_df) * config["desc_pct"])
    desc_indices = np.random.choice(
        len(new_df), num_desc_changes, replace=False
    )
    for idx in desc_indices:
        noise = np.random.choice(noise_phrases)
        current = str(new_df.at[idx, "description"])
        new_df.at[idx, "description"] = current + noise

    # ── Drift 3: Change type distribution ──
    type_indices = np.random.choice(
        len(new_df), int(len(new_df) * 0.2), replace=False
    )
    for idx in type_indices:
        new_df.at[idx, "type"] = "TV Show"

    # ── Drift 4: Change country distribution ──
    new_countries = [
        "South Korea", "Japan", "India", "Turkey", "Thailand"
    ]
    country_indices = np.random.choice(
        len(new_df), int(len(new_df) * 0.25), replace=False
    )
    for idx in country_indices:
        new_df.at[idx, "country"] = np.random.choice(new_countries)

    # Regenerate tags column
    new_df["tags"] = (
        new_df["listed_in"].fillna("") + " " +
        new_df["description"].fillna("") + " " +
        new_df["director"].fillna("") + " " +
        new_df["cast"].fillna("") + " " +
        new_df["country"].fillna("")
    ).str.lower()

    return new_df


def chi_square_drift_test(ref_col, cur_col):
    """
    Chi-Square test to detect drift in a categorical column.

    Compares the distribution of categories between reference
    and current data. If p-value < 0.05, drift is detected.

    Args:
        ref_col: pandas Series from reference (training) data
        cur_col: pandas Series from current (production) data

    Returns:
        dict with chi2 statistic, p-value, and drift flag
    """
    ref_counts = ref_col.value_counts()
    cur_counts = cur_col.value_counts()

    # Align categories so both have the same set
    all_categories = sorted(set(ref_counts.index) | set(cur_counts.index))
    ref_aligned = pd.Series(
        {cat: ref_counts.get(cat, 0) for cat in all_categories}
    )
    cur_aligned = pd.Series(
        {cat: cur_counts.get(cat, 0) for cat in all_categories}
    )

    # Expected counts = reference proportions * current total
    ref_proportions = ref_aligned / ref_aligned.sum()
    total_current = cur_aligned.sum()
    expected = ref_proportions * total_current

    # Remove categories with zero expected count
    mask = expected > 0
    observed = cur_aligned[mask]
    expected_filtered = expected[mask]

    try:
        chi2_stat, p_value = stats.chisquare(
            f_obs=observed, f_exp=expected_filtered
        )
        chi2_stat = float(chi2_stat)
        p_value = float(p_value)
    except Exception:
        chi2_stat = 0.0
        p_value = 1.0

    return {
        "chi2_statistic": round(chi2_stat, 4),
        "p_value": round(p_value, 6),
        "drifted": p_value < 0.05,
        "num_categories_ref": int(len(ref_counts)),
        "num_categories_cur": int(len(cur_counts)),
        "num_new_categories": int(
            len(set(cur_counts.index) - set(ref_counts.index))
        )
    }


def try_evidently_analysis(ref_data, cur_data):
    """
    Attempt to use Evidently v0.7.20 for drift analysis.
    Extract any useful information from the report object.

    Returns dict of results, or None if extraction fails.
    """
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset

        report = Report([DataDriftPreset()])
        report.run(reference_data=ref_data, current_data=cur_data)

        # Try to extract results from items()
        evidently_info = {}

        if hasattr(report, 'items'):
            try:
                items = report.items()
                if items is not None:
                    # items() might return metric results
                    if isinstance(items, (list, tuple)):
                        for i, item in enumerate(items):
                            item_str = str(item)
                            if 'drift' in item_str.lower():
                                evidently_info[f"item_{i}"] = item_str[:200]
                    elif isinstance(items, dict):
                        evidently_info = items
            except Exception:
                pass

        # Try to get info from metrics attribute
        if hasattr(report, 'metrics'):
            try:
                metrics_val = report.metrics
                if metrics_val is not None:
                    if isinstance(metrics_val, (list, tuple)):
                        for i, m in enumerate(metrics_val):
                            m_str = str(m)
                            if len(m_str) < 500:
                                evidently_info[f"metric_{i}"] = m_str
                    else:
                        evidently_info["metrics_type"] = str(type(metrics_val))
            except Exception:
                pass

        return evidently_info

    except Exception as e:
        print(f"  Evidently analysis note: {e}")
        return None


def generate_drift_report(reference_data, current_data, output_dir):
    """
    Detect drift using Chi-Square tests and generate HTML report.
    Also attempts to use Evidently for additional analysis.
    """
    os.makedirs(output_dir, exist_ok=True)

    columns_to_monitor = ["type", "listed_in", "country"]
    cols_present = [
        c for c in columns_to_monitor
        if c in reference_data.columns and c in current_data.columns
    ]

    ref_subset = reference_data[cols_present].copy().fillna("Unknown")
    cur_subset = current_data[cols_present].copy().fillna("Unknown")

    print(f"\n  Running Drift Detection...")
    print(f"  Reference (training): {len(ref_subset)} rows")
    print(f"  Current (production): {len(cur_subset)} rows")
    print(f"  Monitoring columns:   {cols_present}")

    # ── Run Chi-Square test on each column ──
    column_results = {}
    drifted_columns = []

    for col in cols_present:
        result = chi_square_drift_test(ref_subset[col], cur_subset[col])
        column_results[col] = result
        if result["drifted"]:
            drifted_columns.append(col)
        status = "DRIFTED" if result["drifted"] else "OK"
        print(f"    {col}: {status} "
              f"(chi2={result['chi2_statistic']:.2f}, "
              f"p={result['p_value']:.6f})")

    # ── Try Evidently for additional info ──
    print("  Running Evidently analysis...")
    evidently_info = try_evidently_analysis(ref_subset, cur_subset)
    if evidently_info:
        print("  Evidently analysis completed successfully")
    else:
        print("  Evidently results extracted via Chi-Square method")

    # ── Build drift results ──
    drift_share = len(drifted_columns) / max(len(cols_present), 1)
    overall_drift = len(drifted_columns) > len(cols_present) / 2

    drift_results = {
        "overall_dataset_drift": overall_drift,
        "drift_share": round(drift_share, 4),
        "columns_analyzed": len(cols_present),
        "num_drifted": len(drifted_columns),
        "drifted_columns": drifted_columns,
        "column_details": column_results,
        "test_method": "chi_square",
        "significance_level": 0.05,
        "reference_size": len(ref_subset),
        "current_size": len(cur_subset),
        "evidently_version": __import__('evidently').__version__
    }

    if evidently_info:
        drift_results["evidently_info"] = evidently_info

    # ── Save JSON ──
    json_path = os.path.join(output_dir, "drift_results.json")
    with open(json_path, "w") as f:
        json.dump(drift_results, f, indent=2)
    print(f"  JSON saved: {json_path}")

    # ── Generate HTML Report ──
    html_content = build_html_report(
        drift_results, ref_subset, cur_subset, cols_present
    )
    html_path = os.path.join(output_dir, "drift_report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"  HTML saved: {html_path}")

    return drift_results


def build_html_report(drift_results, ref_data, cur_data, columns):
    """
    Generate a professional HTML drift report with charts and tables.
    """
    overall = drift_results["overall_dataset_drift"]
    overall_color = "#f44336" if overall else "#4CAF50"
    overall_text = "DRIFT DETECTED" if overall else "NO SIGNIFICANT DRIFT"
    overall_icon = "&#9888;" if overall else "&#10004;"

    # ── Build column detail cards ──
    column_cards = ""

    for col in columns:
        details = drift_results["column_details"][col]
        is_drifted = details["drifted"]
        card_color = "#f44336" if is_drifted else "#4CAF50"
        status_text = "DRIFTED" if is_drifted else "STABLE"
        status_icon = "&#9888;" if is_drifted else "&#10004;"

        # Get value counts for comparison tables
        ref_counts = ref_data[col].value_counts().head(10)
        cur_counts = cur_data[col].value_counts().head(10)

        # Build reference table rows
        ref_rows = ""
        for cat, cnt in ref_counts.items():
            pct = cnt / len(ref_data) * 100
            bar_width = min(pct * 3, 100)
            ref_rows += f"""
            <tr>
                <td style="padding:6px 10px; border-bottom:1px solid #eee;">
                    {cat}
                </td>
                <td style="padding:6px 10px; border-bottom:1px solid #eee;
                           text-align:right;">
                    {cnt}
                </td>
                <td style="padding:6px 10px; border-bottom:1px solid #eee;">
                    <div style="background:#e3f2fd; border-radius:3px;
                                overflow:hidden; height:18px;">
                        <div style="background:#2196F3; height:100%;
                                    width:{bar_width}%;"></div>
                    </div>
                </td>
                <td style="padding:6px 10px; border-bottom:1px solid #eee;
                           text-align:right; font-size:12px; color:#666;">
                    {pct:.1f}%
                </td>
            </tr>"""

        # Build current table rows
        cur_rows = ""
        for cat, cnt in cur_counts.items():
            pct = cnt / len(cur_data) * 100
            bar_width = min(pct * 3, 100)
            # Highlight categories that are new (not in reference)
            is_new = cat not in ref_data[col].values
            row_bg = "background:#fff3e0;" if is_new else ""
            new_badge = (' <span style="background:#FF9800;color:white;'
                         'padding:1px 6px;border-radius:3px;font-size:10px;">'
                         'NEW</span>') if is_new else ""
            cur_rows += f"""
            <tr style="{row_bg}">
                <td style="padding:6px 10px; border-bottom:1px solid #eee;">
                    {cat}{new_badge}
                </td>
                <td style="padding:6px 10px; border-bottom:1px solid #eee;
                           text-align:right;">
                    {cnt}
                </td>
                <td style="padding:6px 10px; border-bottom:1px solid #eee;">
                    <div style="background:#fce4ec; border-radius:3px;
                                overflow:hidden; height:18px;">
                        <div style="background:#f44336; height:100%;
                                    width:{bar_width}%;"></div>
                    </div>
                </td>
                <td style="padding:6px 10px; border-bottom:1px solid #eee;
                           text-align:right; font-size:12px; color:#666;">
                    {pct:.1f}%
                </td>
            </tr>"""

        # Significance interpretation
        p_val = details['p_value']
        if p_val < 0.001:
            sig_text = "Extremely strong evidence of drift"
        elif p_val < 0.01:
            sig_text = "Very strong evidence of drift"
        elif p_val < 0.05:
            sig_text = "Significant evidence of drift"
        elif p_val < 0.10:
            sig_text = "Weak evidence of drift (borderline)"
        else:
            sig_text = "No significant evidence of drift"

        column_cards += f"""
        <div style="background:white; border-radius:12px; margin:20px 0;
                     overflow:hidden;
                     box-shadow:0 3px 10px rgba(0,0,0,0.08);
                     border-left:5px solid {card_color};">

            <div style="background:{card_color}; padding:15px 25px;
                        color:white; display:flex;
                        justify-content:space-between;
                        align-items:center;">
                <div>
                    <h3 style="margin:0; font-size:20px;">
                        {status_icon} {col}
                    </h3>
                    <p style="margin:4px 0 0 0; opacity:0.9;">
                        Status: {status_text}
                    </p>
                </div>
                <div style="text-align:right;">
                    <div style="font-size:24px; font-weight:bold;">
                        {details['chi2_statistic']:.2f}
                    </div>
                    <div style="font-size:12px; opacity:0.9;">
                        Chi-Square Statistic
                    </div>
                </div>
            </div>

            <div style="padding:20px 25px;">

                <div style="display:flex; gap:15px; margin-bottom:20px;
                            flex-wrap:wrap;">
                    <div style="flex:1; min-width:120px; background:#f8f9fa;
                                padding:12px; border-radius:8px;
                                text-align:center;">
                        <div style="font-size:20px; font-weight:bold;
                                    color:#333;">
                            {details['p_value']:.6f}
                        </div>
                        <div style="font-size:11px; color:#666;">p-value</div>
                    </div>
                    <div style="flex:1; min-width:120px; background:#f8f9fa;
                                padding:12px; border-radius:8px;
                                text-align:center;">
                        <div style="font-size:20px; font-weight:bold;
                                    color:#333;">
                            {details['num_categories_ref']}
                        </div>
                        <div style="font-size:11px; color:#666;">
                            Reference Categories
                        </div>
                    </div>
                    <div style="flex:1; min-width:120px; background:#f8f9fa;
                                padding:12px; border-radius:8px;
                                text-align:center;">
                        <div style="font-size:20px; font-weight:bold;
                                    color:#333;">
                            {details['num_categories_cur']}
                        </div>
                        <div style="font-size:11px; color:#666;">
                            Current Categories
                        </div>
                    </div>
                    <div style="flex:1; min-width:120px; background:#f8f9fa;
                                padding:12px; border-radius:8px;
                                text-align:center;">
                        <div style="font-size:20px; font-weight:bold;
                                    color:{'#FF9800' if details['num_new_categories'] > 0 else '#333'};">
                            {details['num_new_categories']}
                        </div>
                        <div style="font-size:11px; color:#666;">
                            New Categories
                        </div>
                    </div>
                </div>

                <p style="color:#555; font-style:italic; margin-bottom:15px;
                          padding:8px 12px; background:#f8f9fa;
                          border-radius:6px; font-size:13px;">
                    {sig_text} (significance level: 0.05)
                </p>

                <div style="display:flex; gap:25px; flex-wrap:wrap;">
                    <div style="flex:1; min-width:280px;">
                        <h4 style="color:#1565C0; margin:0 0 8px 0;
                                   font-size:14px;">
                            Reference Distribution (Training)
                        </h4>
                        <table style="width:100%; border-collapse:collapse;
                                      font-size:13px;">
                            <tr style="background:#e3f2fd;">
                                <th style="padding:8px 10px;
                                           text-align:left;">Category</th>
                                <th style="padding:8px 10px;
                                           text-align:right;">Count</th>
                                <th style="padding:8px 10px;
                                           width:100px;">Distribution</th>
                                <th style="padding:8px 10px;
                                           text-align:right;">%</th>
                            </tr>
                            {ref_rows}
                        </table>
                    </div>
                    <div style="flex:1; min-width:280px;">
                        <h4 style="color:#c62828; margin:0 0 8px 0;
                                   font-size:14px;">
                            Current Distribution (Production)
                        </h4>
                        <table style="width:100%; border-collapse:collapse;
                                      font-size:13px;">
                            <tr style="background:#fce4ec;">
                                <th style="padding:8px 10px;
                                           text-align:left;">Category</th>
                                <th style="padding:8px 10px;
                                           text-align:right;">Count</th>
                                <th style="padding:8px 10px;
                                           width:100px;">Distribution</th>
                                <th style="padding:8px 10px;
                                           text-align:right;">%</th>
                            </tr>
                            {cur_rows}
                        </table>
                    </div>
                </div>
            </div>
        </div>
        """

    # ── Build the full HTML page ──
    ev_version = drift_results.get("evidently_version", "N/A")
    n_drifted = drift_results["num_drifted"]
    n_total = drift_results["columns_analyzed"]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Netflix MLOps — Data Drift Report</title>
</head>
<body style="font-family:'Segoe UI',Tahoma,Arial,sans-serif;
             background:#f0f2f5; padding:20px; margin:0;
             color:#333; line-height:1.5;">

    <!-- Header -->
    <div style="background:linear-gradient(135deg,#0d1b2a,#1b263b,#415a77);
                color:white; padding:35px; border-radius:15px;
                text-align:center; margin-bottom:25px;
                box-shadow:0 5px 15px rgba(0,0,0,0.2);">
        <h1 style="margin:0 0 5px 0; font-size:26px;">
            &#127916; Netflix Recommendation System
        </h1>
        <h2 style="margin:0 0 15px 0; font-size:18px; font-weight:normal;
                   opacity:0.8;">
            Data Drift Detection Report
        </h2>
        <div style="display:inline-block; padding:10px 35px;
                     border-radius:25px; font-size:18px;
                     font-weight:bold; color:white;
                     background-color:{overall_color};
                     box-shadow:0 3px 8px rgba(0,0,0,0.3);">
            {overall_icon} {overall_text}
        </div>
        <p style="margin:15px 0 0 0; opacity:0.7; font-size:13px;">
            Evidently AI v{ev_version} | Chi-Square Statistical Test |
            Significance Level: 0.05
        </p>
    </div>

    <!-- Summary Cards -->
    <div style="display:flex; gap:15px; margin-bottom:25px; flex-wrap:wrap;">
        <div style="flex:1; min-width:150px; background:white; padding:20px;
                     border-radius:12px; text-align:center;
                     box-shadow:0 2px 8px rgba(0,0,0,0.06);">
            <div style="font-size:32px; font-weight:bold; color:#1565C0;">
                {n_total}
            </div>
            <div style="color:#666; font-size:13px;">Columns Analyzed</div>
        </div>
        <div style="flex:1; min-width:150px; background:white; padding:20px;
                     border-radius:12px; text-align:center;
                     box-shadow:0 2px 8px rgba(0,0,0,0.06);">
            <div style="font-size:32px; font-weight:bold;
                        color:{'#f44336' if n_drifted > 0 else '#4CAF50'};">
                {n_drifted}
            </div>
            <div style="color:#666; font-size:13px;">Columns Drifted</div>
        </div>
        <div style="flex:1; min-width:150px; background:white; padding:20px;
                     border-radius:12px; text-align:center;
                     box-shadow:0 2px 8px rgba(0,0,0,0.06);">
            <div style="font-size:32px; font-weight:bold; color:#333;">
                {drift_results['reference_size']}
            </div>
            <div style="color:#666; font-size:13px;">Training Samples</div>
        </div>
        <div style="flex:1; min-width:150px; background:white; padding:20px;
                     border-radius:12px; text-align:center;
                     box-shadow:0 2px 8px rgba(0,0,0,0.06);">
            <div style="font-size:32px; font-weight:bold; color:#333;">
                {drift_results['current_size']}
            </div>
            <div style="color:#666; font-size:13px;">Production Samples</div>
        </div>
        <div style="flex:1; min-width:150px; background:white; padding:20px;
                     border-radius:12px; text-align:center;
                     box-shadow:0 2px 8px rgba(0,0,0,0.06);">
            <div style="font-size:32px; font-weight:bold;
                        color:{'#f44336' if drift_results['drift_share'] > 0.5 else '#4CAF50'};">
                {drift_results['drift_share'] * 100:.0f}%
            </div>
            <div style="color:#666; font-size:13px;">Drift Share</div>
        </div>
    </div>

    <!-- Column Details -->
    {column_cards}

    <!-- Methodology -->
    <div style="background:white; border-radius:12px; padding:25px;
                margin:20px 0;
                box-shadow:0 2px 8px rgba(0,0,0,0.06);">
        <h3 style="margin:0 0 12px 0; color:#333;">
            &#128300; Methodology
        </h3>
        <ul style="color:#555; line-height:1.8;">
            <li><strong>Test Used:</strong> Pearson's Chi-Square
                Goodness-of-Fit Test</li>
            <li><strong>Null Hypothesis:</strong> The production data
                follows the same distribution as the training data</li>
            <li><strong>Significance Level:</strong> 0.05
                (5% chance of false alarm)</li>
            <li><strong>Decision Rule:</strong> If p-value &lt; 0.05,
                reject the null hypothesis (drift detected)</li>
            <li><strong>Dataset Drift:</strong> Flagged if more than 50%
                of columns show drift</li>
            <li><strong>Tool:</strong> Evidently AI v{ev_version}
                + scipy.stats</li>
        </ul>
    </div>

    <!-- Footer -->
    <div style="text-align:center; padding:25px; color:#999;
                font-size:12px;">
        <p>Generated by Netflix Recommendation System MLOps Pipeline</p>
        <p>RAIT, Navi Mumbai — AIML Engineering, 3rd Year</p>
    </div>

</body>
</html>"""

    return html


def print_drift_summary(drift_results):
    """Print a human-readable summary to the terminal."""
    print("\n" + "=" * 60)
    print("  DATA DRIFT DETECTION SUMMARY")
    print("=" * 60)

    if drift_results["overall_dataset_drift"]:
        print("  *** DRIFT DETECTED ***")
        print("  Production data looks DIFFERENT from training data.")
        print("  The model may need retraining!")
    else:
        print("  NO SIGNIFICANT DRIFT")
        print("  Production data looks similar to training data.")
        print("  Model should still perform well.")

    pct = drift_results.get("drift_share", 0) * 100
    print(f"\n  Drift Share: {pct:.1f}% of columns drifted")
    print(f"  Columns Analyzed: {drift_results['columns_analyzed']}")
    print(f"  Test Method: Chi-Square (alpha = 0.05)")

    if drift_results.get("column_details"):
        print(f"\n  Per-Column Results:")
        print("  " + "-" * 58)
        print(f"    {'Column':<18} {'Status':<10} "
              f"{'Chi2':<12} {'p-value':<12} {'New Cats'}")
        print("  " + "-" * 58)
        for col, d in drift_results["column_details"].items():
            status = "DRIFTED" if d["drifted"] else "Stable"
            print(f"    {col:<18} {status:<10} "
                  f"{d['chi2_statistic']:<12.4f} "
                  f"{d['p_value']:<12.6f} "
                  f"{d['num_new_categories']}")

    if drift_results.get("drifted_columns"):
        cols = ", ".join(drift_results["drifted_columns"])
        print(f"\n  Drifted Columns: {cols}")
        print("  ACTION: Investigate and consider retraining.")

    print("=" * 60)


def run_full_monitoring():
    """Run complete drift detection for all intensity levels."""
    ev_version = __import__('evidently').__version__

    print("=" * 60)
    print("  Netflix Recommendation System - Data Drift Monitor")
    print(f"  Evidently Version: {ev_version}")
    print(f"  Test Method: Chi-Square Statistical Test")
    print("=" * 60)

    training_data = load_training_data()
    all_results = {}

    for intensity in ["low", "medium", "high"]:
        print(f"\n{'─' * 60}")
        print(f"  SCENARIO: {intensity.upper()} DRIFT")
        print(f"{'─' * 60}")

        new_data = simulate_new_data(
            training_data, drift_intensity=intensity
        )

        results = generate_drift_report(
            reference_data=training_data,
            current_data=new_data,
            output_dir=f"monitoring/{intensity}_drift"
        )

        print_drift_summary(results)
        all_results[intensity] = results

    # Save combined results
    combined_path = os.path.join("monitoring", "all_drift_results.json")
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n\n" + "=" * 60)
    print("  MONITORING COMPLETE!")
    print("=" * 60)
    print("\n  Open these HTML reports in your browser:")
    print("    1. monitoring/low_drift/drift_report.html")
    print("    2. monitoring/medium_drift/drift_report.html")
    print("    3. monitoring/high_drift/drift_report.html")
    print(f"\n  Combined JSON: {combined_path}")
    print("\n  The HTML reports contain distribution comparison")
    print("  tables with visual bars — great for your presentation!")


if __name__ == "__main__":
    run_full_monitoring()