# test_and_train.py
import os
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

sns.set(style="whitegrid", palette="muted")
plt.rcParams.update({
    "figure.autolayout": True,
    "axes.labelsize": 11,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

DATA_FN = "data.csv"


def tidy_axes(ax):
    ax.set_axisbelow(True)
    ax.grid(alpha=0.25)
    for spine in ax.spines.values():
        spine.set_color("#333333")
    return ax


def load_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    required_cols = {
        "year", "district", "area", "population_density",
        "urbanization_index", "infrastructure_score",
        "economic_index", "actual_demand", "predicted_baseline"
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    # numeric coercion
    numeric_cols = [
        "year", "population_density", "urbanization_index",
        "infrastructure_score", "economic_index",
        "actual_demand", "predicted_baseline"
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def clean_and_feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # fill numeric NaNs with median (global)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    df["actual_demand"] = df["actual_demand"].clip(lower=0)

    # categorical combined key and code (safe to create)
    df["district_area"] = df["district"].astype(str) + " - " + df["area"].astype(str)
    df["district_area_code"] = df["district_area"].astype("category").cat.codes

    # time features
    df["years_since_2010"] = df["year"] - 2010
    df["is_post_2015"] = (df["year"] >= 2015).astype(int)

    # simple interactions
    df["pop_x_infra"] = df["population_density"] * df["infrastructure_score"]
    df["urban_x_econ"] = df["urbanization_index"] * df["economic_index"]

    # initial demand per density (can be noisy) — keep but treat carefully
    df["demand_per_density"] = df["actual_demand"] / (df["population_density"] + 1e-6)

    # clip extreme numeric outliers (winsorize at 1st/99th)
    for col in ["population_density", "infrastructure_score", "economic_index", "actual_demand"]:
        lower, upper = np.percentile(df[col].dropna(), [1, 99])
        df[col] = df[col].clip(lower, upper)

    return df


def oof_target_encoding(df: pd.DataFrame, group_col: str, target_col: str, n_splits: int = 5, seed: int = RANDOM_SEED) -> pd.Series:
    """
    Compute out-of-fold (GroupKFold-internal) group mean target encoding.
    For each row, the value is the mean target computed from folds that did not include that group's rows.
    Returns a Series aligned with df.index containing the OOF encoding; for groups entirely in one fold,
    it falls back to global mean.
    """
    groups = df[group_col].values
    y = df[target_col].values
    oof = pd.Series(index=df.index, dtype=float)
    gkf = GroupKFold(n_splits=n_splits)
    global_mean = y.mean()
    for train_idx, val_idx in gkf.split(df, y, groups):
        train_df = df.iloc[train_idx]
        means = train_df.groupby(group_col)[target_col].mean()
        # map means to validation rows where available
        val_groups = df.iloc[val_idx][group_col]
        mapped = val_groups.map(means)
        # fill unmapped with global mean
        mapped = mapped.fillna(global_mean)
        oof.iloc[val_idx] = mapped.values
    # any remaining NaNs -> fill with global mean
    oof = oof.fillna(global_mean)
    return oof


def regression_report(y_true: np.ndarray, y_pred: np.ndarray, tolerance_units: float = 20.0) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    pct_within = 100.0 * np.mean(np.abs(y_true - y_pred) <= tolerance_units)
    return {"MAE": mae, "RMSE": rmse, "R2": r2, f"pct_within_{int(tolerance_units)}_units": pct_within}


def train_and_evaluate(df: pd.DataFrame, holdout_year_cut: int = 2020, cv_splits: int = 5, rnd_search_iters: int = 40):
    # final feature list (exclude predicted_baseline)
    base_features = [
        "district_area_code", "year", "population_density", "urbanization_index",
        "infrastructure_score", "economic_index", "years_since_2010", "is_post_2015",
        "pop_x_infra", "urban_x_econ"
    ]

    # create out-of-fold target encoding (use only on training set later)
    # BUT first split time-holdout to avoid future leakage
    train_df = df[df["year"] <= holdout_year_cut].copy()
    test_df = df[df["year"] > holdout_year_cut].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Holdout produced empty train/test. Adjust holdout_year_cut or data.")

    # Compute OOF target-encoding for 'district_area' using only train_df
    train_df = train_df.reset_index()  # preserve original index in column 'index'
    test_df = test_df.reset_index()
    oof_enc = oof_target_encoding(train_df, group_col="district_area", target_col="actual_demand", n_splits=cv_splits)
    train_df["district_area_oof_mean"] = oof_enc.values
    # For test set, map using full train means (safe): aggregated from train only (no leakage from test)
    train_group_means = train_df.groupby("district_area")["actual_demand"].mean()
    test_df["district_area_oof_mean"] = test_df["district_area"].map(train_group_means).fillna(train_df["actual_demand"].mean())

    # Add a feature: historical mean per district (from train only)
    district_means = train_df.groupby("district")["actual_demand"].mean()
    train_df["district_mean_train"] = train_df["district"].map(district_means)
    test_df["district_mean_train"] = test_df["district"].map(district_means).fillna(train_df["actual_demand"].mean())

    # combine feature names
    features = base_features + ["district_area_oof_mean", "district_mean_train"]

    # Prepare arrays
    X_train = train_df[features]
    y_train = train_df["actual_demand"].values
    groups = train_df["district_area"].values

    X_test = test_df[features]
    y_test = test_df["actual_demand"].values

    # scaler fit on training set only
    scaler = StandardScaler()
    Xs_train = scaler.fit_transform(X_train)
    Xs_test = scaler.transform(X_test)

    # base model with constrained hyperparams
    base_model = RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1)

    # Randomized search space
    param_dist = {
        "n_estimators": [100, 200, 300],
        "max_depth": [8, 12, 16, None],
        "min_samples_split": [2, 5, 8, 12],
        "min_samples_leaf": [1, 4, 8, 12],
        "max_features": ["sqrt", 0.5, 0.75]
    }

    # scorer: negative MAE for selection, but we will report MAE and pct-within separately
    scorer = make_scorer(mean_absolute_error, greater_is_better=False)

    gkf = GroupKFold(n_splits=cv_splits)
    rnd = RandomizedSearchCV(base_model, param_distributions=param_dist, n_iter=rnd_search_iters,
                             scoring=scorer, cv=gkf.split(Xs_train, y_train, groups), random_state=RANDOM_SEED,
                             verbose=1, n_jobs=-1)

    rnd.fit(Xs_train, y_train)
    print("RandomizedSearchCV best params:", rnd.best_params_)
    best_model = rnd.best_estimator_

    # evaluate on test (time holdout)
    y_test_pred = best_model.predict(Xs_test)
    holdout_report = regression_report(y_test, y_test_pred, tolerance_units=20.0)
    print("\nHoldout (time split) metrics:")
    print(f"MAE: {holdout_report['MAE']:.3f}, RMSE: {holdout_report['RMSE']:.3f}, R2: {holdout_report['R2']:.3f}, pct_within_20={holdout_report['pct_within_20_units']:.1f}%")

    # attach predictions to full df for diagnostics (use scaler trained on train_df)
    df_out = df.copy()
    # Create same features for full df: compute mapping using train_df aggregates only
    df_out["district_area_oof_mean"] = df_out["district_area"].map(train_group_means).fillna(train_df["actual_demand"].mean())
    df_out["district_mean_train"] = df_out["district"].map(district_means).fillna(train_df["actual_demand"].mean())
    X_all = df_out[features].values
    Xs_all = scaler.transform(X_all)
    df_out["predicted_ml"] = best_model.predict(Xs_all)

    return {
        "model": best_model,
        "scaler": scaler,
        "features": features,
        "holdout_report": holdout_report,
        "df_out": df_out,
        "train_df": train_df,
        "test_df": test_df,
        "rnd_search": rnd
    }


def plot_residual_diagnostics(y_true: np.ndarray, y_pred: np.ndarray, title_suffix: str = "", save_png: bool = False, out_path: str = None):
    resid = y_true - y_pred
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4.0))
    ax = axes[0]
    ax.scatter(y_pred, resid, alpha=0.7, s=18, color="tab:purple")
    ax.axhline(0, color="k", linestyle="--", linewidth=1)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual (true - pred)")
    ax.set_title(f"Residuals vs Predicted {title_suffix}")
    tidy_axes(ax)

    ax = axes[1]
    ax.hist(resid, bins=30, color="tab:orange", alpha=0.9)
    ax.set_title(f"Residual histogram {title_suffix}")
    ax.set_xlabel("Residual")
    tidy_axes(ax)

    plt.tight_layout()
    if save_png and out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()


def per_group_metrics(df_eval: pd.DataFrame, group_col: str = "district", y_col: str = "actual_demand", pred_col: str = "predicted_ml", tolerance_units: float = 20.0) -> Dict[str, Dict[str, float]]:
    groups = {}
    for name, g in df_eval.groupby(group_col):
        y_true = g[y_col].values
        y_pred = g[pred_col].values
        groups[name] = regression_report(y_true, y_pred, tolerance_units=tolerance_units)
    return groups


def plot_district_overall_demand(df: pd.DataFrame, save_png: bool = False, out_dir: str = "."):
    df_agg = df.groupby(["district", "year"])[["actual_demand", "predicted_ml", "predicted_baseline"]].sum().reset_index()
    districts = sorted(df_agg["district"].unique())
    for d in districts:
        sub = df_agg[df_agg["district"] == d].sort_values("year")
        mae = mean_absolute_error(sub["actual_demand"], sub["predicted_ml"])
        rmse = np.sqrt(mean_squared_error(sub["actual_demand"], sub["predicted_ml"]))
        r2 = r2_score(sub["actual_demand"], sub["predicted_ml"])
        print(f"\n{d} summary metrics: MAE: {mae:.2f} RMSE: {rmse:.2f} R2: {r2:.3f}")

        fig, ax = plt.subplots(figsize=(8, 4.2))
        ax.plot(sub["year"], sub["actual_demand"], marker="o", label="Actual", color="tab:blue", linewidth=1.8)
        ax.plot(sub["year"], sub["predicted_ml"], marker="s", label="Predicted (ML)", color="tab:orange", linewidth=1.5)
        ax.plot(sub["year"], sub["predicted_baseline"], marker="d", label="Predicted (Baseline)", color="tab:green", linestyle=":", alpha=0.9, linewidth=1.25)

        y_min = sub[["actual_demand", "predicted_ml", "predicted_baseline"]].min().min()
        y_max = sub[["actual_demand", "predicted_ml", "predicted_baseline"]].max().max()
        pad = 0.06 * (y_max - y_min) if y_max > y_min else 1.0
        ax.set_ylim(y_min - pad, y_max + pad)

        ax.set_title(f"{d} — Total Demand (2010-2025)")
        ax.set_xlabel("Year")
        ax.set_ylabel("Units / year")
        ax.legend(loc="upper left")
        tidy_axes(ax)
        plt.tight_layout()
        if save_png:
            out_path = os.path.join(out_dir, f"total_demand_{d}.png")
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {out_path}")
        plt.show()


def main():
    base_dir = os.path.dirname(__file__) or "."
    data_path = os.path.join(base_dir, DATA_FN)
    df_raw = load_dataset(data_path)
    df = clean_and_feature_engineer(df_raw)

    results = train_and_evaluate(df, holdout_year_cut=2020, cv_splits=5, rnd_search_iters=40)
    hr = results["holdout_report"]
    print("\nFINAL HOLDOUT SUMMARY:")
    print(f"MAE: {hr['MAE']:.3f} units")
    print(f"RMSE: {hr['RMSE']:.3f} units")
    print(f"R2: {hr['R2']:.3f}")
    print(f"Percent within ±20 units: {hr['pct_within_20_units']:.1f}%")

    # residuals: align test_df and df_out predictions
    test_df = results["test_df"]
    df_out = results["df_out"]
    # test_df has original index in column 'index' because we reset_index earlier
    test_idx = test_df["index"].values
    y_test = test_df["actual_demand"].values
    y_test_pred = df_out.loc[df_out.index.isin(test_idx), "predicted_ml"].values
    plot_residual_diagnostics(y_test, y_test_pred, title_suffix="(holdout)")

    # per-district diagnostics & save
    district_stats = per_group_metrics(df_out, group_col="district", y_col="actual_demand", pred_col="predicted_ml", tolerance_units=20.0)
    print("\nPer-district metrics:")
    for d, s in sorted(district_stats.items()):
        print(f"{d}: MAE={s['MAE']:.2f}, RMSE={s['RMSE']:.2f}, R2={s['R2']:.3f}, pct_within_20={s['pct_within_20_units']:.1f}%")
    metrics_df = pd.DataFrame.from_dict(district_stats, orient="index")
    metrics_csv = os.path.join(base_dir, "per_district_metrics_timeholdout.csv")
    metrics_df.to_csv(metrics_csv)
    print(f"Per-district metrics saved to: {metrics_csv}")

    # time-series plots
    plot_district_overall_demand(df_out, save_png=False, out_dir=base_dir)


if __name__ == "__main__":
    main()
