"""Synthetic smoke test for TL (OLS) vs TL (Ridge).

This fabricates a small, noisy, highly collinear experiment where OLS is
intentionally unstable enough that ridge should produce different estimates.
It is not a statistical benchmark; it is a quick implementation check.
"""

import numpy as np
import pandas as pd

from helper import (
    ALL_WEEKS,
    FEATURES_DEFAULT,
    _RIDGE_LAMBDAS_CUSTOM,
    run_method_evaluation_with_ridge,
)


def make_synthetic_experiment(n: int = 90, seed: int = 123) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    latent = rng.normal(size=n)
    latent2 = rng.normal(size=n)
    treatment = rng.binomial(1, 0.5, size=n)

    df = pd.DataFrame(
        {
            "experiment_id": "synthetic_collinear",
            "user_id": np.arange(n),
            "W": treatment,
        }
    )

    # Make many feature columns noisy transforms of the same few latent drivers.
    # With small per-fold group samples, this gives OLS enough collinearity/noise
    # for ridge to visibly diverge.
    for i, feature in enumerate(FEATURES_DEFAULT):
        signal = (1.0 + 0.03 * i) * latent + (0.25 if i % 2 else -0.25) * latent2
        noise = rng.normal(scale=0.12 + 0.02 * (i % 4), size=n)
        values = signal + noise
        if "recency" in feature or "days_since" in feature:
            values = np.maximum(0, 30 + 10 * values)
        elif "txns" in feature or "searches" in feature or "clicks" in feature or "favorites" in feature:
            values = np.maximum(0, np.round(4 + 2.5 * values))
        df[feature] = values

    # Cumulative post outcomes. The outcome is driven by the same latent signal,
    # with heavy noise and a small treatment effect. Clipping creates count-like Y.
    for week in ALL_WEEKS:
        raw = (
            0.5 * week
            + 2.5 * latent
            + 0.8 * latent2
            + 0.12 * week * treatment
            + rng.normal(scale=4.0 + 0.08 * week, size=n)
        )
        df[f"Y_buyer_txns_post_{week}w"] = np.maximum(0, np.round(raw)).astype(float)

    return df


def main() -> None:
    df = make_synthetic_experiment()

    results_df, r2_df_ols, r2_df_ridge, best_feat_df = run_method_evaluation_with_ridge(
        df,
        features=FEATURES_DEFAULT,
        exp_order=["synthetic_collinear"],
        weeks=ALL_WEEKS,
    )

    methods = set(results_df["method"].unique())
    assert {"tl_ols", "tl_ridge"}.issubset(methods), methods

    ridge_w20 = results_df[(results_df["method"] == "tl_ridge") & (results_df["week"] == 20)].iloc[0]
    ols_w20 = results_df[(results_df["method"] == "tl_ols") & (results_df["week"] == 20)].iloc[0]

    lam0 = float(r2_df_ridge.loc[r2_df_ridge["week"] == 20, "lam0"].iloc[0])
    lam1 = float(r2_df_ridge.loc[r2_df_ridge["week"] == 20, "lam1"].iloc[0])
    assert lam0 in _RIDGE_LAMBDAS_CUSTOM, lam0
    assert lam1 in _RIDGE_LAMBDAS_CUSTOM, lam1

    diff = {
        "tau_abs_diff": abs(float(ridge_w20["tau"]) - float(ols_w20["tau"])),
        "se_abs_diff": abs(float(ridge_w20["se"]) - float(ols_w20["se"])),
        "ds_abs_diff": abs(float(ridge_w20["ds"]) - float(ols_w20["ds"])),
    }
    assert max(diff.values()) > 1e-6, diff

    print("Synthetic TL Ridge vs OLS smoke test passed.")
    print(f"Selected lambdas: control={lam0}, treatment={lam1}")
    print("Best CUPED feature:")
    print(best_feat_df[["experiment_id", "best_feature", "pearson"]].to_string(index=False))
    print("\nW20 comparison:")
    print(
        results_df[
            (results_df["week"] == 20)
            & (results_df["method"].isin(["naive", "tl_ols", "tl_ridge"]))
        ][["method", "tau", "se", "ds", "std_diff"]].to_string(index=False)
    )
    print("\nDifferences:")
    print(diff)
    print("\nR2 W20:")
    print("OLS")
    print(r2_df_ols[r2_df_ols["week"] == 20].to_string(index=False))
    print("Ridge")
    print(r2_df_ridge[r2_df_ridge["week"] == 20].to_string(index=False))


if __name__ == "__main__":
    main()
