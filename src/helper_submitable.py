# helper.py
# External review copy.
#
# Data-access logic, infrastructure details, credentials, storage locations,
# and non-public identifiers are intentionally omitted.
#
# The analytical routines below are preserved for code review. To run this code,
# a caller must provide a pandas DataFrame with the expected schema.

from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, ttest_ind
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold


FEATURES_DEFAULT: List[str] = [
    # Add feature names here 
]


ALL_WEEKS = list(range(1, 21))
OUTCOME_KIND = "outcome"  # Set to your metric's column name prefix
# Experiment identifiers are omitted from this external copy.
# Use anonymized identifiers if adapting the code.
SELECTED_EXPERIMENTS_DEFAULT: List[str] = []
METHOD_ORDER = ["naive", "winsor", "cuped", "tl_ols", "tl_ridge"]
METHOD_LABELS = {
    "naive": "Be metodo",
    "winsor": "Variacinės eilutės karpymas",
    "cuped": "CUPED",
    "tl_ols": "TL (OLS)",
    "tl_ridge": "TL (Ridge)",
}
METHOD_COLORS = {
    "naive": "#7f7f7f",
    "winsor": "#2ca25f",
    "cuped": "#3182bd",
    "tl_ols": "#f28e2b",
    "tl_ridge": "#9467bd",
}

_RANDOM_SEED = 42
_RIDGE_LAMBDAS_CUSTOM: List[float] = [0.1, 1.0, 2.5, 10.0, 20.0, 50.0]


def load_analysis_data(*args, **kwargs) -> pd.DataFrame:
    """Placeholder for user-provided input data.

    This external copy does not include data-access logic, infrastructure
    details, credentials, storage locations, or non-public identifiers.

    Expected input schema:
      - one row per analysis unit
      - an experiment identifier column, e.g. "experiment_id"
      - a treatment indicator column, e.g. "W"
      - a user identifier column, e.g. "user_id"
      - pre-period feature columns
      - post-period outcome columns using the expected naming convention

    The caller should provide a pandas DataFrame with the required columns.
    """
    raise NotImplementedError(
        "Data loading is intentionally omitted from this external copy. "
        "Provide a pandas DataFrame from an appropriate source before running the analysis."
    )


def add_horizon_outcomes_by_experiment(
    df: pd.DataFrame,
    horizons_weeks: Iterable[int],
    experiment_col: str = "experiment_id",
    copy: bool = True,
    fill_missing_weeks_as_zero: bool = True,
) -> pd.DataFrame:
    out = df.copy() if copy else df
    for exp_id, idx in out.groupby(experiment_col).groups.items():
        exp_df = out.loc[idx]
        for n in horizons_weeks:
            txn_cols = [f"{OUTCOME_KIND}_post_week_{i}" for i in range(1, n + 1)]
            missing_cols = [c for c in txn_cols if c not in out.columns]
            if missing_cols:
                raise ValueError(
                    f"Experiment {exp_id}: missing columns for {n}w horizon: {missing_cols}"
                )
            y_col = f"Y_{OUTCOME_KIND}_post_{n}w"
            if fill_missing_weeks_as_zero:
                out.loc[idx, y_col] = exp_df[txn_cols].fillna(0).sum(axis=1)
            else:
                out.loc[idx, y_col] = exp_df[txn_cols].sum(axis=1, min_count=len(txn_cols))
    return out


def build_experiment_summary(
    df: pd.DataFrame,
    experiment_col: str = "experiment_id",
    user_col: str = "user_id",
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    summary = (
        df.groupby(experiment_col)[user_col]
        .nunique()
        .rename("n_e")
        .reset_index()
        .sort_values("n_e", ascending=False)
        .reset_index(drop=True)
    )
    summary["exp_label"] = [f"e_{i + 1}" for i in range(len(summary))]
    label_map = dict(zip(summary[experiment_col], summary["exp_label"]))
    return summary[["exp_label", experiment_col, "n_e"]], label_map


def build_experiment_data_summary(
    df: pd.DataFrame,
    exp_summary_df: pd.DataFrame,
    experiment_col: str = "experiment_id",
    treatment_col: str = "W",
    user_col: str = "user_id",
    outcome_week: int = 20,
) -> pd.DataFrame:
    """Per-experiment data summary for quick diagnostics."""
    y_col = f"Y_{OUTCOME_KIND}_post_{outcome_week}w"
    rows = []
    for exp_id in exp_summary_df[experiment_col]:
        d = df[df[experiment_col] == exp_id].copy()
        y = pd.to_numeric(d[y_col], errors="coerce").fillna(0)
        n_ctrl = int((d[treatment_col] == 0).sum())
        n_trt = int((d[treatment_col] == 1).sum())
        rows.append(
            {
                "experiment_id": exp_id,
                "n_rows": int(len(d)),
                "n_users": int(d[user_col].nunique()),
                "n_control": n_ctrl,
                "n_treated": n_trt,
                "testinės_grupės_dalis": n_trt / max(n_ctrl + n_trt, 1),
                "y20_mean": float(y.mean()),
                "y20_median": float(y.median()),
                "y20_p99": float(np.percentile(y, 99)),
            }
        )
    return pd.DataFrame(rows)


def plot_users_by_experiment_stacked(
    df: pd.DataFrame,
    label_map: Dict[str, str],
    hide_experiment_names: bool = True,
    experiment_col: str = "experiment_id",
    treatment_col: str = "W",
    user_col: str = "user_id",
    figsize: Tuple[int, int] = (10, 5),
) -> pd.DataFrame:
    counts = (
        df.groupby([experiment_col, treatment_col])[user_col]
        .nunique()
        .unstack(fill_value=0)
        .sort_index()
    ).rename(columns={0: "Kontrolinė gr.", 1: "Eksperimentinė gr."})

    ax = counts.plot(
        kind="bar",
        stacked=True,
        figsize=figsize,
        color=["#1f77b4", "#ff7f0e"],
        edgecolor="white",
    )
    if hide_experiment_names:
        tick_labels = [label_map.get(x, "experiment") for x in counts.index]
        ax.set_xlabel("Eksperimentas")
    else:
        tick_labels = [str(x) for x in counts.index]
        ax.set_xlabel("Experiment ID")
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.set_title("Vartotojų skaičius pagal eksperimentą ir grupę")
    ax.set_ylabel("Vartotojų skaičius")
    ax.legend(title="Grupė")
    ax.grid(axis="y", alpha=0.3)
    for container in ax.containers:
        labels = [f"{v / 1000:.1f}k" if v > 0 else "" for v in container.datavalues]
        ax.bar_label(container, labels=labels, label_type="center", fontsize=9, color="white")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
    plt.tight_layout()
    plt.show()
    return counts


def _balance_one_experiment(
    df_exp: pd.DataFrame,
    features: List[str],
    treatment_col: str = "W",
    alpha: float = 0.05,
) -> pd.DataFrame:
    ctrl = df_exp[df_exp[treatment_col] == 0]
    trt = df_exp[df_exp[treatment_col] == 1]
    records = []
    for i, feat in enumerate(features, start=1):
        c = pd.to_numeric(ctrl[feat], errors="coerce").dropna()
        t = pd.to_numeric(trt[feat], errors="coerce").dropna()
        if len(c) < 2 or len(t) < 2:
            records.append(
                {
                    "feature": feat,
                    "feature_public": str(i),
                    "smd": np.nan,
                    "p_value": np.nan,
                    "sig": False,
                }
            )
            continue
        stat, pval = ttest_ind(c, t, equal_var=False)
        std_pool = np.sqrt((c.var(ddof=1) + t.var(ddof=1)) / 2)
        smd = (t.mean() - c.mean()) / std_pool if std_pool > 0 else np.nan
        records.append(
            {
                "feature": feat,
                "feature_public": str(i),
                "smd": float(smd),
                "p_value": float(pval),
                "sig": bool(pval < alpha),
            }
        )
    return pd.DataFrame(records)


def plot_figure1_distribution_grid(
    df: pd.DataFrame,
    exp_order: List[str],
    label_map: Dict[str, str],
    weeks: Tuple[int, int, int] = (1, 10, 20),
    experiment_col: str = "experiment_id",
    bins: int = 40,
) -> None:
    n_exp = len(exp_order)
    fig, axes = plt.subplots(n_exp, len(weeks), figsize=(13, 3.2 * n_exp), sharex="col")
    if n_exp == 1:
        axes = np.array([axes])
    _mean_line_label_added = False
    for r, exp_id in enumerate(exp_order):
        d = df[df[experiment_col] == exp_id].copy()
        row_vals = []
        for w in weeks:
            y = pd.to_numeric(d[f"Y_{OUTCOME_KIND}_post_{w}w"], errors="coerce").dropna()
            row_vals.append(y.values)
        row_all = np.concatenate([x for x in row_vals if len(x) > 0]) if row_vals else np.array([0.0])
        x_min, x_max = np.nanpercentile(row_all, [0.5, 99.5])
        for c, w in enumerate(weeks):
            ax = axes[r, c]
            y = row_vals[c]
            if len(y) == 0:
                ax.axis("off")
                continue
            x_grid = np.linspace(x_min, x_max, 250)
            try:
                kde = gaussian_kde(y)
                density = kde(x_grid)
            except Exception:
                density, edges = np.histogram(y, bins=bins, density=True)
                centers = 0.5 * (edges[1:] + edges[:-1])
                density = np.interp(x_grid, centers, density, left=0.0, right=0.0)
            ax.plot(x_grid, density, color="#4c78a8", linewidth=1.8)
            ax.fill_between(x_grid, density, color="#4c78a8", alpha=0.2)
            _mean_lbl = "Vidurkis" if not _mean_line_label_added else "_nolegend_"
            ax.axvline(
                float(y.mean()),
                color="#ff0000",
                linestyle=":",
                linewidth=2.2,
                label=_mean_lbl,
            )
            _mean_line_label_added = True
            ax.set_xlim(x_min, x_max)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"Sav. = {int(w)}", fontsize=13)
            if c == 0:
                ax.set_ylabel(label_map.get(exp_id, "experiment"), fontsize=13)
    fig.suptitle("$Y^{\\mathrm{post}}$ skirstinių kaita laike", y=1.01, fontsize=15)
    fig.legend(
        ["Vidurkis"],
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        fontsize=11,
        framealpha=0.8,
    )
    plt.tight_layout()
    plt.show()


def compute_best_feature_correlations(
    df: pd.DataFrame,
    features: List[str],
    exp_order: List[str],
    target_week: int = 20,
    experiment_col: str = "experiment_id",
) -> pd.DataFrame:
    out = []
    for exp_id in exp_order:
        d = df[df[experiment_col] == exp_id].copy()
        y = pd.to_numeric(d[f"Y_{OUTCOME_KIND}_post_{target_week}w"], errors="coerce")
        best = None
        for feat in features:
            x = pd.to_numeric(d[feat], errors="coerce")
            valid = x.notna() & y.notna()
            if valid.sum() < 3:
                continue
            pearson = x[valid].corr(y[valid], method="pearson")
            spearman = x[valid].corr(y[valid], method="spearman")
            score = abs(float(pearson))
            if best is None or score > best["abs_pearson"]:
                best = {
                    "experiment_id": exp_id,
                    "best_feature": feat,
                    "pearson": float(pearson),
                    "spearman": float(spearman),
                    "abs_pearson": score,
                }
        if best is not None:
            out.append(best)
    return pd.DataFrame(out)

def compute_best_feature_correlations_spearman(
    df: pd.DataFrame,
    features: List[str],
    exp_order: List[str],
    target_week: int = 20,
    experiment_col: str = "experiment_id",
) -> pd.DataFrame:
    out = []
    for exp_id in exp_order:
        d = df[df[experiment_col] == exp_id].copy()
        y = pd.to_numeric(d[f"Y_{OUTCOME_KIND}_post_{target_week}w"], errors="coerce")
        best = None
        for feat in features:
            x = pd.to_numeric(d[feat], errors="coerce")
            valid = x.notna() & y.notna()
            if valid.sum() < 3:
                continue
            pearson = x[valid].corr(y[valid], method="pearson")
            spearman = x[valid].corr(y[valid], method="spearman")
            score = abs(float(spearman))
            if best is None or score > best["abs_spearman"]:
                best = {
                    "experiment_id": exp_id,
                    "best_feature": feat,
                    "pearson": float(pearson),
                    "spearman": float(spearman),
                    "abs_spearman": score,
                }
        if best is not None:
            out.append(best)
    return pd.DataFrame(out)

def plot_figure2_correlations(
    corr_best_df: pd.DataFrame,
    label_map: Dict[str, str],
) -> None:
    d = corr_best_df.copy()
    d["exp_label"] = d["experiment_id"].map(label_map)
    d = d.sort_values("exp_label")
    x = np.arange(len(d))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - w / 2, d["pearson"], width=w, label="Pearson", color="#4c78a8")
    ax.bar(x + w / 2, d["spearman"], width=w, label="Spearman", color="#f58518")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(d["exp_label"])
    ax.set_ylim(-1, 1)
    ax.set_ylabel("Koreliacijos koeficientas")
    ax.set_title("$X^{\\mathrm{pre}}$ ir $Y^{\\mathrm{post}}$ (20 savaitė) ryšys")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def run_balance_analysis(
    df: pd.DataFrame,
    features: List[str],
    exp_order: List[str],
    label_map: Dict[str, str],
    experiment_col: str = "experiment_id",
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    balance_by_exp = {}
    summary_rows = []
    for exp_id in exp_order:
        d = df[df[experiment_col] == exp_id].copy()
        b = _balance_one_experiment(d, features)
        balance_by_exp[exp_id] = b
        summary_rows.append(
            {
                "Eksperimentas": label_map.get(exp_id, "experiment"),
                "max |SMD|": float(np.nanmax(np.abs(b["smd"]))),
                "|SMD| > 0.1": int((np.abs(b["smd"]) > 0.1).sum()),
                "p < 0.05": int((b["p_value"] < 0.05).sum()),
            }
        )
    return balance_by_exp, pd.DataFrame(summary_rows)


def plot_figure3_balance_grid(
    balance_by_exp: Dict[str, pd.DataFrame],
    exp_order: List[str],
    label_map: Dict[str, str],
) -> None:
    n_exp = len(exp_order)
    fig, axes = plt.subplots(n_exp, 1, figsize=(12, 4.2 * n_exp), sharex=True)
    if n_exp == 1:
        axes = [axes]
    for i, exp_id in enumerate(exp_order):
        ax = axes[i]
        b = balance_by_exp[exp_id].dropna(subset=["smd"]).copy()
        y = np.arange(len(b))
        colors = np.where(b["p_value"] < 0.05, "#d62728", "#4c78a8")
        ax.barh(y, b["smd"], color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.axvline(0.1, color="gray", linestyle="--", linewidth=0.9)
        ax.axvline(-0.1, color="gray", linestyle="--", linewidth=0.9)
        ax.set_yticks(y)
        ax.set_yticklabels(b["feature_public"], fontsize=7)
        ax.set_title(label_map.get(exp_id, "experiment"))
        ax.grid(axis="x", alpha=0.3)
    axes[-1].set_xlabel("SMD")
    fig.suptitle("Grupių pusiausvyra pagal eksperimentą", y=1.0)
    plt.tight_layout()
    plt.show()





def _prep_X(df_exp: pd.DataFrame, features: List[str], fill_value: float = 0.0) -> pd.DataFrame:
    X = df_exp[list(features)].copy()
    X = X.fillna(fill_value).apply(pd.to_numeric, errors="coerce").fillna(fill_value)
    return (X)


def _diff_in_means_stats(Y: np.ndarray, W: np.ndarray, z: float = 1.96) -> Dict[str, float]:
    y1 = Y[W == 1]
    y0 = Y[W == 0]
    ate = y1.mean() - y0.mean()
    se = np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0))
    return {
        "tau": float(ate),
        "se": float(se),
        "ci_low": float(ate - z * se),
        "ci_high": float(ate + z * se),
        "pi_width": float(2 * z * se),
    }


_RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]


def _make_tlearner_arm_model(tl_model: str = "ols") -> object:
    """Return a fresh unfitted arm model. tl_model: 'ols' or 'ridge'."""
    if tl_model == "ridge":
        return RidgeCV(alphas=_RIDGE_ALPHAS, fit_intercept=True)
    return LinearRegression()


def _run_tlearner_week_ols(
    X_full: pd.DataFrame,
    Y: np.ndarray,
    W: np.ndarray,
    n_splits: int = 5,
    random_state: int = _RANDOM_SEED,
    z: float = 1.96,
    tl_model: str = "ols",
) -> Dict[str, float]:
    n = len(Y)
    mu0_hat = np.zeros(n)
    mu1_hat = np.zeros(n)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for train_idx, test_idx in kf.split(X_full):
        X_train, X_test = X_full.iloc[train_idx], X_full.iloc[test_idx]
        Y_train, W_train = Y[train_idx], W[train_idx]
        m0 = _make_tlearner_arm_model(tl_model)
        m1 = _make_tlearner_arm_model(tl_model)
        m0.fit(X_train[W_train == 0], Y_train[W_train == 0])
        m1.fit(X_train[W_train == 1], Y_train[W_train == 1])
        mu0_hat[test_idx] = m0.predict(X_test)
        mu1_hat[test_idx] = m1.predict(X_test)
    p = W.mean()
    scores = mu1_hat - mu0_hat + W / p * (Y - mu1_hat) - (1 - W) / (1 - p) * (Y - mu0_hat)
    tau = float(scores.mean())
    se = float(scores.std(ddof=1) / np.sqrt(n))
    r2_control = float(r2_score(Y[W == 0], mu0_hat[W == 0])) if (W == 0).sum() >= 2 else np.nan
    r2_treated = float(r2_score(Y[W == 1], mu1_hat[W == 1])) if (W == 1).sum() >= 2 else np.nan
    return {
        "tau": tau,
        "se": se,
        "ci_low": float(tau - z * se),
        "ci_high": float(tau + z * se),
        "pi_width": float(2 * z * se),
        "r2_control": r2_control,
        "r2_treated": r2_treated,
    }


def _std_fit_transform(X_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X_scaled, mean, std) computed solely from training data."""
    mu = X_train.mean(axis=0)
    std = X_train.std(axis=0, ddof=1)
    std[std == 0.0] = 1.0
    return (X_train - mu) / std, mu, std


def _std_apply(X: np.ndarray, mu: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mu) / std


def _ridge_fit(X_s: np.ndarray, Y: np.ndarray, lam: float) -> np.ndarray:
    """Ridge with intercept column prepended. Y may be (n,) or (n, k).
    Solves (X'X + lam*I) coef = X'Y using np.linalg.solve — never np.linalg.inv.
    """
    Xa = np.hstack([np.ones((len(X_s), 1)), X_s])
    p = Xa.shape[1]
    A = Xa.T @ Xa + lam * np.eye(p)
    b = Xa.T @ Y
    return np.linalg.solve(A, b)


def _ridge_predict(X_s: np.ndarray, coef: np.ndarray) -> np.ndarray:
    Xa = np.hstack([np.ones((len(X_s), 1)), X_s])
    return Xa @ coef


def _select_lambda_ridge(
    X: np.ndarray,
    Y_mat: np.ndarray,
    mask: np.ndarray,
    n_splits: int = 5,
    random_state: int = _RANDOM_SEED,
) -> float:
    """Select the best lambda for one treatment group via OOF R² averaged across
    all metric columns (weeks).  X.T @ X + lam*I is computed once per (fold, lam)
    because _ridge_fit solves for all k columns of Y simultaneously."""
    X_g = X[mask]
    Y_g = Y_mat[mask]  # (n_g, k)
    k = Y_g.shape[1]
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    best_lam, best_r2 = _RIDGE_LAMBDAS_CUSTOM[0], -np.inf
    for lam in _RIDGE_LAMBDAS_CUSTOM:
        y_oof = np.zeros_like(Y_g, dtype=float)
        for tr_idx, te_idx in kf.split(X_g):
            X_tr_s, mu_tr, std_tr = _std_fit_transform(X_g[tr_idx])
            X_te_s = _std_apply(X_g[te_idx], mu_tr, std_tr)
            coef = _ridge_fit(X_tr_s, Y_g[tr_idx], lam)  # solves all k at once
            y_oof[te_idx] = _ridge_predict(X_te_s, coef)

        # Average R² across metric columns
        r2_sum = 0.0
        for j in range(k):
            ss_res = np.sum((Y_g[:, j] - y_oof[:, j]) ** 2)
            ss_tot = np.sum((Y_g[:, j] - Y_g[:, j].mean()) ** 2)
            r2_sum += 1.0 - ss_res / max(ss_tot, 1e-12)
        avg_r2 = r2_sum / k
        if avg_r2 > best_r2:
            best_r2 = avg_r2
            best_lam = lam

    return best_lam


def _run_tlearner_week_ridge_v2(
    X_full: np.ndarray,
    Y: np.ndarray,
    W: np.ndarray,
    lam0: float,
    lam1: float,
    n_splits: int = 5,
    random_state: int = _RANDOM_SEED,
    z: float = 1.96,
) -> Dict[str, float]:
    """AIPW TL with custom ridge estimator.
    Standardization parameters are computed only on each fold's training group data
    and then applied to the full test fold — never on the full dataset.
    Pre-selected per-group lambdas (lam0 for W=0, lam1 for W=1) are passed in.
    """
    n = len(Y)
    mu0_hat = np.zeros(n)
    mu1_hat = np.zeros(n)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for tr_idx, te_idx in kf.split(X_full):
        X_tr, X_te = X_full[tr_idx], X_full[te_idx]
        Y_tr, W_tr = Y[tr_idx], W[tr_idx]

        mask0 = W_tr == 0
        X_tr0_s, mu0, std0 = _std_fit_transform(X_tr[mask0])
        coef0 = _ridge_fit(X_tr0_s, Y_tr[mask0], lam0)
        mu0_hat[te_idx] = _ridge_predict(_std_apply(X_te, mu0, std0), coef0)

        mask1 = W_tr == 1
        X_tr1_s, mu1, std1 = _std_fit_transform(X_tr[mask1])
        coef1 = _ridge_fit(X_tr1_s, Y_tr[mask1], lam1)
        mu1_hat[te_idx] = _ridge_predict(_std_apply(X_te, mu1, std1), coef1)

    p = W.mean()
    scores = (
        mu1_hat - mu0_hat
        + W / p * (Y - mu1_hat)
        - (1 - W) / (1 - p) * (Y - mu0_hat)
    )
    tau = float(scores.mean())
    se = float(scores.std(ddof=1) / np.sqrt(n))
    r2_ctrl = float(r2_score(Y[W == 0], mu0_hat[W == 0])) if (W == 0).sum() >= 2 else np.nan
    r2_trt = float(r2_score(Y[W == 1], mu1_hat[W == 1])) if (W == 1).sum() >= 2 else np.nan
    return {
        "tau": tau,
        "se": se,
        "ci_low": float(tau - z * se),
        "ci_high": float(tau + z * se),
        "pi_width": float(2 * z * se),
        "r2_control": r2_ctrl,
        "r2_treated": r2_trt,
        "lam0": lam0,
        "lam1": lam1,
    }


def _winsorize_upper_by_group(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Winsorize only upper tail per treatment group at mean + 5 * SD."""
    y = np.asarray(y, dtype=float).copy()
    w = np.asarray(w, dtype=int)
    out = y.copy()
    for arm in [0, 1]:
        mask = w == arm
        if mask.sum() == 0:
            continue
        arm_y = y[mask]
        hi = arm_y.mean() + 5.0 * arm_y.std(ddof=1)
        out[mask] = np.where(arm_y > hi, hi, arm_y)
    return out


def _cuped_adjust(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    var_x = np.var(x, ddof=1)
    if var_x <= 0:
        return y.copy()
    theta = np.cov(y, x, ddof=1)[0, 1] / var_x
    return y - theta * (x - np.mean(x))


def run_method_evaluation(
    df: pd.DataFrame,
    features: List[str],
    exp_order: List[str],
    weeks: List[int] = ALL_WEEKS,
    experiment_col: str = "experiment_id",
    treatment_col: str = "W",
    tl_model: str = "ols",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    tl_model: 'ols' (default, plain LinearRegression) or 'ridge' (RidgeCV with cross-validated alpha).
    Passing 'ols' produces identical results to prior behaviour — no changes needed in existing callers.
    """
    rows = []
    r2_rows = []
    best_feat_df = compute_best_feature_correlations(df, features, exp_order, target_week=20)
    best_feat_map = dict(zip(best_feat_df["experiment_id"], best_feat_df["best_feature"]))
    for exp_id in exp_order:
        d = df[df[experiment_col] == exp_id].copy()
        X_full = _prep_X(d, features)
        W = d[treatment_col].astype(int).to_numpy()
        cuped_feat = best_feat_map.get(exp_id, features[0])
        x_cuped = pd.to_numeric(d[cuped_feat], errors="coerce").fillna(0).to_numpy()
        for w in weeks:
            y_col = f"Y_{OUTCOME_KIND}_post_{w}w"
            Y = pd.to_numeric(d[y_col], errors="coerce").fillna(0).astype(float).to_numpy()

            naive = _diff_in_means_stats(Y, W)
            rows.append(
                {
                    "experiment_id": exp_id,
                    "week": w,
                    "method": "naive",
                    **naive,
                }
            )

            yw = _winsorize_upper_by_group(Y, W)
            wins = _diff_in_means_stats(yw, W)
            rows.append(
                {
                    "experiment_id": exp_id,
                    "week": w,
                    "method": "winsor",
                    **wins,
                }
            )

            y_cuped = _cuped_adjust(Y, x_cuped)
            cuped = _diff_in_means_stats(y_cuped, W)
            rows.append(
                {
                    "experiment_id": exp_id,
                    "week": w,
                    "method": "cuped",
                    "cuped_feature": cuped_feat,
                    **cuped,
                }
            )

            tl = _run_tlearner_week_ols(X_full, Y, W, tl_model=tl_model)
            rows.append(
                {
                    "experiment_id": exp_id,
                    "week": w,
                    "method": "tl_ols",
                    **{k: tl[k] for k in ["tau", "se", "ci_low", "ci_high", "pi_width"]},
                }
            )
            r2_rows.append(
                {
                    "experiment_id": exp_id,
                    "week": w,
                    "r2_control": tl["r2_control"],
                    "r2_treated": tl["r2_treated"],
                }
            )

    res = pd.DataFrame(rows)
    r2_df = pd.DataFrame(r2_rows)
    base = (
        res[res["method"] == "naive"][["experiment_id", "week", "se"]]
        .rename(columns={"se": "se_naive"})
        .copy()
    )
    tau_base = (
        res[res["method"] == "naive"][["experiment_id", "week", "tau"]]
        .rename(columns={"tau": "tau_naive"})
        .copy()
    )
    res = res.merge(base, on=["experiment_id", "week"], how="left")
    res = res.merge(tau_base, on=["experiment_id", "week"], how="left")
    res["ds"] = 1.0 - (res["se"] ** 2) / (res["se_naive"] ** 2)
    res["std_diff"] = (res["tau"] - res["tau_naive"]) / res["se_naive"].replace(0, np.nan)
    return res, r2_df, best_feat_df


def run_method_evaluation_with_ridge(
    df: pd.DataFrame,
    features: List[str],
    exp_order: List[str],
    weeks: List[int] = ALL_WEEKS,
    experiment_col: str = "experiment_id",
    treatment_col: str = "W",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run all four methods (naive, winsor, cuped, tl_ols) PLUS tl_ridge.

    Lambda selection for ridge is done once per experiment using all weeks as a
    joint (n, k) target matrix, so X.T @ X + lam*I is computed at most once per
    (group, fold, lambda) across all k=len(weeks) outcomes.

    Returns
    -------
    results_df   : rows for all five methods, same columns as run_method_evaluation
    r2_df_ols    : R² per week per experiment for tl_ols
    r2_df_ridge  : R² per week per experiment for tl_ridge (+ lam0, lam1 columns)
    best_feat_df : best CUPED feature per experiment
    """
    rows: List[dict] = []
    r2_rows_ols: List[dict] = []
    r2_rows_ridge: List[dict] = []

    best_feat_df = compute_best_feature_correlations(df, features, exp_order, target_week=20)
    best_feat_map = dict(zip(best_feat_df["experiment_id"], best_feat_df["best_feature"]))

    for exp_id in exp_order:
        d = df[df[experiment_col] == exp_id].copy()
        X_full_df = _prep_X(d, features)
        X_full = X_full_df.to_numpy().astype(float)
        W = d[treatment_col].astype(int).to_numpy()
        cuped_feat = best_feat_map.get(exp_id, features[0])
        x_cuped = pd.to_numeric(d[cuped_feat], errors="coerce").fillna(0).to_numpy()

        # Build Y matrix for all weeks — used for ridge lambda selection
        Y_mat = np.column_stack([
            pd.to_numeric(d[f"Y_{OUTCOME_KIND}_post_{w}w"], errors="coerce").fillna(0).astype(float).to_numpy()
            for w in weeks
        ])  # shape (n, len(weeks))

        # Select lambda per group jointly across all weeks
        lam0 = _select_lambda_ridge(X_full, Y_mat, W == 0)
        lam1 = _select_lambda_ridge(X_full, Y_mat, W == 1)

        for wi, w in enumerate(weeks):
            Y = Y_mat[:, wi]

            naive = _diff_in_means_stats(Y, W)
            rows.append({"experiment_id": exp_id, "week": w, "method": "naive", **naive})

            yw = _winsorize_upper_by_group(Y, W)
            wins = _diff_in_means_stats(yw, W)
            rows.append({"experiment_id": exp_id, "week": w, "method": "winsor", **wins})

            y_cuped = _cuped_adjust(Y, x_cuped)
            cuped = _diff_in_means_stats(y_cuped, W)
            rows.append({
                "experiment_id": exp_id,
                "week": w,
                "method": "cuped",
                "cuped_feature": cuped_feat,
                **cuped,
            })

            tl = _run_tlearner_week_ols(X_full_df, Y, W)
            rows.append({
                "experiment_id": exp_id,
                "week": w,
                "method": "tl_ols",
                **{k: tl[k] for k in ["tau", "se", "ci_low", "ci_high", "pi_width"]},
            })
            r2_rows_ols.append({
                "experiment_id": exp_id,
                "week": w,
                "r2_control": tl["r2_control"],
                "r2_treated": tl["r2_treated"],
            })

            tl_r = _run_tlearner_week_ridge_v2(X_full, Y, W, lam0, lam1)
            rows.append({
                "experiment_id": exp_id,
                "week": w,
                "method": "tl_ridge",
                **{k: tl_r[k] for k in ["tau", "se", "ci_low", "ci_high", "pi_width"]},
            })
            r2_rows_ridge.append({
                "experiment_id": exp_id,
                "week": w,
                "r2_control": tl_r["r2_control"],
                "r2_treated": tl_r["r2_treated"],
                "lam0": tl_r["lam0"],
                "lam1": tl_r["lam1"],
            })

    res = pd.DataFrame(rows)
    r2_df_ols = pd.DataFrame(r2_rows_ols)
    r2_df_ridge = pd.DataFrame(r2_rows_ridge)

    base = (
        res[res["method"] == "naive"][["experiment_id", "week", "se"]]
        .rename(columns={"se": "se_naive"})
        .copy()
    )
    tau_base = (
        res[res["method"] == "naive"][["experiment_id", "week", "tau"]]
        .rename(columns={"tau": "tau_naive"})
        .copy()
    )
    res = res.merge(base, on=["experiment_id", "week"], how="left")
    res = res.merge(tau_base, on=["experiment_id", "week"], how="left")
    res["ds"] = 1.0 - (res["se"] ** 2) / (res["se_naive"] ** 2)
    res["std_diff"] = (res["tau"] - res["tau_naive"]) / res["se_naive"].replace(0, np.nan)

    return res, r2_df_ols, r2_df_ridge, best_feat_df


def build_table_1(results_df: pd.DataFrame, exp_summary_df: pd.DataFrame, label_map: Dict[str, str]) -> pd.DataFrame:
    d = results_df[results_df["week"] == 20].copy()
    d = d[d["method"].isin(["winsor", "cuped", "tl_ols"])]
    pvt = d.pivot_table(index="experiment_id", columns="method", values="ds", aggfunc="first")
    pvt = pvt.rename(
        columns={
            "winsor": "DS_winsor_%",
            "cuped": "DS_cuped_%",
            "tl_ols": "DS_tl_%",
        }
    )
    pvt = pvt * 100
    out = (
        exp_summary_df.rename(columns={"experiment_id": "experiment_id"})
        .set_index("experiment_id")
        .join(pvt)
        .reset_index()
    )
    out["Eksperimentas"] = out["experiment_id"].map(label_map)
    out = out[["Eksperimentas", "n_e", "DS_winsor_%", "DS_cuped_%", "DS_tl_%"]]
    avg = pd.DataFrame(
        [
            {
                "Eksperimentas": "Vidurkis",
                "n_e": out["n_e"].mean(),
                "DS_winsor_%": out["DS_winsor_%"].mean(),
                "DS_cuped_%": out["DS_cuped_%"].mean(),
                "DS_tl_%": out["DS_tl_%"].mean(),
            }
        ]
    )
    return pd.concat([out, avg], ignore_index=True)


def plot_figure4(results_df: pd.DataFrame, exp_summary_df: pd.DataFrame, label_map: Dict[str, str]) -> None:
    d = results_df[(results_df["week"] == 20) & (results_df["method"] != "naive")].copy()
    d["method_label"] = d["method"].map(METHOD_LABELS)
    n_map = dict(zip(exp_summary_df["experiment_id"], exp_summary_df["n_e"]))
    d["n_e"] = d["experiment_id"].map(n_map)

    fig, ax = plt.subplots(figsize=(9, 5))
    methods = ["winsor", "cuped", "tl_ols"]
    x = np.arange(len(methods))
    means = [100 * d.loc[d["method"] == m, "ds"].mean() for m in methods]
    ax.bar(
        x,
        means,
        width=0.6,
        color=[METHOD_COLORS[m] for m in methods],
        edgecolor="black",
        alpha=0.7,
    )
    for i, m in enumerate(methods):
        sub = d[d["method"] == m]
        rng = np.random.default_rng(_RANDOM_SEED + i)
        jitter = rng.uniform(-0.12, 0.12, size=len(sub))
        sc = ax.scatter(
            np.full(len(sub), i) + jitter,
            100 * sub["ds"],
            c=sub["n_e"],
            cmap="viridis",
            s=60,
            edgecolor="black",
            linewidth=0.4,
        )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("n_e")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in methods])
    ax.set_ylabel("Dispersijos sumažinimas (%)")
    ax.set_title("Dispersijos sumažinimas 20 savaitę")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def _fit_line_with_band(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    coef = np.polyfit(x, y, 1)
    xg = np.linspace(x.min(), x.max(), 100)
    yg = coef[0] * xg + coef[1]
    yhat = coef[0] * x + coef[1]
    resid = y - yhat
    s2 = np.sum(resid**2) / max(len(x) - 2, 1)
    xbar = np.mean(x)
    sxx = np.sum((x - xbar) ** 2)
    se_pred = np.sqrt(s2 * (1 / len(x) + (xg - xbar) ** 2 / max(sxx, 1e-12)))
    lo = yg - 1.96 * se_pred
    hi = yg + 1.96 * se_pred
    return xg, lo, hi, yg


def plot_figure5(results_df: pd.DataFrame, exp_summary_df: pd.DataFrame) -> None:
    d = results_df[(results_df["week"] == 20) & (results_df["method"].isin(["winsor", "cuped", "tl_ols"]))].copy()
    d = d.merge(exp_summary_df[["experiment_id", "n_e"]], on="experiment_id", how="left")
    fig, ax = plt.subplots(figsize=(9, 5))
    for m in ["winsor", "cuped", "tl_ols"]:
        sub = d[d["method"] == m]
        x = sub["n_e"].astype(float).to_numpy()
        y = (100 * sub["ds"]).astype(float).to_numpy()
        ax.scatter(x, y, color=METHOD_COLORS[m], label=METHOD_LABELS[m], alpha=0.9)
        if len(sub) >= 2:
            lx = np.log10(x)
            xg, _lo, _hi, yg = _fit_line_with_band(lx, y)
            ax.plot(10 ** xg, yg, color=METHOD_COLORS[m], linewidth=1.7)
    ax.set_xscale("log")
    ax.set_xlabel("$n_e$")
    ax.set_ylabel("Dispersijos sumažinimas (%)")
    ax.set_title("Dispersijos sumažinimas pagal eksperimento dydį (20 savaitė)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_figure6(results_df: pd.DataFrame, exp_order: List[str], label_map: Dict[str, str], exp_summary_df: pd.DataFrame) -> None:
    methods = METHOD_ORDER
    n = len(exp_order)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.3 * nrows), sharex=True, sharey=True)
    axes = np.array(axes).reshape(-1)

    ymin = results_df["tau"].min()
    ymax = results_df["tau"].max()
    y_pad = 0.05 * max(ymax - ymin, 1e-9)
    n_map = dict(zip(exp_summary_df["experiment_id"], exp_summary_df["n_e"]))

    for i, exp_id in enumerate(exp_order):
        ax = axes[i]
        sube = results_df[results_df["experiment_id"] == exp_id]
        for m in methods:
            s = sube[sube["method"] == m].sort_values("week")
            ax.plot(
                s["week"],
                s["tau"],
                color=METHOD_COLORS[m],
                label=METHOD_LABELS[m],
                linewidth=1.6,
                marker="o",
                markersize=3.5,
            )
        ax.axhline(0, linestyle="--", color="gray", alpha=0.6)
        ax.set_ylim(ymin - y_pad, ymax + y_pad)
        ax.set_xlim(min(ALL_WEEKS), max(ALL_WEEKS))
        ax.set_xticks(ALL_WEEKS)
        ax.set_yticklabels([])
        ax.set_title(f"{label_map.get(exp_id, "experiment")}, n_e={n_map.get(exp_id)}")
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    fig.legend([METHOD_LABELS[m] for m in methods], loc="upper right", ncol=1)
    fig.suptitle("Poveikio įvertis (ATE) pagal savaites")
    plt.tight_layout()
    plt.show()


def plot_figure6_absolute_error(
    results_df: pd.DataFrame,
    exp_order: List[str],
    label_map: Dict[str, str],
    exp_summary_df: pd.DataFrame,
    highlight_weeks: Tuple[int, int] = (10, 20),
) -> None:
    """
    6 pav.: absolute model error vs RCT mean-difference over time, per experiment.
    Delta_im = tau_model - tau_RCT, plotted as |Delta_im|.
    """
    d = results_df.copy()
    naive = d[d["method"] == "naive"][["experiment_id", "week", "tau"]].rename(columns={"tau": "tau_rct"})
    d = d[d["method"].isin(["winsor", "cuped", "tl_ols"])].merge(naive, on=["experiment_id", "week"], how="left")
    d["abs_error"] = (d["tau"] - d["tau_rct"]).abs()

    n = len(exp_order)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.4 * nrows), sharex=True, sharey=False)
    axes = np.array(axes).reshape(-1)
    n_map = dict(zip(exp_summary_df["experiment_id"], exp_summary_df["n_e"]))

    for i, exp_id in enumerate(exp_order):
        ax = axes[i]
        sube = d[d["experiment_id"] == exp_id]
        for m in ["winsor", "cuped", "tl_ols"]:
            s = sube[sube["method"] == m].sort_values("week")
            ax.plot(
                s["week"],
                s["abs_error"],
                color=METHOD_COLORS[m],
                label=METHOD_LABELS[m],
                linewidth=1.8,
                marker="o",
                markersize=3.5,
            )

        # Highlight least biased model at requested weeks
        for w in highlight_weeks:
            sw = sube[sube["week"] == w].dropna(subset=["abs_error"])
            if sw.empty:
                continue
            idx = sw["abs_error"].idxmin()
            best = sw.loc[idx]
            ax.scatter(
                [w],
                [best["abs_error"]],
                marker="*",
                s=170,
                color="black",
                zorder=5,
            )
            ax.annotate(
                f"w={w}: {METHOD_LABELS.get(best['method'], best['method'])}",
                (w, best["abs_error"]),
                textcoords="offset points",
                xytext=(5, 8),
                fontsize=8,
                color="black",
            )

        ax.set_xlim(min(ALL_WEEKS), max(ALL_WEEKS))
        ax.set_xticks(ALL_WEEKS)
        ax.set_yticklabels([])
        ax.set_ylabel("")
        ax.set_title(f"{label_map.get(exp_id, "experiment")}, n_e={n_map.get(exp_id)}")
        ax.grid(axis="y", alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.legend(
        [METHOD_LABELS[m] for m in ["winsor", "cuped", "tl_ols"]],
        loc="upper right",
        ncol=1,
    )
    fig.suptitle("Absoliuti modelių paklaida pagal savaites")
    plt.tight_layout()
    plt.show()


def plot_std_diff_per_experiment(
    results_df: pd.DataFrame,
    exp_order: List[str],
    label_map: Dict[str, str],
    exp_summary_df: pd.DataFrame,
) -> None:
    """
    Plot std_diff = (tau_method - tau_naive) / se_naive over weeks, per experiment.
    One subplot per experiment, all non-naive methods shown.
    """
    d = results_df[results_df["method"].isin(["winsor", "cuped", "tl_ols"])].copy()
    methods = ["winsor", "cuped", "tl_ols"]
    n = len(exp_order)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.3 * nrows), sharex=True, sharey=True)
    axes = np.array(axes).reshape(-1)
    n_map = dict(zip(exp_summary_df["experiment_id"], exp_summary_df["n_e"]))

    for i, exp_id in enumerate(exp_order):
        ax = axes[i]
        sube = d[d["experiment_id"] == exp_id]
        for m in methods:
            s = sube[sube["method"] == m].sort_values("week")
            ax.plot(
                s["week"],
                s["std_diff"],
                color=METHOD_COLORS[m],
                label=METHOD_LABELS[m],
                linewidth=1.6,
                marker="o",
                markersize=3.5,
            )
        ax.axhline(0, linestyle="--", color="gray", alpha=0.6)
        ax.set_xlim(min(ALL_WEEKS), max(ALL_WEEKS))
        ax.set_xticks(ALL_WEEKS)
        ax.set_yticklabels([])
        ax.set_title(f"{label_map.get(exp_id, "experiment")}, n_e={n_map.get(exp_id)}")
        ax.grid(axis="y", alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    fig.legend([METHOD_LABELS[m] for m in methods], loc="upper right", ncol=1)
    fig.suptitle("Standartizuotas ATE skirtumas nuo stebėto ATE pagal eksperimentą")
    plt.tight_layout()
    plt.show()


def plot_std_diff_aggregated(
    results_df: pd.DataFrame,
) -> None:
    """
    Plot mean and median std_diff across all experiments over time, per method.
    Two subplots: top = mean, bottom = median.
    """
    d = results_df[results_df["method"].isin(["winsor", "cuped", "tl_ols"])].copy()
    methods = ["winsor", "cuped", "tl_ols"]

    fig, (ax_mean, ax_med) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for m in methods:
        sub = d[d["method"] == m].groupby("week")["std_diff"]
        mean_vals = sub.mean()
        med_vals = sub.median()

        ax_mean.plot(
            mean_vals.index,
            mean_vals.values,
            color=METHOD_COLORS[m],
            linewidth=2,
            marker="o",
            markersize=4,
            label=METHOD_LABELS[m],
        )
        ax_med.plot(
            med_vals.index,
            med_vals.values,
            color=METHOD_COLORS[m],
            linewidth=2,
            marker="o",
            markersize=4,
            label=METHOD_LABELS[m],
        )

    for ax in (ax_mean, ax_med):
        ax.axhline(0, linestyle="--", color="gray", linewidth=0.9)
        ax.set_xlim(min(ALL_WEEKS), max(ALL_WEEKS))
        ax.set_xticks(ALL_WEEKS)
        ax.grid(axis="y", alpha=0.3)
        ax.set_yticklabels([])

    ax_mean.set_title("Vidurkis tarp eksperimentų")
    ax_mean.set_ylabel("std_diff (vidurkis)")
    ax_med.set_title("Mediana tarp eksperimentų")
    ax_med.set_ylabel("std_diff (mediana)")
    ax_med.set_xlabel("Kaupiamoji savaitė")

    fig.legend([METHOD_LABELS[m] for m in methods], loc="upper right", ncol=1)
    fig.suptitle("Standartizuotas ATE skirtumas nuo stebėto ATE — agreguota per eksperimentus", y=1.01)
    plt.tight_layout()
    plt.show()


def plot_ate_ir_pi_plocio_dinamika_be_y_etikeciu(
    results_df: pd.DataFrame,
    exp_order: List[str],
    label_map: Dict[str, str],
    exp_summary_df: pd.DataFrame,
) -> None:
    """
    Papildomas grafikas kaip 05-10, bet kiekvienam eksperimentui atskirai:
    kiekvienoje eilutėje kairėje ATE dinamika, dešinėje PI (95%) pločio dinamika.
    Visi grafikai be Y ašies etikečių/skaičių.
    """
    d = results_df.copy()
    methods = ["naive", "winsor", "cuped", "tl_ols"]
    week_min = int(d["week"].min())
    week_max = int(d["week"].max())
    weeks = list(range(week_min, week_max + 1))
    n = len(exp_order)
    fig, axes = plt.subplots(n, 2, figsize=(14, max(3.2 * n, 4.0)), sharex=True, sharey=False)
    if n == 1:
        axes = np.array([axes])
    n_map = dict(zip(exp_summary_df["experiment_id"], exp_summary_df["n_e"]))

    for r, exp_id in enumerate(exp_order):
        de = d[d["experiment_id"] == exp_id]

        ax_ate = axes[r, 0]
        for m in methods:
            s = de[de["method"] == m].sort_values("week")
            ax_ate.plot(
                s["week"],
                s["tau"],
                color=METHOD_COLORS[m],
                marker="o",
                markersize=3.5,
                linewidth=1.6,
                label=METHOD_LABELS[m],
            )
        ax_ate.axhline(0, color="grey", linewidth=0.8, linestyle="--")
        ax_ate.set_ylabel("")
        ax_ate.set_yticklabels([])
        ax_ate.set_xticks(weeks)
        ax_ate.xaxis.set_minor_locator(mticker.NullLocator())
        ax_ate.grid(axis="y", alpha=0.3)
        if r == 0:
            ax_ate.set_title("Poveikio įverčio (ATE) dinamika", fontsize=11)
        ax_ate.set_title(f"{label_map.get(exp_id, "experiment")}, n_e={n_map.get(exp_id)}", loc="left", fontsize=10)

        ax_pi = axes[r, 1]
        for m in methods:
            s = de[de["method"] == m].sort_values("week")
            ax_pi.plot(
                s["week"],
                s["pi_width"],
                color=METHOD_COLORS[m],
                marker="o",
                markersize=3.5,
                linewidth=1.6,
                label=METHOD_LABELS[m],
            )
        ax_pi.set_ylabel("")
        ax_pi.set_yticklabels([])
        ax_pi.set_xticks(weeks)
        ax_pi.xaxis.set_minor_locator(mticker.NullLocator())
        ax_pi.grid(axis="y", alpha=0.3)
        if r == 0:
            ax_pi.set_title("95% PI pločio dinamika", fontsize=11)

    axes[-1, 0].set_xlabel("Kaupiamoji savaitė")
    axes[-1, 1].set_xlabel("Kaupiamoji savaitė")
    fig.legend([METHOD_LABELS[m] for m in methods], loc="upper right", ncol=1)
    plt.suptitle("Poveikio įverčio ir PI pločio dinamika kiekvienam eksperimentui", fontsize=11, y=1.01)
    plt.tight_layout()
    plt.show()


def plot_figure7(results_df: pd.DataFrame, exp_order: List[str], label_map: Dict[str, str]) -> None:
    methods = METHOD_ORDER
    n = len(exp_order)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.3 * nrows), sharex=True)
    axes = np.array(axes).reshape(-1)
    for i, exp_id in enumerate(exp_order):
        ax = axes[i]
        sube = results_df[results_df["experiment_id"] == exp_id]
        for m in methods:
            s = sube[sube["method"] == m].sort_values("week")
            ax.plot(
                s["week"],
                s["se"],
                color=METHOD_COLORS[m],
                label=METHOD_LABELS[m],
                linewidth=1.6,
                marker="o",
                markersize=3.5,
            )
        ax.set_xlim(min(ALL_WEEKS), max(ALL_WEEKS))
        ax.set_xticks(ALL_WEEKS)
        ax.set_yticklabels([])
        ax.set_ylabel("Standartinė paklaida", fontsize=12)
        ax.set_title(label_map.get(exp_id, "experiment"), fontsize=13)
        ax.grid(axis="y", alpha=0.3)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    fig.legend([METHOD_LABELS[m] for m in methods], loc="lower right", ncol=1)
    plt.tight_layout()
    plt.show()


def plot_figure8_median(results_df: pd.DataFrame) -> None:
    """Dispersijos sumažinimo mediana per eksperimentus laike."""
    d = results_df[results_df["method"].isin(["winsor", "cuped", "tl_ols"])].copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    for m in ["winsor", "cuped", "tl_ols"]:
        sub = d[d["method"] == m]
        vals = 100 * sub.groupby("week")["ds"].median()
        ax.plot(
            vals.index,
            vals.values,
            color=METHOD_COLORS[m],
            linewidth=2,
            marker="o",
            markersize=4,
            label=METHOD_LABELS[m],
        )
    ax.axhline(0, linestyle="--", color="gray")
    ax.set_xlim(min(ALL_WEEKS), max(ALL_WEEKS))
    ax.set_xticks(ALL_WEEKS)
    ax.set_xlabel("Savaitė")
    ax.set_ylabel("Dispersijos sumažinimas (%)")
    ax.set_title("Dispersijos sumažinimo dinamika (mediana per eksperimentus)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_figure8_mean(results_df: pd.DataFrame) -> None:
    """Dispersijos sumažinimo vidurkis per eksperimentus laike."""
    d = results_df[results_df["method"].isin(["winsor", "cuped", "tl_ols"])].copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    for m in ["winsor", "cuped", "tl_ols"]:
        sub = d[d["method"] == m]
        vals = 100 * sub.groupby("week")["ds"].mean()
        ax.plot(
            vals.index,
            vals.values,
            color=METHOD_COLORS[m],
            linewidth=2,
            marker="o",
            markersize=4,
            label=METHOD_LABELS[m],
        )
    ax.axhline(0, linestyle="--", color="gray")
    ax.set_xlim(min(ALL_WEEKS), max(ALL_WEEKS))
    ax.set_xticks(ALL_WEEKS)
    ax.set_xlabel("Savaitė")
    ax.set_ylabel("Dispersijos sumažinimas (%)")
    ax.set_title("Dispersijos sumažinimo dinamika (vidurkis per eksperimentus)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_figure8(results_df: pd.DataFrame) -> None:
    """Backwards-compatible wrapper — plots both median and mean."""
    plot_figure8_median(results_df)
    plot_figure8_mean(results_df)


def plot_figure9(results_df: pd.DataFrame, exp_order: List[str], label_map: Dict[str, str], exp_summary_df: pd.DataFrame) -> None:
    d = results_df[results_df["method"].isin(["winsor", "cuped", "tl_ols"])].copy()
    n = len(exp_order)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.4 * nrows), sharex=True, sharey=True)
    axes = np.array(axes).reshape(-1)
    n_map = dict(zip(exp_summary_df["experiment_id"], exp_summary_df["n_e"]))
    for i, exp_id in enumerate(exp_order):
        ax = axes[i]
        sube = d[d["experiment_id"] == exp_id]
        for m in ["winsor", "cuped", "tl_ols"]:
            s = sube[sube["method"] == m].sort_values("week")
            ax.plot(
                s["week"],
                100 * s["ds"],
                color=METHOD_COLORS[m],
                label=METHOD_LABELS[m],
                linewidth=1.6,
                marker="o",
                markersize=3.5,
            )
        ax.axhline(0, linestyle="--", color="gray", alpha=0.7)
        ax.set_xlim(min(ALL_WEEKS), max(ALL_WEEKS))
        ax.set_xticks(ALL_WEEKS)
        ax.set_ylabel("Dispersijos sumažinimas (%)", fontsize=12)
        ax.set_title(f"{label_map.get(exp_id, "experiment")}, $n_e$={n_map.get(exp_id)}", fontsize=13)
        ax.tick_params(axis="y", labelsize=11)
        ax.grid(axis="y", alpha=0.3)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    fig.legend([METHOD_LABELS[m] for m in ["winsor", "cuped", "tl_ols"]], loc="lower right", ncol=1, fontsize=11)
    fig.suptitle("Dispersijos sumažinimo dinamika per eksperimentus", fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_figure10_median(r2_df: pd.DataFrame) -> None:
    """TL R² mediana per eksperimentus laike."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for col, lbl, c in [
        ("r2_control", "Kontrolinė gr.", "#1f77b4"),
        ("r2_treated", "Eksperimentinė gr.", "#ff7f0e"),
    ]:
        vals = r2_df.groupby("week")[col].median()
        ax.plot(vals.index, vals.values, color=c, linewidth=2, marker="o", markersize=4, label=lbl)
    ax.set_xlim(min(ALL_WEEKS), max(ALL_WEEKS))
    ax.set_xticks(ALL_WEEKS)
    ax.set_xlabel("Savaitė")
    ax.set_ylabel("$R^2$")
    ax.set_title("TL $R^2$ dinamika (mediana per eksperimentus)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_figure10_mean(r2_df: pd.DataFrame) -> None:
    """TL R² vidurkis per eksperimentus laike."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for col, lbl, c in [
        ("r2_control", "Kontrolinė gr.", "#1f77b4"),
        ("r2_treated", "Eksperimentinė gr.", "#ff7f0e"),
    ]:
        vals = r2_df.groupby("week")[col].mean()
        ax.plot(vals.index, vals.values, color=c, linewidth=2, marker="o", markersize=4, label=lbl)
    ax.set_xlim(min(ALL_WEEKS), max(ALL_WEEKS))
    ax.set_xticks(ALL_WEEKS)
    ax.set_xlabel("Savaitė")
    ax.set_ylabel("$R^2$")
    ax.set_title("TL $R^2$ dinamika (vidurkis per eksperimentus)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_figure10(r2_df: pd.DataFrame) -> None:
    """Backwards-compatible wrapper — plots both median and mean."""
    plot_figure10_median(r2_df)
    plot_figure10_mean(r2_df)


def plot_figure10_by_experiment(
    r2_df: pd.DataFrame,
    exp_order: List[str],
    label_map: Dict[str, str],
    exp_summary_df: pd.DataFrame,
) -> None:
    n = len(exp_order)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.2 * nrows), sharex=True, sharey=True)
    axes = np.array(axes).reshape(-1)
    n_map = dict(zip(exp_summary_df["experiment_id"], exp_summary_df["n_e"]))
    for i, exp_id in enumerate(exp_order):
        ax = axes[i]
        sub = r2_df[r2_df["experiment_id"] == exp_id].sort_values("week")
        ax.plot(
            sub["week"],
            sub["r2_control"],
            color="#1f77b4",
            linewidth=1.8,
            marker="o",
            markersize=3.5,
            label="Kontrolinė gr.",
        )
        ax.plot(
            sub["week"],
            sub["r2_treated"],
            color="#ff7f0e",
            linewidth=1.8,
            marker="o",
            markersize=3.5,
            label="Eksperimentinė gr.",
        )
        ax.set_xlim(min(ALL_WEEKS), max(ALL_WEEKS))
        ax.set_xticks(ALL_WEEKS)
        ax.set_ylabel("$R^2$")
        ax.set_title(f"{label_map.get(exp_id, "experiment")}, $n_e$={n_map.get(exp_id)}")
        ax.grid(axis="y", alpha=0.3)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    fig.legend(["Kontrolinė gr.", "Eksperimentinė gr."], loc="upper right", ncol=1)
    fig.suptitle("TL $R^2$ dinamika per eksperimentus")
    plt.tight_layout()
    plt.show()


def plot_figure6_bias_percent(
    results_df: pd.DataFrame,
    exp_order: List[str],
    label_map: Dict[str, str],
) -> None:
    d = results_df.copy()
    naive = d[d["method"] == "naive"][["experiment_id", "week", "tau"]].rename(columns={"tau": "tau_naivus"})
    d = d.merge(naive, on=["experiment_id", "week"], how="left")
    d["poslinkis_proc"] = 100 * (d["tau"] - d["tau_naivus"]) / d["tau_naivus"].replace(0, np.nan)
    d = d[d["method"].isin(["winsor", "cuped", "tl_ols"])].copy()

    n = len(exp_order)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.4 * nrows), sharex=True, sharey=True)
    axes = np.array(axes).reshape(-1)

    for i, exp_id in enumerate(exp_order):
        ax = axes[i]
        sub = d[d["experiment_id"] == exp_id].copy()
        for m in ["winsor", "cuped", "tl_ols"]:
            s = sub[sub["method"] == m].sort_values("week")
            ax.plot(
                s["week"],
                s["poslinkis_proc"],
                marker="o",
                markersize=3.5,
                linewidth=1.8,
                color=METHOD_COLORS[m],
                label=METHOD_LABELS[m],
            )
        ax.axhline(0, linestyle="--", color="gray", linewidth=1.0)
        ax.set_xlim(min(ALL_WEEKS), max(ALL_WEEKS))
        ax.set_xticks(ALL_WEEKS)
        ax.set_ylabel("ATE poslinkis nuo stebėto (%)")
        ax.set_title(label_map.get(exp_id, "experiment"))
        ax.grid(axis="y", alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.legend([METHOD_LABELS[m] for m in ["winsor", "cuped", "tl_ols"]], loc="upper right", ncol=1)
    fig.suptitle("ATE poslinkio dinamika per eksperimentus (%)")
    plt.tight_layout()
    plt.show()


def build_table_2(
    r2_df: pd.DataFrame,
    results_df: pd.DataFrame,
    exp_summary_df: pd.DataFrame,
    label_map: Dict[str, str],
) -> pd.DataFrame:
    r20 = r2_df[r2_df["week"] == 20][["experiment_id", "r2_control", "r2_treated"]].copy()
    ds20 = results_df[(results_df["week"] == 20) & (results_df["method"] == "tl_ols")][
        ["experiment_id", "ds"]
    ].copy()
    out = (
        exp_summary_df.set_index("experiment_id")
        .join(r20.set_index("experiment_id"))
        .join(ds20.set_index("experiment_id"))
        .reset_index()
    )
    out["Eksperimentas"] = out["experiment_id"].map(label_map)
    out["TL dispersijos sumažinimas (%)"] = 100 * out["ds"]
    out = out[["Eksperimentas", "n_e", "r2_control", "r2_treated", "TL dispersijos sumažinimas (%)"]]
    out = out.rename(
        columns={
            "r2_control": "R² kontrolės grupei",
            "r2_treated": "R² testinei grupei",
        }
    )
    avg = pd.DataFrame(
        [
            {
                "Eksperimentas": "Vidurkis",
                "n_e": out["n_e"].mean(),
                "R² kontrolės grupei": out["R² kontrolės grupei"].mean(),
                "R² testinei grupei": out["R² testinei grupei"].mean(),
                "TL dispersijos sumažinimas (%)": out["TL dispersijos sumažinimas (%)"].mean(),
            }
        ]
    )
    return pd.concat([out, avg], ignore_index=True)


def build_table_figure1_density_stats(
    df: pd.DataFrame,
    exp_order: List[str],
    label_map: Dict[str, str],
    weeks: Tuple[int, int, int] = (1, 10, 20),
) -> pd.DataFrame:
    rows = []
    for exp_id in exp_order:
        for w in weeks:
            y = pd.to_numeric(df.loc[df["experiment_id"] == exp_id, f"Y_{OUTCOME_KIND}_post_{w}w"], errors="coerce").dropna()
            if len(y) == 0:
                continue
            rows.append(
                {
                    "Eksperimentas": label_map.get(exp_id, "experiment"),
                    "Savaitė": int(w),
                    "N": int(len(y)),
                    "Vidurkis": float(y.mean()),
                    "Mediana": float(y.median()),
                    "P1": float(np.percentile(y, 1)),
                    "P99": float(np.percentile(y, 99)),
                }
            )
    return pd.DataFrame(rows)


def build_table_figure5_points(
    results_df: pd.DataFrame,
    exp_summary_df: pd.DataFrame,
    label_map: Dict[str, str],
) -> pd.DataFrame:
    d = results_df[(results_df["week"] == 20) & (results_df["method"].isin(["winsor", "cuped", "tl_ols"]))].copy()
    d = d.merge(exp_summary_df[["experiment_id", "n_e"]], on="experiment_id", how="left")
    d["Eksperimentas"] = d["experiment_id"].map(label_map)
    d["Metodas"] = d["method"].map(METHOD_LABELS)
    d["Dispersijos sumažinimas (%)"] = 100 * d["ds"]
    return d[["Eksperimentas", "n_e", "Metodas", "Dispersijos sumažinimas (%)"]].sort_values(
        ["Eksperimentas", "Metodas"]
    )


def build_table_figure6_ate_bias(
    results_df: pd.DataFrame,
    exp_summary_df: pd.DataFrame,
    label_map: Dict[str, str],
) -> pd.DataFrame:
    d = results_df.copy()
    naive = d[d["method"] == "naive"][["experiment_id", "week", "tau"]].rename(
        columns={"tau": "tau_naivus"}
    )
    d = d.merge(naive, on=["experiment_id", "week"], how="left")
    d["poslinkis_proc_nuo_naivaus"] = 100 * (d["tau"] - d["tau_naivus"]) / d["tau_naivus"].replace(0, np.nan)
    d = d.merge(exp_summary_df[["experiment_id", "n_e"]], on="experiment_id", how="left")
    d["Eksperimentas"] = d["experiment_id"].map(label_map)
    d["Metodas"] = d["method"].map(METHOD_LABELS)
    d["absoliuti_paklaida"] = (d["tau"] - d["tau_naivus"]).abs()
    out = d[
        [
            "Eksperimentas",
            "n_e",
            "week",
            "Metodas",
            "tau_naivus",
            "tau",
            "poslinkis_proc_nuo_naivaus",
            "absoliuti_paklaida",
        ]
    ].rename(
        columns={
            "week": "Savaitė",
            "tau_naivus": "Be metodo ATE",
            "tau": "Poveikio įvertis",
            "poslinkis_proc_nuo_naivaus": "ATE poslinkis nuo stebėto (%)",
            "absoliuti_paklaida": "Absoliuti paklaida |Δ_im|",
        }
    )
    return out.sort_values(["Eksperimentas", "Savaitė", "Metodas"])


def build_table_figure6_least_biased(
    results_df: pd.DataFrame,
    exp_summary_df: pd.DataFrame,
    label_map: Dict[str, str],
    weeks: Tuple[int, int] = (10, 20),
) -> pd.DataFrame:
    d = results_df.copy()
    naive = d[d["method"] == "naive"][["experiment_id", "week", "tau"]].rename(columns={"tau": "tau_rct"})
    d = d[d["method"].isin(["winsor", "cuped", "tl_ols"])].merge(naive, on=["experiment_id", "week"], how="left")
    d["abs_error"] = (d["tau"] - d["tau_rct"]).abs()
    d = d.merge(exp_summary_df[["experiment_id", "n_e"]], on="experiment_id", how="left")
    d["Eksperimentas"] = d["experiment_id"].map(label_map)
    d["Metodas"] = d["method"].map(METHOD_LABELS)

    rows = []
    for exp_id in d["experiment_id"].dropna().unique():
        de = d[d["experiment_id"] == exp_id]
        for w in weeks:
            dw = de[de["week"] == w].dropna(subset=["abs_error"])
            if dw.empty:
                continue
            best = dw.loc[dw["abs_error"].idxmin()]
            rows.append(
                {
                    "Eksperimentas": best["Eksperimentas"],
                    "n_e": int(best["n_e"]),
                    "Savaitė": int(w),
                    "Mažiausio poslinkio metodas": best["Metodas"],
                    "Absoliuti paklaida |Δ_im|": float(best["abs_error"]),
                }
            )
    return pd.DataFrame(rows).sort_values(["Eksperimentas", "Savaitė"])


def build_table_figure7_se(
    results_df: pd.DataFrame,
    exp_summary_df: pd.DataFrame,
    label_map: Dict[str, str],
) -> pd.DataFrame:
    d = results_df.merge(exp_summary_df[["experiment_id", "n_e"]], on="experiment_id", how="left").copy()
    d["Eksperimentas"] = d["experiment_id"].map(label_map)
    d["Metodas"] = d["method"].map(METHOD_LABELS)
    out = d[["Eksperimentas", "n_e", "week", "Metodas", "se"]].rename(
        columns={"week": "Savaitė", "se": "Standartinė paklaida"}
    )
    return out.sort_values(["Eksperimentas", "Savaitė", "Metodas"])


def build_table_figure8_weekly_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    d = results_df[results_df["method"].isin(["winsor", "cuped", "tl_ols"])].copy()
    g = d.groupby(["method", "week"])["ds"]
    out = g.median().reset_index(name="Dispersijos sumažinimo mediana")
    out["Dispersijos sumažinimo mediana"] = 100 * out["Dispersijos sumažinimo mediana"]
    out["Metodas"] = out["method"].map(METHOD_LABELS)
    out = out.rename(columns={"week": "Savaitė"})
    return out[["Savaitė", "Metodas", "Dispersijos sumažinimo mediana"]].sort_values(["Savaitė", "Metodas"])


def build_table_figure9_ds_by_experiment(
    results_df: pd.DataFrame,
    exp_summary_df: pd.DataFrame,
    label_map: Dict[str, str],
) -> pd.DataFrame:
    d = results_df[results_df["method"].isin(["winsor", "cuped", "tl_ols"])].copy()
    d = d.merge(exp_summary_df[["experiment_id", "n_e"]], on="experiment_id", how="left")
    d["Eksperimentas"] = d["experiment_id"].map(label_map)
    d["Metodas"] = d["method"].map(METHOD_LABELS)
    d["Dispersijos sumažinimas (%)"] = 100 * d["ds"]
    out = d[["Eksperimentas", "n_e", "week", "Metodas", "Dispersijos sumažinimas (%)"]].rename(
        columns={"week": "Savaitė"}
    )
    return out.sort_values(["Eksperimentas", "Savaitė", "Metodas"])


def build_table_figure10_r2(r2_df: pd.DataFrame, exp_summary_df: pd.DataFrame, label_map: Dict[str, str]) -> pd.DataFrame:
    d = r2_df.merge(exp_summary_df[["experiment_id", "n_e"]], on="experiment_id", how="left").copy()
    d["Eksperimentas"] = d["experiment_id"].map(label_map)
    return d[
        ["Eksperimentas", "n_e", "week", "r2_control", "r2_treated"]
    ].rename(
        columns={
            "week": "Savaitė",
            "r2_control": "R² kontrolės grupei",
            "r2_treated": "R² testinei grupei",
        }
    ).sort_values(["Eksperimentas", "Savaitė"])


def build_table_figure11_points(results_df: pd.DataFrame, r2_df: pd.DataFrame, exp_summary_df: pd.DataFrame, label_map: Dict[str, str]) -> pd.DataFrame:
    ds = results_df[results_df["method"] == "tl_ols"][["experiment_id", "week", "ds"]].copy()
    r2 = r2_df.copy()
    r2["r2_vidurkis"] = (r2["r2_control"] + r2["r2_treated"]) / 2
    out = ds.merge(r2[["experiment_id", "week", "r2_vidurkis"]], on=["experiment_id", "week"], how="inner")
    out = out.merge(exp_summary_df[["experiment_id", "n_e"]], on="experiment_id", how="left")
    out["Eksperimentas"] = out["experiment_id"].map(label_map)
    out["TL dispersijos sumažinimas (%)"] = 100 * out["ds"]
    return out[["Eksperimentas", "n_e", "week", "r2_vidurkis", "TL dispersijos sumažinimas (%)"]].rename(
        columns={"week": "Savaitė", "r2_vidurkis": "R² vidurkis"}
    ).sort_values(["Eksperimentas", "Savaitė"])


def build_modeled_ate_vs_rct_table(
    results_df: pd.DataFrame,
    week: int = 20,
    methods: Tuple[str, ...] = ("winsor", "cuped", "tl_ols"),
) -> pd.DataFrame:
    """
    Best basic comparison table:
    Delta_m = tau_hat_m - tau_hat_RCT

    Uses naive mean-difference as tau_hat_RCT.
    Aggregates by taking mean tau across experiments at a chosen week.
    """
    d = results_df[results_df["week"] == week].copy()

    rct = d[d["method"] == "naive"]["tau"].mean()
    rows = []
    for method in methods:
        modeled = d[d["method"] == method]["tau"].mean()
        diff = modeled - rct
        rows.append(
            {
                "Model": METHOD_LABELS.get(method, method),
                "Modeled ATE": modeled,
                "RCT mean-diff ATE": rct,
                "Difference": diff,
                "Absolute difference": abs(diff),
            }
        )
    return pd.DataFrame(rows)


def build_standardized_ate_error_tables(
    df: pd.DataFrame,
    results_df: pd.DataFrame,
    exp_summary_df: pd.DataFrame,
    label_map: Dict[str, str],
    week: int = 20,
    methods: Tuple[str, ...] = ("winsor", "cuped", "tl_ols"),
    experiment_col: str = "experiment_id",
    treatment_col: str = "W",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Standardized model-vs-RCT ATE errors for cross-experiment comparability.

    Delta_im = tau_hat_im(model) - tau_hat_i(RCT),
    E_im     = Delta_im / s_i,
    where s_i is control-group SD of Y at selected week.

    Returns:
      1) per_experiment_df with Delta and standardized E
      2) summary_df with Bias_m and MASE_m across experiments
    """
    y_col = f"Y_{OUTCOME_KIND}_post_{week}w"
    d_w = results_df[results_df["week"] == week].copy()
    naive = d_w[d_w["method"] == "naive"][["experiment_id", "tau"]].rename(columns={"tau": "tau_rct"})
    modeled = d_w[d_w["method"].isin(methods)][["experiment_id", "method", "tau"]].rename(
        columns={"tau": "tau_model"}
    )
    merged = modeled.merge(naive, on="experiment_id", how="left")

    # s_i: control-group SD (W=0) of selected outcome at selected week
    sd_rows = []
    for exp_id in exp_summary_df[experiment_col]:
        d_exp = df[df[experiment_col] == exp_id]
        y_ctrl = pd.to_numeric(
            d_exp.loc[d_exp[treatment_col] == 0, y_col],
            errors="coerce",
        ).dropna()
        s_i = float(y_ctrl.std(ddof=1)) if len(y_ctrl) >= 2 else np.nan
        sd_rows.append({"experiment_id": exp_id, "s_i": s_i})
    sd_df = pd.DataFrame(sd_rows)

    merged = merged.merge(sd_df, on="experiment_id", how="left")
    merged["delta"] = merged["tau_model"] - merged["tau_rct"]
    merged["E"] = merged["delta"] / merged["s_i"].replace(0, np.nan)
    merged["abs_E"] = merged["E"].abs()
    merged["Eksperimentas"] = merged["experiment_id"].map(label_map)
    merged["Metodas"] = merged["method"].map(METHOD_LABELS)
    merged = merged.merge(exp_summary_df[["experiment_id", "n_e"]], on="experiment_id", how="left")

    per_experiment_df = merged[
        [
            "Eksperimentas",
            "n_e",
            "Metodas",
            "tau_model",
            "tau_rct",
            "delta",
            "s_i",
            "E",
            "abs_E",
        ]
    ].rename(
        columns={
            "tau_model": "Modeliuotas ATE",
            "tau_rct": "RCT vidurkių skirtumo ATE",
            "delta": "Skirtumas Δ_im",
            "s_i": "Kontrolinės grupės SD (s_i)",
            "E": "Standartizuota paklaida E_im",
            "abs_E": "|E_im|",
        }
    ).sort_values(["Eksperimentas", "Metodas"])

    summary_df = (
        merged.groupby("method", as_index=False)
        .agg(
            Bias=("E", "mean"),
            MASE=("abs_E", "mean"),
            Eksperimentų_skaičius=("E", "count"),
        )
        .rename(columns={"method": "method_key"})
    )
    summary_df["Metodas"] = summary_df["method_key"].map(METHOD_LABELS)
    summary_df = summary_df[["Metodas", "Bias", "MASE", "Eksperimentų_skaičius"]].sort_values("Metodas")

    return per_experiment_df, summary_df


def build_overall_absolute_error_stability_table(
    results_df: pd.DataFrame,
    methods: Tuple[str, ...] = ("winsor", "cuped", "tl_ols"),
) -> pd.DataFrame:
    """
    Stability summary across all weeks (and all experiments):
    absolute error vs RCT mean-difference ATE.

    Delta_imw = tau_model - tau_RCT
    Uses |Delta_imw| aggregated over all available (experiment, week) points.
    """
    d = results_df.copy()
    naive = d[d["method"] == "naive"][["experiment_id", "week", "tau"]].rename(columns={"tau": "tau_rct"})
    modeled = d[d["method"].isin(methods)].merge(naive, on=["experiment_id", "week"], how="left")
    modeled["abs_error"] = (modeled["tau"] - modeled["tau_rct"]).abs()

    summary = (
        modeled.groupby("method", as_index=False)
        .agg(
            Vidutinė_absoliuti_paklaida=("abs_error", "mean"),
            Mediana_absoliuti_paklaida=("abs_error", "median"),
            Didžiausia_absoliuti_paklaida=("abs_error", "max"),
            Taškų_skaičius=("abs_error", "count"),
        )
        .rename(columns={"method": "method_key"})
    )

    summary["Metodas"] = summary["method_key"].map(METHOD_LABELS)
    summary["Rangas"] = summary["Vidutinė_absoliuti_paklaida"].rank(method="dense", ascending=True).astype(int)
    min_val = summary["Vidutinė_absoliuti_paklaida"].min()
    summary["Stabiliausias_modelis"] = np.where(
        np.isclose(summary["Vidutinė_absoliuti_paklaida"], min_val),
        "TAIP",
        "",
    )

    return summary[
        [
            "Metodas",
            "Vidutinė_absoliuti_paklaida",
            "Mediana_absoliuti_paklaida",
            "Didžiausia_absoliuti_paklaida",
            "Taškų_skaičius",
            "Rangas",
            "Stabiliausias_modelis",
        ]
    ].sort_values(["Rangas", "Metodas"])


def plot_figure11(results_df: pd.DataFrame, r2_df: pd.DataFrame) -> None:
    ds = results_df[results_df["method"] == "tl_ols"][["experiment_id", "week", "ds"]].copy()
    r2 = r2_df.copy()
    r2["r2_mean"] = (r2["r2_control"] + r2["r2_treated"]) / 2
    plot_df = ds.merge(r2[["experiment_id", "week", "r2_mean"]], on=["experiment_id", "week"], how="inner")
    plot_df["ds_pct"] = 100 * plot_df["ds"]

    x = plot_df["r2_mean"].to_numpy()
    y = plot_df["ds_pct"].to_numpy()

    fig, ax = plt.subplots(figsize=(9, 5))
    sc = ax.scatter(x, y, c=plot_df["week"], cmap="viridis", alpha=0.9, edgecolor="black", linewidth=0.3)
    if len(plot_df) >= 2:
        b1, b0 = np.polyfit(x, y, 1)
        xg = np.linspace(max(0, np.nanmin(x)), np.nanmax(x) + 0.05, 100)
        ax.plot(xg, b1 * xg + b0, color="#d62728", linewidth=2, label="Empirinė tendencija")
    xline = np.linspace(0, max(0.01, np.nanmax(x) + 0.05), 100)
    ax.plot(xline, 100 * xline, color="gray", linestyle="--", linewidth=1.2, label="Teorinė DS = $R^2$")
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Savaitė")
    ax.set_xlim(0, max(0.01, np.nanmax(x) + 0.05))
    ax.set_ylim(0, 40)
    ax.set_xlabel("$R^2$ vidurkis ($(R^2_0 + R^2_1)/2$)")
    ax.set_ylabel("TL dispersijos sumažinimas (%)")
    ax.set_title("Empirinis ryšys tarp $R^2$ ir dispersijos sumažinimo")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()
