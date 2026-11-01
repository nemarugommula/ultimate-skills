# %% [markdown]
# # Lab 9: The capstone, an end-to-end churn project 🎓 (SOLUTION)
# Solution file: every TODO is filled in. Try labs/phase9_capstone.py first!
#
# You'll run a small but complete project, the way Lessons 9.1–9.3 describe it,
# on a synthetic subscription business built right here in the file (3,000
# customers observed over 3 monthly snapshots, about 13–16% churn a month):
#   1. a leakage-safe split: drop a column that leaks the label, train on
#      months 1–2 and test on month 3 (the future);
#   2. a ColumnTransformer + HistGradientBoostingClassifier Pipeline;
#   3. a cross-validated comparison against a baseline, grouped by customer;
#   4. a cost-based decision threshold (an offer costs $20, a lost customer $100);
#   5. a calibration check (do predicted 30% churners churn 30% of the time?);
#   6. permutation importance on the test month;
#   7. a model card: what the model is for, how good it is, and its limits.
#
# How to run: cell by cell in VS Code ("Run Cell" above each `# %%`), or the
# whole file with `python3 phase9_capstone.py`. The file stops at the first
# TODO you haven't filled in. Fill each TODO until its ✅ checks pass.
# Setup (once): `pip install -r requirements.txt`. Solutions: labs/solutions/
# Lessons to reread if stuck: 0.6 and 5.1 (splits, leakage), 1.6 (Pipelines),
# 4.4 (boosting), 5.5 (thresholds), 5.6 (calibration), 8.1 (permutation
# importance), 9.1–9.3 (design, cases, end-to-end projects)

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

rng = np.random.default_rng(0)  # one seeded random generator → reproducible results


def check(cond, msg):
    assert cond, "❌ " + msg
    print("✅", msg)


def show():
    """plt.show(), but quietly skipped when there is no screen (e.g. MPLBACKEND=Agg)."""
    if plt.get_backend().lower() == "agg":
        plt.close("all")
    else:
        plt.show()


def close(a, b, tol=1e-3):
    """True if a and b (numbers or arrays) agree to within tol."""
    return bool(np.allclose(a, b, atol=tol))


# %% The data: a subscription business, one row per customer per monthly snapshot
def make_churn_data(n_customers=3000, n_months=3, seed=0):
    """Each month we record every active customer's features; `churned` = they cancel within the
    next 30 days (then they disappear from later months). Built so the true drivers are known."""
    r = np.random.default_rng(seed)
    base = pd.DataFrame({
        "customer_id": np.arange(n_customers),
        "contract": r.choice(["monthly", "annual"], n_customers, p=[0.55, 0.45]),
        "plan": r.choice(["basic", "standard", "premium"], n_customers, p=[0.5, 0.35, 0.15]),
        "region": r.choice(["north", "south", "east", "west"], n_customers),
        "favorite_color": r.choice(["red", "green", "blue"], n_customers),  # pure noise, on purpose
        "tenure0": r.integers(1, 61, n_customers),
        "usage0": r.gamma(2.0, 2.5, n_customers),
    })
    price = base["plan"].map({"basic": 15, "standard": 30, "premium": 55}).to_numpy()
    rows, active = [], np.ones(n_customers, bool)
    for m in range(1, n_months + 1):
        idx = np.flatnonzero(active)
        b = base.iloc[idx]
        usage = np.maximum(0, b["usage0"].to_numpy() * r.normal(1, 0.25, len(idx)))
        df = pd.DataFrame({
            "customer_id": b["customer_id"].to_numpy(), "snapshot_month": m,
            "tenure_months": b["tenure0"].to_numpy() + m - 1,
            "contract": b["contract"].to_numpy(), "plan": b["plan"].to_numpy(), "region": b["region"].to_numpy(),
            "favorite_color": b["favorite_color"].to_numpy(),
            "monthly_charges": np.round(price[idx] * r.normal(1, 0.08, len(idx)), 2),
            "support_tickets_90d": r.poisson(0.8, len(idx)),
            "usage_hours_week": np.round(usage, 1),
            "days_since_login": np.round(r.exponential(4 + 6 / (1 + usage)), 0),
        })
        z = (-2.2 + 1.1 * (df["contract"] == "monthly") - 0.025 * df["tenure_months"]
             + 0.35 * df["support_tickets_90d"] + 0.11 * df["days_since_login"]
             - 0.12 * df["usage_hours_week"] + 0.012 * (df["monthly_charges"] - 25))
        df["churned"] = (r.random(len(df)) < 1 / (1 + np.exp(-z))).astype(int)
        # A column that only exists because the customer already asked to cancel: it leaks the label.
        df["cancel_request_logged"] = ((df["churned"] == 1) & (r.random(len(df)) < 0.85)).astype(int)
        rows.append(df)
        active[idx[df["churned"].to_numpy() == 1]] = False
    return pd.concat(rows, ignore_index=True)


data = make_churn_data()
print(data.shape, "rows × columns")
print(data.groupby("snapshot_month")["churned"].agg(["size", "mean"]).round(3))
print(data.head())

LEAKY = ["cancel_request_logged"]
NUM = ["tenure_months", "monthly_charges", "support_tickets_90d", "usage_hours_week", "days_since_login"]
CAT = ["contract", "plan", "region", "favorite_color"]
TARGET = "churned"


# %% TODO 1: a leakage-safe split   (Lessons 0.6, 5.1, 9.3)
def split_by_time(df, test_month=3):
    """Train on snapshots before `test_month`, test on `test_month` (the future), and drop the LEAKY
    columns from both. Return X_train, y_train, X_test, y_test, groups_train (customer ids, for CV)."""
    df = df.drop(columns=LEAKY)
    train, test = df[df["snapshot_month"] < test_month], df[df["snapshot_month"] == test_month]
    feats = NUM + CAT
    return train[feats], train[TARGET], test[feats], test[TARGET], train["customer_id"]


X_tr, y_tr, X_te, y_te, groups = split_by_time(data)
check(not any(c in X_tr.columns for c in LEAKY + [TARGET, "customer_id", "snapshot_month"]),
      "no leaky column, no label, no IDs among the features")
check(len(X_tr) == (data["snapshot_month"] < 3).sum() and len(X_te) == (data["snapshot_month"] == 3).sum(),
      f"train = months 1–2 ({len(X_tr)} rows), test = month 3 ({len(X_te)} rows)")
check(groups.nunique() < len(X_tr), "customers repeat across months: CV must group by customer")
leak_auc = roc_auc_score(data[TARGET], data["cancel_request_logged"])
print(f"the leaky column alone scores ROC AUC {leak_auc:.3f}: too good to be true")
check(leak_auc > 0.9, "the dropped column really was a leak")


# %% TODO 2: a ColumnTransformer + HistGradientBoosting Pipeline   (Lessons 1.6, 4.4)
def make_model():
    """Pipeline: ("prep", ColumnTransformer: NUM passthrough, CAT → OrdinalEncoder that maps unseen
    categories to -1) then ("hgb", HistGradientBoostingClassifier told which columns are categorical)."""
    prep = ColumnTransformer([
        ("num", "passthrough", NUM),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT),
    ])
    hgb = HistGradientBoostingClassifier(
        categorical_features=[False] * len(NUM) + [True] * len(CAT),  # NUM come first, then CAT
        learning_rate=0.1, max_iter=100, max_leaf_nodes=15, random_state=0)
    return Pipeline([("prep", prep), ("hgb", hgb)])


model = make_model()
check(isinstance(model, Pipeline) and list(model.named_steps) == ["prep", "hgb"], "a two-step Pipeline: prep → hgb")
check(isinstance(model.named_steps["hgb"], HistGradientBoostingClassifier), "the model is a HistGradientBoostingClassifier")
model.fit(X_tr, y_tr)
p_te = model.predict_proba(X_te)[:, 1]
test_auc, test_ap = roc_auc_score(y_te, p_te), average_precision_score(y_te, p_te)
print(f"test month: ROC AUC {test_auc:.3f}, PR AUC {test_ap:.3f} (churn rate {y_te.mean():.3f})")
check(test_auc > 0.75, "the model ranks churners well on the future month (ROC AUC > 0.75)")


# %% TODO 3: cross-validated comparison with a baseline   (Lessons 5.1, 9.3)
def compare(X, y, groups, n_splits=5):
    """5-fold GroupKFold by customer. Score a DummyClassifier(strategy="prior") and make_model() with
    ROC AUC and average precision. Return {"baseline": {"roc_auc": m, "ap": m}, "model": {...}} (means)."""
    cv = GroupKFold(n_splits=n_splits)
    out = {}
    for name, est in [("baseline", DummyClassifier(strategy="prior")), ("model", make_model())]:
        s = cross_validate(est, X, y, groups=groups, cv=cv, scoring=["roc_auc", "average_precision"])
        out[name] = {"roc_auc": float(s["test_roc_auc"].mean()), "ap": float(s["test_average_precision"].mean())}
    return out


cv_scores = compare(X_tr, y_tr, groups)
for k, v in cv_scores.items():
    print(f"{k:9s} CV ROC AUC {v['roc_auc']:.3f}  PR AUC {v['ap']:.3f}")
check(close(cv_scores["baseline"]["roc_auc"], 0.5), "the baseline ranks at chance (ROC AUC 0.5)")
check(cv_scores["model"]["ap"] > 2 * cv_scores["baseline"]["ap"], "the model's PR AUC is more than double the baseline's")
check(abs(cv_scores["model"]["roc_auc"] - test_auc) < 0.05, "CV agrees with the future test month (within 0.05): the CV mirrors the test")


# %% TODO 4: a cost-based threshold   (Lessons 5.5, 9.2)
COST_OFFER, COST_LOST = 20, 100  # an offer sent to someone who'd stay wastes $20; a churner we miss costs $100


def best_threshold(y_true, p, cost_fp=COST_OFFER, cost_fn=COST_LOST):
    """Try thresholds 0.01, 0.02, ..., 0.99. Cost = cost_fp × (offers to non-churners) + cost_fn ×
    (churners we didn't contact). Return (best threshold, its total cost)."""
    y_true = np.asarray(y_true)
    best = (None, np.inf)
    for t in np.round(np.arange(0.01, 1.0, 0.01), 2):
        flag = p >= t
        cost = cost_fp * np.sum(flag & (y_true == 0)) + cost_fn * np.sum(~flag & (y_true == 1))
        if cost < best[1]:
            best = (float(t), float(cost))
    return best


thr, cost = best_threshold(y_te, p_te)
cost_half = COST_OFFER * np.sum((p_te >= 0.5) & (y_te == 0)) + COST_LOST * np.sum((p_te < 0.5) & (y_te == 1))
theory = COST_OFFER / (COST_OFFER + COST_LOST)
print(f"best threshold {thr:.2f} (theory for calibrated scores: {theory:.3f}); cost ${cost:,.0f} vs ${cost_half:,.0f} at 0.5")
check(cost < cost_half, "the cost-based threshold beats the default 0.5")
check(abs(thr - theory) < 0.1, "and lands near c_FP / (c_FP + c_FN) ≈ 0.167, because the scores are close to calibrated")


# %% TODO 5: a calibration check   (Lesson 5.6)
def calibration_report(y_true, p, n_bins=10):
    """Use calibration_curve(..., strategy="quantile") for the reliability points. Also return the
    expected calibration error ECE = Σ over bins of (share of rows in bin) × |observed − predicted|,
    and the Brier scores of p and of always predicting the base rate."""
    y_true = np.asarray(y_true)
    frac_pos, mean_pred = calibration_curve(y_true, p, n_bins=n_bins, strategy="quantile")
    bins = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    which = np.clip(np.searchsorted(bins[1:-1], p, side="right"), 0, n_bins - 1)
    ece = sum(np.mean(which == b) * abs(y_true[which == b].mean() - p[which == b].mean())
              for b in range(n_bins) if np.any(which == b))
    return {"frac_pos": frac_pos, "mean_pred": mean_pred, "ece": float(ece),
            "brier": float(brier_score_loss(y_true, p)),
            "brier_base": float(brier_score_loss(y_true, np.full(len(y_true), y_true.mean())))}


cal = calibration_report(y_te, p_te)
print(f"ECE {cal['ece']:.3f}   Brier {cal['brier']:.4f} vs base-rate Brier {cal['brier_base']:.4f}")
check(cal["ece"] < 0.05, "well calibrated: predicted and observed churn differ by < 5 points on average")
check(cal["brier"] < cal["brier_base"], "the probabilities beat the base rate on the Brier score")

plt.figure(figsize=(4.6, 4.2))
plt.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
plt.plot(cal["mean_pred"], cal["frac_pos"], "o-", label="HistGradientBoosting")
plt.xlabel("predicted churn probability"); plt.ylabel("observed churn rate"); plt.legend()
plt.title("Reliability curve, test month"); plt.tight_layout(); show()


# %% TODO 6: permutation importance on the test month   (Lesson 8.1)
def top_features(fitted, X, y, n_repeats=5):
    """permutation_importance with scoring="roc_auc" and random_state=0. Return a pandas Series of
    importances_mean indexed by column name, sorted from most to least important."""
    imp = permutation_importance(fitted, X, y, scoring="roc_auc", n_repeats=n_repeats, random_state=0)
    return pd.Series(imp.importances_mean, index=X.columns).sort_values(ascending=False)


imp = top_features(model, X_te, y_te)
print(imp.round(4))
check(imp.index[0] == "days_since_login", "the strongest true driver, days since last login, ranks first")
check(abs(imp["favorite_color"]) < 0.01, "the pure-noise column barely matters (|importance| < 0.01)")


# %% TODO 7: a model card   (Lessons 8.3, 9.1, 9.3)
def model_card(cv_scores, test_auc, test_ap, thr, cal, imp):
    """Return a dict with keys: name, intended_use, data, metrics, decision_threshold, calibration,
    top_features (the 3 most important names), limitations (a list of at least 2 strings)."""
    return {
        "name": "churn-30d-hgb v1",
        "intended_use": "Rank active customers by 30-day churn risk to target retention offers; not for pricing or credit decisions.",
        "data": "Synthetic monthly snapshots; trained on months 1–2, tested on month 3; leaky column cancel_request_logged removed.",
        "metrics": {"cv_roc_auc": round(cv_scores["model"]["roc_auc"], 3), "cv_pr_auc": round(cv_scores["model"]["ap"], 3),
                    "baseline_pr_auc": round(cv_scores["baseline"]["ap"], 3),
                    "test_roc_auc": round(test_auc, 3), "test_pr_auc": round(test_ap, 3)},
        "decision_threshold": {"value": thr, "cost_offer": COST_OFFER, "cost_lost_customer": COST_LOST},
        "calibration": {"ece": round(cal["ece"], 3), "brier": round(cal["brier"], 4)},
        "top_features": list(imp.index[:3]),
        "limitations": [
            "Predicts who will churn, not who an offer will keep: use an uplift test before scaling offers (Lesson 9.2).",
            "One test month only; retrain monthly and monitor drift in usage and login features (Lesson 8.6).",
            "Synthetic data: the importances describe this simulation, not a real business.",
        ],
    }


card = model_card(cv_scores, test_auc, test_ap, thr, cal, imp)
need = ["name", "intended_use", "data", "metrics", "decision_threshold", "calibration", "top_features", "limitations"]
check(all(k in card for k in need), "the card has every section")
check(len(card["top_features"]) == 3 and card["top_features"][0] == imp.index[0], "it lists the top 3 features")
check(isinstance(card["limitations"], list) and len(card["limitations"]) >= 2, "and at least two honest limitations")
import json
print(json.dumps(card, indent=2, ensure_ascii=False))

# %% [markdown]
# ## 🎓 Done
# You framed a decision, removed a leak, validated on the future, beat a baseline, chose a threshold
# from money, checked the probabilities, explained the model and wrote down its limits. That's the whole
# course in one notebook. Next: swap the synthetic data for a real dataset you care about and publish it.

# %% Optional: compare with LightGBM if it's installed (not required)
try:
    import lightgbm as lgb
except (ImportError, OSError):  # OSError: installed but its OpenMP library is missing (common on macOS)
    print("LightGBM not available; skipping the optional comparison.")
else:
    Xl_tr, Xl_te = X_tr.copy(), X_te.copy()
    for c in CAT:
        cats = pd.CategoricalDtype(sorted(X_tr[c].unique()))
        Xl_tr[c], Xl_te[c] = Xl_tr[c].astype(cats), Xl_te[c].astype(cats)
    lgbm = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, verbose=-1, random_state=0)
    lgbm.fit(Xl_tr, y_tr)
    print(f"LightGBM test ROC AUC {roc_auc_score(y_te, lgbm.predict_proba(Xl_te)[:, 1]):.3f} vs HGB {test_auc:.3f}")
