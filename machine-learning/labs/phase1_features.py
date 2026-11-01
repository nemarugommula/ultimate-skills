# %% [markdown]
# # Lab 1: Data & features, from a messy table to a leak-proof pipeline 🧹
#
# You'll take a small, messy customer table (built right here, no downloads) and
# prepare it for a churn model, the way you would at work: fill missing incomes
# and keep a "was missing" flag, one-hot encode a small category, measure the
# target-encoding leak on a high-cardinality column and fix it with
# cross-fitting, scale features for kNN, engineer a ratio feature, and finally
# put everything in one ColumnTransformer Pipeline scored with cross_val_score.
#
# How to run: cell by cell in VS Code ("Run Cell" above each `# %%`), or the
# whole file with `python3 phase1_features.py`. The file stops at the first
# TODO you haven't filled in. Fill each TODO until its ✅ checks pass.
# Setup (once): `pip install -r requirements.txt`. Solutions: labs/solutions/
# Lessons to reread if stuck: 1.1 (EDA), 1.2 (missing values), 1.3 (encoding),
# 1.4 (scaling), 1.5 (feature engineering), 1.6 (pipelines)

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

rng = np.random.default_rng(42)  # one seeded random generator → reproducible results


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


def auc_of(model, X, y):
    """Area under the ROC curve: 0.5 = random ranking, 1.0 = every churner ranked above every stayer."""
    return roc_auc_score(y, model.predict_proba(X)[:, 1])


# %% The data: 3,000 customers of a phone company (synthetic, so we know the truth)
n = 3000
plan = rng.choice(["basic", "standard", "premium"], n, p=[0.5, 0.3, 0.2])
tenure = rng.integers(1, 73, n)                                   # months as a customer
call_rate = rng.gamma(2.0, 0.08, n)                               # support calls per month (hidden)
calls = rng.poisson(call_rate * tenure)                           # total support calls so far
charge = np.round(np.select([plan == "basic", plan == "standard"], [25, 55], 95) + rng.normal(0, 6, n), 2)
income = np.round(rng.lognormal(10.6, 0.5, n), -2)                # dollars a year: big numbers
city = rng.choice([f"city_{i:03d}" for i in range(600)], n)       # 600 cities, about 4 rows each, no real effect
logit = -1.6 + 12.0 * calls / tenure - 0.03 * tenure + 0.025 * (charge - 50)
churn = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
income = np.where(rng.random(n) < np.where(churn == 1, 0.35, 0.05), np.nan, income)  # churners skip the question

df = pd.DataFrame({"customer_id": np.arange(100001, 100001 + n), "plan": plan, "tenure": tenure,
                   "calls": calls, "charge": charge, "income": income, "city": city})
y = churn
print(df.head())
print(f"churn rate {y.mean():.1%}, missing incomes {df['income'].isna().mean():.1%}")
check(df.shape == (3000, 7), "the table has 3,000 rows and 7 columns")

X_tr, X_te, y_tr, y_te = train_test_split(df, y, test_size=0.3, random_state=0, stratify=y)


# %% TODO 1: fill missing incomes with the median, and keep a "was missing" flag   (Lesson 1.2)
def impute_income(train_col, test_col):
    """train_col, test_col: 1-column DataFrames with NaNs. Returns (train_out, test_out), each a NumPy array
    with 2 columns: [income with blanks filled by the TRAINING median, 1.0 if it was missing else 0.0]."""
    # TODO 1: make a SimpleImputer(strategy="median", add_indicator=True), fit_transform the TRAIN column,
    #         then only transform the TEST column (so the test rows never influence the median).
    raise NotImplementedError("TODO 1: impute_income")


inc_tr, inc_te = impute_income(X_tr[["income"]], X_te[["income"]])
check(inc_tr.shape == (len(X_tr), 2) and not np.isnan(inc_tr).any(), "2 columns, no blanks left")
check(close(inc_tr[inc_tr[:, 1] == 1, 0], np.nanmedian(X_tr["income"])), "blanks were filled with the training median")
flag_rate = [inc_tr[y_tr == c, 1].mean() for c in (0, 1)]
print(f"income missing: {flag_rate[0]:.0%} of stayers vs {flag_rate[1]:.0%} of churners")
check(flag_rate[1] > 3 * flag_rate[0], "missingness is information: churners leave income blank far more often")

base = ["tenure", "calls", "charge"]
no_flag = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(np.c_[X_tr[base], inc_tr[:, :1]], y_tr)
with_flag = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(np.c_[X_tr[base], inc_tr], y_tr)
a0, a1 = auc_of(no_flag, np.c_[X_te[base], inc_te[:, :1]], y_te), auc_of(with_flag, np.c_[X_te[base], inc_te], y_te)
print(f"test AUC without the flag {a0:.3f}, with it {a1:.3f}")
check(a1 > a0 + 0.02, "the missing indicator improves test AUC")


# %% TODO 2: one-hot encode the plan   (Lesson 1.3)
def one_hot_plan(train_col, test_col):
    """Returns (encoder, train_array, test_array): a fitted OneHotEncoder (dense output, unknown plans
    ignored) and the encoded train/test columns."""
    # TODO 2: OneHotEncoder(handle_unknown="ignore", sparse_output=False); fit_transform train, transform test.
    raise NotImplementedError("TODO 2: one_hot_plan")


enc, plan_tr, plan_te = one_hot_plan(X_tr[["plan"]], X_te[["plan"]])
print(enc.get_feature_names_out())
check(plan_tr.shape[1] == 3 and close(plan_tr.sum(axis=1), 1), "3 columns, exactly one 1 per row")
check(list(enc.get_feature_names_out()) == ["plan_basic", "plan_premium", "plan_standard"], "names in alphabetical order")
check(close(enc.transform(pd.DataFrame({"plan": ["platinum"]})), [[0, 0, 0]]), "an unseen plan becomes all zeros")


# %% TODO 3: naive target encoding of city, and the leak it causes   (Lesson 1.3)
def naive_target_encode(train_col, y_train, test_col):
    """Replace each city by the mean churn of ITS training rows (no smoothing, the row's own label included).
    Cities never seen in training get the global training mean. Returns (train_values, test_values) as arrays."""
    # TODO 3: means = pd.Series(y_train, index=train_col.index).groupby(train_col).mean()   (one number per city)
    #         train values: train_col.map(means); test values: test_col.map(means), with unseen cities → np.mean(y_train)
    #         Return both as NumPy arrays (.to_numpy()).
    raise NotImplementedError("TODO 3: naive_target_encode")


te_tr, te_te = naive_target_encode(X_tr["city"], y_tr, X_te["city"])
check(te_tr.min() >= 0 and te_tr.max() <= 1 and not np.isnan(te_te).any(), "encodings are churn rates in [0, 1], no blanks")
leaky = LogisticRegression(max_iter=1000).fit(np.c_[X_tr[base], te_tr], y_tr)
leak_tr, leak_te = auc_of(leaky, np.c_[X_tr[base], te_tr], y_tr), auc_of(leaky, np.c_[X_te[base], te_te], y_te)
honest_base = LogisticRegression(max_iter=1000).fit(X_tr[base], y_tr)
base_te = auc_of(honest_base, X_te[base], y_te)
print(f"naive target encoding: train AUC {leak_tr:.3f}, test AUC {leak_te:.3f} (without city: test {base_te:.3f})")
check(leak_tr - leak_te > 0.1, "the leak: train AUC is far above test AUC")
check(leak_te < base_te, "and the leaky column makes the test score worse than dropping it")


# %% TODO 4: fix it with cross-fitting   (Lesson 1.3)
def crossfit_target_encode(train_col, y_train, test_col):
    """Use sklearn's TargetEncoder: fit_transform on train (cross-fitted, 5 folds), transform on test.
    Returns (train_values, test_values) as 1-D arrays."""
    # TODO 4: enc = TargetEncoder(target_type="binary"). On train call fit_transform (NOT fit then transform:
    #         fit_transform is the cross-fitted version). On test call transform. Encoders want 2-D input:
    #         pass train_col.to_frame(); return 1-D arrays with .ravel().
    raise NotImplementedError("TODO 4: crossfit_target_encode")


cf_tr, cf_te = crossfit_target_encode(X_tr["city"], y_tr, X_te["city"])
fixed = LogisticRegression(max_iter=1000).fit(np.c_[X_tr[base], cf_tr], y_tr)
fix_tr, fix_te = auc_of(fixed, np.c_[X_tr[base], cf_tr], y_tr), auc_of(fixed, np.c_[X_te[base], cf_te], y_te)
print(f"cross-fitted: train AUC {fix_tr:.3f}, test AUC {fix_te:.3f}")
check(abs(fix_tr - fix_te) < 0.05, "train and test AUC now agree: the training score is honest")
check(fix_te > leak_te + 0.03, "and the test score recovers")


# %% TODO 5: scaling for kNN   (Lesson 1.4)
def knn_accuracy(X_train, y_train, X_test, y_test, scale):
    """Test accuracy of KNeighborsClassifier(n_neighbors=15), with a StandardScaler in front if scale is True.
    Use make_pipeline so the scaler is fitted on the training rows only."""
    # TODO 5: build make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15)) when scale is True,
    #         or make_pipeline(KNeighborsClassifier(n_neighbors=15)) when it's False; fit on train; return .score on test.
    raise NotImplementedError("TODO 5: knn_accuracy")


cols = base + ["income"]                                          # income is in dollars: huge numbers
m_tr, m_te = X_tr["income"].notna().to_numpy(), X_te["income"].notna().to_numpy()   # rows where income is known
raw = knn_accuracy(X_tr.loc[m_tr, cols], y_tr[m_tr], X_te.loc[m_te, cols], y_te[m_te], scale=False)
scaled = knn_accuracy(X_tr.loc[m_tr, cols], y_tr[m_tr], X_te.loc[m_te, cols], y_te[m_te], scale=True)
always_stay = 1 - y_te[m_te].mean()
print(f"kNN accuracy: raw units {raw:.3f}, standardized {scaled:.3f} (always 'stays' = {always_stay:.3f})")
check(raw < always_stay, "in raw units kNN is worse than always predicting 'stays': income drowns out everything")
check(scaled > raw + 0.03, "standardizing fixes it")


# %% TODO 6: engineer a ratio feature   (Lesson 1.5)
def add_call_rate(frame):
    """Return a COPY of frame with a new column calls_per_month = calls / tenure."""
    # TODO 6: copy the frame (frame.copy()), add the column, return the copy. Tenure is always ≥ 1, so no division by 0.
    raise NotImplementedError("TODO 6: add_call_rate")


Xr_tr, Xr_te = add_call_rate(X_tr), add_call_rate(X_te)
check("calls_per_month" in Xr_tr and "calls_per_month" not in X_tr, "new column added to a copy")
check(close(Xr_tr["calls_per_month"].iloc[0], X_tr["calls"].iloc[0] / X_tr["tenure"].iloc[0]), "calls_per_month = calls / tenure")
lin = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
auc_raw = auc_of(lin.fit(X_tr[base], y_tr), X_te[base], y_te)
auc_ratio = auc_of(lin.fit(Xr_tr[base + ["calls_per_month"]], y_tr), Xr_te[base + ["calls_per_month"]], y_te)
print(f"logistic regression test AUC: raw columns {auc_raw:.3f}, + calls_per_month {auc_ratio:.3f}")
check(auc_ratio > auc_raw + 0.03, "a linear model can't divide; handing it the ratio helps")

plt.figure(figsize=(6, 3.5))
for c, lab in [(0, "stayed"), (1, "churned")]:
    plt.hist(Xr_tr.loc[y_tr == c, "calls_per_month"], bins=40, range=(0, 1), alpha=0.6, label=lab, density=True)
plt.xlabel("support calls per month"); plt.legend(); plt.title("The ratio separates churners"); plt.tight_layout(); show()


# %% TODO 7: everything in one leak-proof pipeline   (Lesson 1.6)
def build_pipeline():
    """A make_pipeline(make_column_transformer(...), LogisticRegression(max_iter=1000)) where:
       - tenure, calls, charge, calls_per_month: StandardScaler
       - income: SimpleImputer(median, add_indicator=True) then StandardScaler
       - plan: OneHotEncoder(handle_unknown="ignore")
       - city: TargetEncoder(target_type="binary")
       - customer_id: dropped (leave it to remainder="drop")"""
    # TODO 7: prep = make_column_transformer((transformer, [columns]), ..., remainder="drop")
    #         with the four entries described above, then return make_pipeline(prep, LogisticRegression(max_iter=1000)).
    raise NotImplementedError("TODO 7: build_pipeline")


pipe = build_pipeline()
full = add_call_rate(df)
scores = cross_val_score(pipe, full, y, cv=5, scoring="roc_auc")   # every step refitted inside every fold
print(f"5-fold CV AUC {scores.mean():.3f} ± {scores.std():.3f}")
pipe.fit(full, y)
names = list(pipe[:-1].get_feature_names_out())
print(names)
check(len(names) == 10 and not any("customer_id" in s for s in names), "10 model inputs, customer_id dropped")
check(any("missingindicator_income" in s for s in names), "the income missing flag made it into the model")
check(scores.mean() > max(auc_ratio, a1) - 0.02, "the full pipeline is at least as good as the hand-made pieces")
check(scores.std() < 0.05, "and stable across folds")

# %% Optional: gradient boosting handles NaN and small categories natively (no imputer, no encoder)
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402  (installed with scikit-learn)

# Native categorical splits allow at most 255 categories per column (one bin each), so city (600 values) stays out.
hgb_X = full[["tenure", "calls", "charge", "calls_per_month", "income", "plan"]].astype({"plan": "category"})
hgb = HistGradientBoostingClassifier(categorical_features="from_dtype", random_state=0)
with threadpool_limits(limits=1):  # one thread is plenty for 3,000 rows (and avoids slow thread start-up on laptops)
    hgb_auc = cross_val_score(hgb, hgb_X, y, cv=5, scoring="roc_auc").mean()
print(f"HistGradientBoosting 5-fold CV AUC {hgb_auc:.3f} (income NaNs and the plan text handled inside the model)")

print("\n🎉 Lab 1 complete: you can turn a messy table into an honest, deployable pipeline.")
