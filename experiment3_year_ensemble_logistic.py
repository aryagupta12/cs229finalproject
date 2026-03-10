"""
Experiment 3 — Year-Ensemble Logistic Regression
-------------------------------------------------
Train one logistic regression per post-release year (Year 1, Year 2, Year 3),
each learning its own coefficients from pre-release features only.
Aggregate the three year-specific probabilities into a single 3-year risk score:

    P(recidivate within 3 yrs) = 1 − (1 − p₁)(1 − p₂)(1 − p₃)

This captures temporal structure without a pooled time_bin feature, allowing
each year's model to independently weight risk factors.
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
)
from sklearn.calibration import calibration_curve


# ------------------------------------------------------------------ #
#  Column definitions
# ------------------------------------------------------------------ #

PRE_RELEASE_COLS = [
    "Gender",
    "Race",
    "Age_at_Release",
    "Residence_PUMA",
    "Gang_Affiliated",
    "Supervision_Risk_Score_First",
    "Supervision_Level_First",
    "Education_Level",
    "Dependents",
    "Prison_Offense",
    "Prison_Years",
    "Prior_Arrest_Episodes_Felony",
    "Prior_Arrest_Episodes_Misd",
    "Prior_Arrest_Episodes_Violent",
    "Prior_Arrest_Episodes_Property",
    "Prior_Arrest_Episodes_Drug",
    "_v1",
    "Prior_Arrest_Episodes_DVCharges",
    "Prior_Arrest_Episodes_GunCharges",
    "Prior_Conviction_Episodes_Felony",
    "Prior_Conviction_Episodes_Misd",
    "Prior_Conviction_Episodes_Viol",
    "Prior_Conviction_Episodes_Prop",
    "Prior_Conviction_Episodes_Drug",
    "_v2",
    "_v3",
    "_v4",
    "Prior_Revocations_Parole",
    "Prior_Revocations_Probation",
    "Condition_MH_SA",
    "Condition_Cog_Ed",
    "Condition_Other",
]

POST_RELEASE_COLS = [
    "Violations_ElectronicMonitoring",
    "Violations_Instruction",
    "Violations_FailToReport",
    "Violations_MoveWithoutPermission",
    "Delinquency_Reports",
    "Program_Attendances",
    "Program_UnexcusedAbsences",
    "Residence_Changes",
    "Avg_Days_per_DrugTest",
    "DrugTests_THC_Positive",
    "DrugTests_Cocaine_Positive",
    "DrugTests_Meth_Positive",
    "DrugTests_Other_Positive",
    "Percent_Days_Employed",
    "Jobs_Per_Year",
    "Employment_Exempt",
]

FEATURE_COLS = PRE_RELEASE_COLS + POST_RELEASE_COLS

TARGET_COLS = [
    "Recidivism_Arrest_Year1",
    "Recidivism_Arrest_Year2",
    "Recidivism_Arrest_Year3",
]

TARGET_3YR = "Recidivism_Within_3years"


# ------------------------------------------------------------------ #
#  Pipeline builder
# ------------------------------------------------------------------ #

def build_pipeline(categorical_features: list, numeric_features: list) -> Pipeline:
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ])


# ------------------------------------------------------------------ #
#  Evaluation helpers
# ------------------------------------------------------------------ #

def print_calibration_curve(y_true, y_prob, n_bins=10):
    frac_pos, mean_pred = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="quantile"
    )
    print("\nCalibration Curve  (mean predicted prob → actual positive rate)")
    print(f"  {'Predicted':>10}  {'Actual':>8}  {'Δ':>7}")
    for mp, fp in zip(mean_pred, frac_pos):
        print(f"  {mp:>10.3f}  {fp:>8.3f}  {fp - mp:>+7.3f}")
    print("  (Δ > 0 → under-predicts risk; Δ < 0 → over-predicts)")


def print_fairness(results_df: pd.DataFrame, group_col: str):
    print(f"\nFairness — FPR / FNR by {group_col}")
    print(f"  {'Group':<35}  {'N':>6}  {'%Pos':>6}  {'FPR':>6}  {'FNR':>6}")
    for group, grp in results_df.groupby(group_col):
        yt, yp = grp["y_true"], grp["y_pred"]
        if len(yt) < 5:
            continue
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
        fnr = fn / (fn + tp) if (fn + tp) > 0 else float("nan")
        print(f"  {str(group):<35}  {len(yt):>6}  {yt.mean()*100:>5.1f}%  {fpr:>6.3f}  {fnr:>6.3f}")


def print_metrics(label: str, y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n{'=' * 55}")
    print(f"{label}")
    print(f"{'=' * 55}")
    print(f"Accuracy          : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision         : {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall            : {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"ROC AUC           : {roc_auc_score(y_true, y_prob):.4f}")
    print(f"Avg Precision(AP) : {average_precision_score(y_true, y_prob):.4f}")
    print(f"Brier Score       : {brier_score_loss(y_true, y_prob):.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TN={cm[0,0]:>5}  FP={cm[0,1]:>5}")
    print(f"  FN={cm[1,0]:>5}  TP={cm[1,1]:>5}")


# ------------------------------------------------------------------ #
#  Coefficient comparison across years
# ------------------------------------------------------------------ #

def print_top_coefficients_by_year(models: dict, feature_names: list, top_n: int = 10):
    print(f"\n{'=' * 75}")
    print(f"TOP {top_n} COEFFICIENTS BY YEAR (absolute value)")
    print(f"{'=' * 75}")

    coef_frames = {}
    for year, model in models.items():
        coefs = model.named_steps["classifier"].coef_[0]
        df = pd.DataFrame({"feature": feature_names, f"coef_yr{year}": coefs})
        df[f"abs_yr{year}"] = df[f"coef_yr{year}"].abs()
        coef_frames[year] = df.set_index("feature")

    combined = pd.concat([coef_frames[y][[f"coef_yr{y}"]] for y in [1, 2, 3]], axis=1)
    combined["max_abs"] = combined.abs().max(axis=1)
    top = combined.nlargest(top_n, "max_abs").drop(columns="max_abs")

    print(f"  {'Feature':<55}  {'Year 1':>8}  {'Year 2':>8}  {'Year 3':>8}")
    print(f"  {'-'*55}  {'-'*8}  {'-'*8}  {'-'*8}")
    for feat, row in top.iterrows():
        print(
            f"  {str(feat):<55}  {row['coef_yr1']:>+8.4f}  "
            f"{row['coef_yr2']:>+8.4f}  {row['coef_yr3']:>+8.4f}"
        )


# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #

def main():
    print("Loading dataset...")
    df = pd.read_csv("nij_training.csv")
    df["person_id"] = df.index

    required_cols = ["person_id"] + FEATURE_COLS + TARGET_COLS + [TARGET_3YR]
    df = df[required_cols].copy().dropna(subset=TARGET_COLS)

    print(f"Dataset shape: {df.shape}")

    # Person-level split — no person appears in both train and test
    unique_ids = df["person_id"].unique()
    rng = np.random.default_rng(42)
    rng.shuffle(unique_ids)
    n_test    = int(len(unique_ids) * 0.2)
    test_ids  = set(unique_ids[:n_test])
    train_ids = set(unique_ids[n_test:])

    train_df = df[df["person_id"].isin(train_ids)].copy()
    test_df  = df[df["person_id"].isin(test_ids)].copy()

    print(f"Train persons: {len(train_ids)}  |  Test persons: {len(test_ids)}")

    # Infer categorical / numeric from training data
    X_sample = train_df[FEATURE_COLS]
    categorical_features = X_sample.select_dtypes(include="object").columns.tolist()
    numeric_features     = X_sample.select_dtypes(exclude="object").columns.tolist()

    print(f"\nFeature set: {len(FEATURE_COLS)} total  "
          f"({len(numeric_features)} numeric, {len(categorical_features)} categorical)")
    print(f"  Pre-release : {len(PRE_RELEASE_COLS)}")
    print(f"  Post-release: {len(POST_RELEASE_COLS)}")

    # ---------------------------------------------------------------- #
    #  Train one model per year
    # ---------------------------------------------------------------- #
    models       = {}
    year_results = {}   # person_id → p_year for aggregation

    print("\nTraining year-specific models...")
    for year, target_col in zip([1, 2, 3], TARGET_COLS):
        year_train = train_df[FEATURE_COLS + [target_col]].dropna(subset=[target_col])
        year_test  = test_df[FEATURE_COLS + [target_col, "person_id"]].dropna(subset=[target_col])

        X_tr = year_train[FEATURE_COLS]
        y_tr = year_train[target_col].map({"Yes": 1, "No": 0}).astype(int)

        X_te = year_test[FEATURE_COLS]
        y_te = year_test[target_col].map({"Yes": 1, "No": 0}).astype(int)

        model = build_pipeline(categorical_features, numeric_features)
        model.fit(X_tr, y_tr)
        models[year] = model

        y_prob = model.predict_proba(X_te)[:, 1]
        y_pred = model.predict(X_te)

        # Store per-person probabilities for aggregation
        pid_series = year_test["person_id"].values
        year_results[year] = pd.Series(y_prob, index=pid_series, name=f"p{year}")

        print_metrics(
            f"YEAR {year} MODEL — predicting Recidivism_Arrest_Year{year}",
            y_te, y_pred, y_prob,
        )

        # Fairness per year
        fairness_df = test_df.loc[year_test.index, ["Race", "Gender"]].copy()
        fairness_df = fairness_df.reset_index(drop=True)
        fairness_df["y_true"] = y_te.values
        fairness_df["y_pred"] = y_pred
        print_fairness(fairness_df, "Race")
        print_fairness(fairness_df, "Gender")

    # ---------------------------------------------------------------- #
    #  3-year aggregation via complement rule
    # ---------------------------------------------------------------- #
    person_probs = pd.concat(year_results.values(), axis=1).dropna()
    person_probs["p_3yr"] = (
        1 - (1 - person_probs["p1"]) * (1 - person_probs["p2"]) * (1 - person_probs["p3"])
    )

    y_true_3yr = (
        test_df.set_index("person_id")[TARGET_3YR]
        .map({"Yes": 1, "No": 0})
    )
    merged = person_probs.join(y_true_3yr).dropna()

    y_true = merged[TARGET_3YR].astype(int)
    y_prob_3yr = merged["p_3yr"].values
    y_pred_3yr = (y_prob_3yr >= 0.5).astype(int)

    print_metrics(
        "3-YEAR AGGREGATED  P(3yr) = 1 − (1−p₁)(1−p₂)(1−p₃)",
        y_true, y_pred_3yr, y_prob_3yr,
    )
    print_calibration_curve(y_true, y_prob_3yr)

    # Fairness on aggregated 3-year predictions
    demo = test_df.set_index("person_id")[["Race", "Gender"]]
    agg_results = merged[[]].copy()
    agg_results["y_true"] = y_true
    agg_results["y_pred"] = y_pred_3yr
    agg_results = agg_results.join(demo).dropna().reset_index(drop=True)
    print_fairness(agg_results, "Race")
    print_fairness(agg_results, "Gender")

    # ---------------------------------------------------------------- #
    #  Coefficient comparison across years
    # ---------------------------------------------------------------- #
    ohe = models[1].named_steps["preprocessor"].named_transformers_["cat"]["encoder"]
    cat_feat_names   = ohe.get_feature_names_out(categorical_features).tolist()
    all_feature_names = numeric_features + cat_feat_names

    print_top_coefficients_by_year(models, all_feature_names, top_n=15)


if __name__ == "__main__":
    main()
