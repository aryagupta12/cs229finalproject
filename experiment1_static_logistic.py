import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
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


FEATURE_COLUMNS = [
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

TARGET = "Recidivism_Within_3years"


def print_calibration_curve(y_true, y_prob, n_bins=10):
    frac_pos, mean_pred = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="quantile"
    )
    print("\nCalibration Curve  (mean predicted prob → actual positive rate)")
    print(f"  {'Predicted':>10}  {'Actual':>8}  {'Δ':>7}")
    for mp, fp in zip(mean_pred, frac_pos):
        delta = fp - mp
        print(f"  {mp:>10.3f}  {fp:>8.3f}  {delta:>+7.3f}")
    print("  (Δ > 0 → model under-predicts risk; Δ < 0 → over-predicts)")


def print_fairness_by_group(results_df, group_col):
    print(f"\nFairness — FPR / FNR by {group_col}")
    print(f"  {'Group':<35}  {'N':>6}  {'%Pos':>6}  {'FPR':>6}  {'FNR':>6}")
    for group, grp in results_df.groupby(group_col):
        yt = grp["y_true"]
        yp = grp["y_pred"]
        if len(yt) < 5:
            continue
        cm = confusion_matrix(yt, yp, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
        fnr = fn / (fn + tp) if (fn + tp) > 0 else float("nan")
        pct_pos = yt.mean() * 100
        print(f"  {str(group):<35}  {len(yt):>6}  {pct_pos:>5.1f}%  {fpr:>6.3f}  {fnr:>6.3f}")


def main():
    df = pd.read_csv("nij_training.csv")

    X = df[FEATURE_COLUMNS]
    y = df[TARGET].map({"Yes": 1, "No": 0})

    # Dynamically determine types from actual data so string-valued columns
    # (Yes/No booleans, "X or more" ordinals, etc.) are never passed to StandardScaler.
    categorical_features = X.select_dtypes(include="object").columns.tolist()
    numeric_features = X.select_dtypes(exclude="object").columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # ------------------------------------------------------------------ #
    #  Core metrics
    # ------------------------------------------------------------------ #
    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    roc_auc   = roc_auc_score(y_test, y_prob)
    ap        = average_precision_score(y_test, y_prob)
    brier     = brier_score_loss(y_test, y_prob)
    cm        = confusion_matrix(y_test, y_pred)

    print("=" * 55)
    print("EXPERIMENT 1 — STATIC LOGISTIC REGRESSION")
    print("=" * 55)
    print(f"Accuracy          : {accuracy:.4f}")
    print(f"Precision         : {precision:.4f}")
    print(f"Recall            : {recall:.4f}")
    print(f"ROC AUC           : {roc_auc:.4f}")
    print(f"Avg Precision(AP) : {ap:.4f}  ← PR-AUC, better for imbalance")
    print(f"Brier Score       : {brier:.4f}  ← lower is better (0=perfect)")
    print("\nConfusion Matrix:")
    print(f"  TN={cm[0,0]:>5}  FP={cm[0,1]:>5}")
    print(f"  FN={cm[1,0]:>5}  TP={cm[1,1]:>5}")

    # ------------------------------------------------------------------ #
    #  Calibration
    # ------------------------------------------------------------------ #
    print_calibration_curve(y_test, y_prob)

    # ------------------------------------------------------------------ #
    #  Fairness
    # ------------------------------------------------------------------ #
    results_df = X_test[["Race", "Gender"]].copy().reset_index(drop=True)
    results_df["y_true"] = y_test.values
    results_df["y_pred"] = y_pred

    print_fairness_by_group(results_df, "Race")
    print_fairness_by_group(results_df, "Gender")

    # ------------------------------------------------------------------ #
    #  Top 15 coefficients
    # ------------------------------------------------------------------ #
    ohe = pipeline.named_steps["preprocessor"].named_transformers_["cat"]["encoder"]
    cat_feature_names = ohe.get_feature_names_out(categorical_features).tolist()
    all_feature_names = numeric_features + cat_feature_names

    coefficients = pipeline.named_steps["classifier"].coef_[0]
    coef_df = pd.DataFrame(
        {"feature": all_feature_names, "coefficient": coefficients}
    )
    coef_df["abs_coefficient"] = coef_df["coefficient"].abs()
    top15 = coef_df.nlargest(15, "abs_coefficient").reset_index(drop=True)

    print("\n" + "=" * 55)
    print("TOP 15 COEFFICIENTS (by absolute value)")
    print("=" * 55)
    for _, row in top15.iterrows():
        print(f"  {row['feature']:<55} {row['coefficient']:+.4f}")


if __name__ == "__main__":
    main()
