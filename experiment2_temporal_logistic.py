import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
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


def build_temporal_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Convert original dataset into person-year (long) format, preserving person_id."""
    records = []
    base = df[FEATURE_COLS + ["person_id"]].copy()

    for year, target_col in enumerate(TARGET_COLS, start=1):
        year_df = base.copy()
        year_df["time_bin"] = year
        year_df["recidivism"] = df[target_col].values
        records.append(year_df)

    temporal_df = pd.concat(records, ignore_index=True)
    return temporal_df


def build_pipeline(categorical_features: list, numeric_features: list) -> Pipeline:
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    return pipeline


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


def print_fairness_by_group(results_df: pd.DataFrame, group_col: str):
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


def evaluate_year_specific(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    test_temporal: pd.DataFrame,
) -> None:
    """Evaluate the model on year-specific recidivism predictions."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n" + "=" * 55)
    print("YEAR-SPECIFIC EVALUATION (per person-year row)")
    print("=" * 55)
    print(f"Accuracy          : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision         : {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"Recall            : {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"ROC AUC           : {roc_auc_score(y_test, y_prob):.4f}")
    print(f"Avg Precision(AP) : {average_precision_score(y_test, y_prob):.4f}")
    print(f"Brier Score       : {brier_score_loss(y_test, y_prob):.4f}")
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(f"  TN={cm[0,0]:>5}  FP={cm[0,1]:>5}")
    print(f"  FN={cm[1,0]:>5}  TP={cm[1,1]:>5}")

    print_calibration_curve(y_test, y_prob)

    results_df = test_temporal[["Race", "Gender"]].copy().reset_index(drop=True)
    results_df["y_true"] = y_test.values
    results_df["y_pred"] = y_pred
    print_fairness_by_group(results_df, "Race")
    print_fairness_by_group(results_df, "Gender")


def evaluate_3year_aggregated(
    model: Pipeline,
    test_df: pd.DataFrame,
    all_features: list,
) -> None:
    """
    Aggregate year-specific probabilities into a single 3-year risk score using
    the complement rule: P(recidivate in 3yr) = 1 − (1−p1)(1−p2)(1−p3).
    Evaluate against Recidivism_Within_3years for apples-to-apples comparison
    with the static model.
    """
    records = []
    for year in [1, 2, 3]:
        yr_df = test_df[FEATURE_COLS + ["person_id"]].copy()
        yr_df["time_bin"] = year
        records.append(yr_df)

    long_df = pd.concat(records, ignore_index=True)
    long_df["y_prob"] = model.predict_proba(long_df[all_features])[:, 1]
    long_df["year"] = [y for y in [1, 2, 3] for _ in range(len(test_df))]

    person_probs = long_df.pivot_table(
        index="person_id", columns="year", values="y_prob"
    ).rename(columns={1: "p1", 2: "p2", 3: "p3"})

    person_probs["p_3yr"] = 1 - (1 - person_probs["p1"]) * (1 - person_probs["p2"]) * (1 - person_probs["p3"])

    y_true_3yr = test_df.set_index("person_id")[TARGET_3YR].map({"Yes": 1, "No": 0})
    merged = person_probs.join(y_true_3yr).dropna()

    y_true = merged[TARGET_3YR].astype(int)
    y_prob  = merged["p_3yr"].values
    y_pred  = (y_prob >= 0.5).astype(int)

    print("\n" + "=" * 55)
    print("3-YEAR AGGREGATED EVALUATION (apples-to-apples vs static)")
    print("  P(3yr) = 1 − (1−p₁)(1−p₂)(1−p₃)")
    print("=" * 55)
    print(f"Accuracy          : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision         : {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall            : {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"ROC AUC           : {roc_auc_score(y_true, y_prob):.4f}")
    print(f"Avg Precision(AP) : {average_precision_score(y_true, y_prob):.4f}")
    print(f"Brier Score       : {brier_score_loss(y_true, y_prob):.4f}")
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(f"  TN={cm[0,0]:>5}  FP={cm[0,1]:>5}")
    print(f"  FN={cm[1,0]:>5}  TP={cm[1,1]:>5}")

    print_calibration_curve(y_true, y_prob)

    race_col  = test_df.set_index("person_id")["Race"]
    gender_col = test_df.set_index("person_id")["Gender"]
    results_df = merged[[]].copy()
    results_df["y_true"] = y_true
    results_df["y_pred"] = y_pred
    results_df["Race"]   = race_col
    results_df["Gender"] = gender_col
    results_df = results_df.dropna(subset=["Race", "Gender"])

    print_fairness_by_group(results_df.reset_index(drop=True), "Race")
    print_fairness_by_group(results_df.reset_index(drop=True), "Gender")


def print_time_bin_coefficient(model: Pipeline, numeric_features: list) -> None:
    time_bin_index = numeric_features.index("time_bin")
    coef = model.named_steps["classifier"].coef_[0][time_bin_index]
    print("\n" + "=" * 55)
    print("TIME BIN COEFFICIENT")
    print("=" * 55)
    print(f"time_bin coefficient: {coef:.4f}")
    if coef > 0:
        print("Interpretation: Risk of recidivism INCREASES with each later year.")
    elif coef < 0:
        print("Interpretation: Risk of recidivism DECREASES with each later year.")
    else:
        print("Interpretation: Time bin has no effect on recidivism risk.")


def main() -> None:
    print("Loading dataset...")
    df = pd.read_csv("nij_training.csv")

    df["person_id"] = df.index

    required_cols = ["person_id"] + FEATURE_COLS + TARGET_COLS + [TARGET_3YR]
    df = df[required_cols].copy()
    df = df.dropna(subset=TARGET_COLS)

    print(f"Original dataset shape: {df.shape}")

    # Person-level train/test split — ensures no person appears in both sets
    unique_ids = df["person_id"].unique()
    rng = np.random.default_rng(42)
    rng.shuffle(unique_ids)
    n_test    = int(len(unique_ids) * 0.2)
    test_ids  = set(unique_ids[:n_test])
    train_ids = set(unique_ids[n_test:])

    train_df = df[df["person_id"].isin(train_ids)].copy()
    test_df  = df[df["person_id"].isin(test_ids)].copy()

    print(f"Train persons: {len(train_ids)}  |  Test persons: {len(test_ids)}")

    print("Building person-year (temporal) dataset...")
    train_temporal = build_temporal_dataset(train_df).dropna(subset=["recidivism"])
    test_temporal  = build_temporal_dataset(test_df).dropna(subset=["recidivism"])

    print(f"Temporal train shape: {train_temporal.shape}")
    print(f"Temporal test shape : {test_temporal.shape}")

    all_features = FEATURE_COLS + ["time_bin"]

    X_train = train_temporal[all_features]
    y_train = train_temporal["recidivism"].map({"Yes": 1, "No": 0}).astype(int)

    X_test  = test_temporal[all_features]
    y_test  = test_temporal["recidivism"].map({"Yes": 1, "No": 0}).astype(int)

    # Infer types from training data (handles Yes/No, bucketed strings, and floats)
    categorical_features = X_train.select_dtypes(include="object").columns.tolist()
    numeric_features     = X_train.select_dtypes(exclude="object").columns.tolist()

    print(f"\nFeature set: {len(all_features)} total  "
          f"({len(numeric_features)} numeric, {len(categorical_features)} categorical)")
    print(f"  Pre-release : {len(PRE_RELEASE_COLS)}")
    print(f"  Post-release: {len(POST_RELEASE_COLS)}")
    print(f"  time_bin    : 1")
    print(f"\nTrain rows: {len(X_train)}  |  Test rows: {len(X_test)}")

    print("\nBuilding and training model...")
    model = build_pipeline(categorical_features, numeric_features)
    model.fit(X_train, y_train)
    print("Training complete.")

    evaluate_year_specific(model, X_test, y_test, test_temporal)
    evaluate_3year_aggregated(model, test_df, all_features)
    print_time_bin_coefficient(model, numeric_features)


if __name__ == "__main__":
    main()
