"""
freight_rate_predict.py

Trains XGBoost and CatBoost regressors on freight rate data, evaluates both
on a held-out test split, picks the better model, and generates:
  1. validation_predictions.csv — exactly load_id,predicted_rate (12,000 rows),
     ready for scorer.py's --predictions argument.
  2. a completed December chart-inputs file — same 7 columns as the provided
     template, with predicted_rate filled in, ready for scorer.py's
     --december-predictions argument.

Fixes applied vs. the original notebook:
  1. CatBoost training cell was incomplete (syntax error) -> completed properly.
  2. CatBoost test-set predictions were never generated -> added.
  3. CatBoost test-set predictions were never inverse-log-transformed -> added
     np.expm1() before evaluation.
  4. Validation predictions (both models) were never inverse-log-transformed ->
     added np.expm1() so predictions contain real dollar values, not log-space.
  5. `weight` column was cleaned (abs()) only on train-test data, not on the
     validation set -> now cleaned identically everywhere.
  6. Column-order mismatch between train and validation CatBoost frames ->
     fixed by always selecting columns via the same `FEATURE_COLS` list and
     passing cat_features by name (not position) to CatBoost's Pool.
  7. The final submission file must contain EXACTLY load_id,predicted_rate
     (scorer.py rejects extra columns) -> written as a separate clean file;
     a full diagnostics CSV (both models' predictions, low_confidence flag)
     is written alongside it for your own inspection, not for submission.

Usage:
    python Prediction.py \
        --train-csv train-test.csv \
        --validation-csv validation.csv \
        --output-csv validation_predictions.csv \
        --december-csv december-chart-inputs.csv \
        --december-output december-chart-inputs-completed.csv \
        [--test-size 0.2] [--random-state 42] [--metric MAE]
"""

import argparse
import sys

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from catboost import CatBoostRegressor, Pool


CAT_COLS = ["pickup", "delivery", "equipment"]
BASE_NUM_COLS = [
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
    "distance", "weight", "market_index", "quote_signal",
]
DATE_DERIVED_COLS = ["dow", "month", "day_of_year", "year"]
NUM_COLS = BASE_NUM_COLS + DATE_DERIVED_COLS
FEATURE_COLS = NUM_COLS + CAT_COLS
TARGET_COL = "posted_rate"
DECEMBER_TEMPLATE_COLS = ["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"]


# --------------------------------------------------------------------------- #
# Data preparation
# --------------------------------------------------------------------------- #

def load_and_clean(path: str) -> pd.DataFrame:
    """Load a CSV and apply the same cleaning/feature-engineering to any
    dataset (train-test or validation), so both stay in sync."""
    df = pd.read_csv(path)

    # Fix negative weights (sign error in source data) — NaNs left untouched.
    df["weight"] = df["weight"].abs()

    # Cast raw categorical/string columns cleanly.
    obj_cols = df.select_dtypes(include=["object", "string"]).columns
    obj_cols = [c for c in obj_cols if c in CAT_COLS]
    df[obj_cols] = df[obj_cols].astype("string")

    # Date-derived features.
    df["date"] = pd.to_datetime(df["date"])
    df["dow"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    df["year"] = df["date"].dt.year
    # df = df.drop(columns=["date"])

    return df


# --------------------------------------------------------------------------- #
# December chart-inputs handling
# --------------------------------------------------------------------------- #

def fill_december_features(december_path: str, train_df: pd.DataFrame) -> pd.DataFrame:
    """Load the December chart-inputs template (pickup, delivery, distance,
    equipment, weight, date, predicted_rate) and attach the feature columns
    the model needs but the template doesn't carry (lat/lon, market_index,
    quote_signal). Values are pulled from the historical rows in train_df
    that match the same pickup/delivery/equipment combination — a route-
    specific proxy, since market_index/quote_signal vary per-load rather
    than being a single market-wide daily value (verified: ~144/145 unique
    values among rows sharing the same date)."""
    dec_df = pd.read_csv(december_path)

    if list(dec_df.columns) != DECEMBER_TEMPLATE_COLS:
        print(f"WARNING: December file columns {list(dec_df.columns)} don't "
              f"match the expected template {DECEMBER_TEMPLATE_COLS}")

    pickup = dec_df["pickup"].iloc[0]
    delivery = dec_df["delivery"].iloc[0]
    equipment = dec_df["equipment"].iloc[0]

    route = train_df[
        (train_df["pickup"] == pickup)
        & (train_df["delivery"] == delivery)
        & (train_df["equipment"] == equipment)
    ]
    if route.empty:
        # fall back to pickup+delivery only (ignore equipment) if no exact match
        route = train_df[(train_df["pickup"] == pickup) & (train_df["delivery"] == delivery)]
    if route.empty:
        sys.exit(f"ERROR: no historical rows found for route {pickup} -> {delivery} "
                  f"in train-test data; cannot derive lat/lon/market_index/quote_signal.")

    dec_df["pickup_lat"] = route["pickup_lat"].iloc[0]
    dec_df["pickup_lon"] = route["pickup_lon"].iloc[0]
    dec_df["delivery_lat"] = route["delivery_lat"].iloc[0]
    dec_df["delivery_lon"] = route["delivery_lon"].iloc[0]
    dec_df["market_index"] = route["market_index"].median()
    dec_df["quote_signal"] = route["quote_signal"].median()

    dec_df["date"] = pd.to_datetime(dec_df["date"])
    dec_df["dow"] = dec_df["date"].dt.dayofweek
    dec_df["month"] = dec_df["date"].dt.month
    dec_df["day_of_year"] = dec_df["date"].dt.dayofyear
    dec_df["year"] = dec_df["date"].dt.year

    return dec_df


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

def evaluate(y_true, y_pred, name: str) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print(f"{name:10s} — MAE: {mae:8.2f} | RMSE: {rmse:8.2f} | "
          f"R2: {r2:.4f} | MAPE: {mape:.2f}%")
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Train XGBoost & CatBoost freight rate models, pick the "
                    "better one, and predict on a validation set."
    )
    parser.add_argument("--train-csv", required=True,
                         help="Path to train-test.csv (must include posted_rate).")
    parser.add_argument("--validation-csv", required=True,
                         help="Path to validation.csv (no posted_rate column).")
    parser.add_argument("--output-csv", default="validation_predictions.csv",
                         help="Where to write the FINAL submission file "
                              "(exactly load_id,predicted_rate — this is what "
                              "you hand in and what scorer.py --predictions expects).")
    parser.add_argument("--diagnostics-csv", default=None,
                         help="Optional: where to write a full diagnostics CSV "
                              "(both models' predictions, low_confidence flag, "
                              "all features). NOT for submission — for your own review.")
    parser.add_argument("--december-csv", default=None,
                         help="Path to the December chart-inputs template "
                              "(pickup,delivery,distance,equipment,weight,date,predicted_rate).")
    parser.add_argument("--december-output", default="december-chart-inputs-completed.csv",
                         help="Where to write the completed December file "
                              "(this is what scorer.py --december-predictions expects).")
    parser.add_argument("--test-size", type=float, default=0.2,
                         help="Fraction of train-test.csv held out for evaluation.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--metric", choices=["MAE", "RMSE", "R2", "MAPE"],
                         default="MAE",
                         help="Metric used to pick the better model (lower is "
                              "better, except R2 where higher is better).")
    args = parser.parse_args()

    # ---------------- Load & clean ---------------- #
    print(f"Loading train-test data from {args.train_csv} ...")
    df = load_and_clean(args.train_csv)

    if TARGET_COL not in df.columns:
        sys.exit(f"ERROR: '{TARGET_COL}' column not found in {args.train_csv}")

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    df_sorted = df.sort_values("date")
    split_idx = int(len(df_sorted) * (1 - args.test_size))
    train_idx = df_sorted.index[:split_idx]
    test_idx = df_sorted.index[split_idx:]

    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]

    # Train on log1p(target) — posted_rate is right-skewed (a small number of
    # very expensive loads). Predictions are inverse-transformed with expm1()
    # before every evaluation/output step below.
    y_train_log = np.log1p(y_train)

    # ---------------- XGBoost ---------------- #
    print("\nPreparing XGBoost data...")
    X_train_xgb = X_train.copy()
    for col in CAT_COLS:
        X_train_xgb[col] = X_train_xgb[col].astype("category")
    train_categories = {col: X_train_xgb[col].cat.categories for col in CAT_COLS}

    X_test_xgb = X_test.copy()
    for col in CAT_COLS:
        X_test_xgb[col] = X_test_xgb[col].astype(
            pd.CategoricalDtype(categories=train_categories[col])
        )

    print("Training XGBoost...")
    xgb_model = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        tree_method="hist",
        enable_categorical=True,
        random_state=args.random_state,
        n_jobs=1,
    )
    xgb_model.fit(X_train_xgb, y_train_log)

    preds_xgb_log = xgb_model.predict(X_test_xgb)
    preds_xgb = np.expm1(preds_xgb_log)  # back to dollar scale

    # ---------------- CatBoost ---------------- #
    print("\nPreparing CatBoost data...")
    X_train_cb = X_train.copy()
    X_test_cb = X_test.copy()
    for col in CAT_COLS:
        X_train_cb[col] = X_train_cb[col].astype(str)
        X_test_cb[col] = X_test_cb[col].astype(str)

    train_pool = Pool(X_train_cb, y_train_log, cat_features=CAT_COLS)
    # test_pool target kept in log space too, purely for eval_set monitoring
    test_pool = Pool(X_test_cb, np.log1p(y_test), cat_features=CAT_COLS)

    print("Training CatBoost...")
    cb_model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        random_seed=args.random_state,
        verbose=100,
    )
    cb_model.fit(train_pool, eval_set=test_pool)  # FIX: was missing ')' and eval_set

    preds_cb_log = cb_model.predict(X_test_cb)     # FIX: was never called
    preds_cb = np.expm1(preds_cb_log)              # FIX: was never inverse-transformed

    # ---------------- Evaluate & pick winner ---------------- #
    print("\n=== Test set evaluation ===")
    results_xgb = evaluate(y_test, preds_xgb, "XGBoost")
    results_cb = evaluate(y_test, preds_cb, "CatBoost")

    higher_is_better = args.metric == "R2"
    xgb_score = results_xgb[args.metric]
    cb_score = results_cb[args.metric]
    xgb_wins = (xgb_score > cb_score) if higher_is_better else (xgb_score < cb_score)

    winner_name = "XGBoost" if xgb_wins else "CatBoost"
    winner_model = xgb_model if xgb_wins else cb_model
    print(f"\nBetter model by {args.metric}: {winner_name}")

    # ---------------- Validation predictions ---------------- #
    print(f"\nLoading validation data from {args.validation_csv} ...")
    val_df = load_and_clean(args.validation_csv)  # FIX: weight.abs() now applied here too

    missing_cols = [c for c in FEATURE_COLS if c not in val_df.columns]
    if missing_cols:
        sys.exit(f"ERROR: validation file is missing columns: {missing_cols}")

    X_val = val_df[FEATURE_COLS].copy()

    X_val_xgb = X_val.copy()
    for col in CAT_COLS:
        X_val_xgb[col] = X_val_xgb[col].astype(
            pd.CategoricalDtype(categories=train_categories[col])
        )

    X_val_cb = X_val.copy()
    for col in CAT_COLS:
        X_val_cb[col] = X_val_cb[col].astype(str)
    val_pool = Pool(X_val_cb, cat_features=CAT_COLS)  # name-based -> order-safe

    preds_val_xgb = np.expm1(xgb_model.predict(X_val_xgb))   # FIX: expm1 added
    preds_val_cb = np.expm1(cb_model.predict(val_pool))       # FIX: expm1 added

    val_df["predicted_rate_xgb"] = preds_val_xgb
    val_df["predicted_rate_cb"] = preds_val_cb
    val_df["predicted_rate_best"] = preds_val_xgb if xgb_wins else preds_val_cb
    val_df["best_model"] = winner_name

    # Flag rows with pickup/delivery cities unseen during training —
    # XGBoost's category alignment turns these into NaN internally.
    unseen_cities = (
        set(X_val["pickup"].unique()) - set(train_categories["pickup"])
    ) | (
        set(X_val["delivery"].unique()) - set(train_categories["delivery"])
    )
    val_df["low_confidence"] = (
        val_df["pickup"].isin(unseen_cities) | val_df["delivery"].isin(unseen_cities)
    )

    # ---- FINAL SUBMISSION FILE — exactly load_id,predicted_rate ---- #
    submission = val_df[["load_id", "predicted_rate_best"]].rename(
        columns={"predicted_rate_best": "predicted_rate"}
    )
    submission.to_csv(args.output_csv, index=False)
    print(f"\nSaved final submission file to {args.output_csv} "
          f"(load_id,predicted_rate — {len(submission)} rows)")

    if args.diagnostics_csv:
        val_df.to_csv(args.diagnostics_csv, index=False)
        print(f"Saved full diagnostics file to {args.diagnostics_csv} "
              f"(NOT for submission)")

    print(f"Rows flagged low_confidence (unseen pickup/delivery city): "
          f"{val_df['low_confidence'].sum()} / {len(val_df)}")

    # ---------------- December chart-inputs ---------------- #
    if args.december_csv:
        print(f"\nLoading December chart-inputs from {args.december_csv} ...")
        dec_df = fill_december_features(args.december_csv, df)

        X_dec_xgb = dec_df[FEATURE_COLS].copy()
        for col in CAT_COLS:
            X_dec_xgb[col] = X_dec_xgb[col].astype(
                pd.CategoricalDtype(categories=train_categories[col])
            )

        X_dec_cb = dec_df[FEATURE_COLS].copy()
        for col in CAT_COLS:
            X_dec_cb[col] = X_dec_cb[col].astype(str)
        dec_pool = Pool(X_dec_cb, cat_features=CAT_COLS)

        preds_dec = (
            np.expm1(xgb_model.predict(X_dec_xgb)) if xgb_wins
            else np.expm1(cb_model.predict(dec_pool))
        )
        dec_df["predicted_rate"] = preds_dec

        dec_output = dec_df[DECEMBER_TEMPLATE_COLS].copy()
        dec_output["date"] = pd.to_datetime(dec_output["date"]).dt.strftime("%Y-%m-%d")
        dec_output.to_csv(args.december_output, index=False)
        print(f"Saved completed December file to {args.december_output} "
              f"({winner_name} predictions, {len(dec_output)} rows)")


if __name__ == "__main__":
    main()
