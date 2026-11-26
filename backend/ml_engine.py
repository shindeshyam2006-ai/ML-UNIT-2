"""Core ML workflow for Activity 2: Simple House Price Prediction (Ames Housing)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from backend.config import CLEAN_DATA_PATH, RAW_DATA_PATH

DATASET_NAME = "Ames Housing Dataset (De Cock, 2011)"
DATASET_SOURCE_URL = "https://www.openml.org/d/42165"
DATASET_LOCATION = "Ames, Iowa, USA"
PRIMARY_FEATURE = "GrLivArea"
TARGET_COLUMN = "SalePrice"
MULTI_FEATURES = [
    "GrLivArea",
    "OverallQual",
    "TotalBsmtSF",
    "GarageCars",
    "FullBath",
    "YearBuilt",
]
STRUCTURAL_NONE_COLS = [
    "PoolQC",
    "MiscFeature",
    "Alley",
    "Fence",
    "FireplaceQu",
    "GarageType",
    "GarageFinish",
    "GarageQual",
    "GarageCond",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "MasVnrType",
]


def clean_ames_dataframe(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Clean Ames Housing dataset while preserving all authentic property sales."""
    df_clean = df_raw.copy()
    for col in STRUCTURAL_NONE_COLS:
        if col in df_clean.columns:
            # Use 'Absent' rather than 'None' so pandas.read_csv does not re-parse it as NaN
            df_clean[col] = df_clean[col].fillna("Absent")

    if "LotFrontage" in df_clean.columns:
        df_clean["LotFrontage"] = df_clean.groupby("Neighborhood")["LotFrontage"].transform(
            lambda s: s.fillna(s.median())
        )
        df_clean["LotFrontage"] = df_clean["LotFrontage"].fillna(df_clean["LotFrontage"].median())

    if "MasVnrArea" in df_clean.columns:
        df_clean["MasVnrArea"] = df_clean["MasVnrArea"].fillna(0.0)
    if "GarageYrBlt" in df_clean.columns:
        df_clean["GarageYrBlt"] = df_clean["GarageYrBlt"].fillna(0)
    if "Electrical" in df_clean.columns and df_clean["Electrical"].isna().any():
        df_clean["Electrical"] = df_clean["Electrical"].fillna(df_clean["Electrical"].mode()[0])

    return df_clean


@dataclass
class HousingMLPipeline:
    """Encapsulates the Activity 2 Simple Linear Regression pipeline and diagnostics."""

    raw_path: Path = RAW_DATA_PATH
    clean_path: Path = CLEAN_DATA_PATH
    test_size: float = 0.20
    random_state: int = 42

    def __post_init__(self) -> None:
        if not self.raw_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {self.raw_path}. Please run the Activity 2 setup or notebook first."
            )

        self.df_raw = pd.read_csv(self.raw_path)
        self.df_clean = clean_ames_dataframe(self.df_raw)
        if not self.clean_path.exists():
            self.clean_path.parent.mkdir(parents=True, exist_ok=True)
            self.df_clean.to_csv(self.clean_path, index=False)

        X = self.df_clean[[PRIMARY_FEATURE]]
        y = self.df_clean[TARGET_COLUMN]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        # Primary Activity 2 Model: Simple Linear Regression
        self.model = LinearRegression()
        self.model.fit(self.X_train, self.y_train)
        self.y_pred = self.model.predict(self.X_test)

        # Educational Benchmark: Multi-Feature Linear Regression on identical split
        self.multi_model = LinearRegression()
        self.multi_model.fit(self.df_clean.loc[self.X_train.index, MULTI_FEATURES], self.y_train)
        self.multi_pred = self.multi_model.predict(self.df_clean.loc[self.X_test.index, MULTI_FEATURES])

        self._summary_cache = self._build_summary()

    def _build_summary(self) -> dict[str, Any]:
        missing_series = self.df_raw.isna().sum()
        missing_cols = [
            {
                "column": str(col),
                "missing_count": int(cnt),
                "missing_pct": round(float(cnt / len(self.df_raw) * 100), 2),
                "dtype": str(self.df_raw[col].dtype),
            }
            for col, cnt in missing_series[missing_series > 0].sort_values(ascending=False).items()
        ]

        numeric_cols = self.df_clean.select_dtypes(include=[np.number]).columns.drop("Id", errors="ignore")
        corr_series = self.df_clean[numeric_cols].corr()[TARGET_COLUMN].sort_values(ascending=False)
        top_correlations = [
            {"feature": str(feat), "correlation": round(float(val), 4)}
            for feat, val in corr_series.drop(TARGET_COLUMN, errors="ignore").head(10).items()
        ]
        id_corr = round(float(self.df_clean["Id"].corr(self.df_clean[TARGET_COLUMN])), 4) if "Id" in self.df_clean else 0.0

        intercept = float(self.model.intercept_)
        slope = float(self.model.coef_[0])

        mae = float(mean_absolute_error(self.y_test, self.y_pred))
        mse = float(mean_squared_error(self.y_test, self.y_pred))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(self.y_test, self.y_pred))

        baseline_pred = np.full(shape=len(self.y_test), fill_value=float(self.y_train.mean()))
        baseline_mae = float(mean_absolute_error(self.y_test, baseline_pred))
        baseline_mse = float(mean_squared_error(self.y_test, baseline_pred))
        baseline_rmse = float(np.sqrt(baseline_mse))

        multi_mae = float(mean_absolute_error(self.y_test, self.multi_pred))
        multi_rmse = float(np.sqrt(mean_squared_error(self.y_test, self.multi_pred)))
        multi_r2 = float(r2_score(self.y_test, self.multi_pred))

        test_df = pd.DataFrame(
            {
                "id": self.df_clean.loc[self.X_test.index, "Id"].astype(int).values,
                "neighborhood": self.df_clean.loc[self.X_test.index, "Neighborhood"].astype(str).values,
                "overall_qual": self.df_clean.loc[self.X_test.index, "OverallQual"].astype(int).values,
                "year_built": self.df_clean.loc[self.X_test.index, "YearBuilt"].astype(int).values,
                "gr_liv_area": self.X_test[PRIMARY_FEATURE].astype(float).values,
                "actual_price": self.y_test.astype(float).values,
                "predicted_price": np.round(self.y_pred, 2),
            }
        )
        test_df["residual"] = np.round(test_df["actual_price"] - test_df["predicted_price"], 2)
        test_df["abs_error"] = np.abs(test_df["residual"]).round(2)
        test_df["pct_error"] = np.round(test_df["abs_error"] / test_df["actual_price"] * 100, 2)

        train_set = set(self.X_train.index.tolist())
        scatter_points = [
            {
                "id": int(row["Id"]),
                "area": float(row[PRIMARY_FEATURE]),
                "price": float(row[TARGET_COLUMN]),
                "qual": int(row["OverallQual"]),
                "year_built": int(row["YearBuilt"]),
                "neighborhood": str(row["Neighborhood"]),
                "split": "train" if idx in train_set else "test",
                "outlier": bool(row[PRIMARY_FEATURE] > 4000 and row[TARGET_COLUMN] < 300000),
            }
            for idx, row in self.df_clean.iterrows()
        ]

        feature_medians = {
            feat: float(self.df_clean[feat].median()) for feat in MULTI_FEATURES
        }

        return {
            "course": {
                "institution": "Zeal College of Engineering and Research, Narhe, Pune",
                "department": "AI & DS",
                "class_name": "T.Y. B.Tech (Sem V)",
                "academic_year": "2026-27",
                "course_code": "ADPC509",
                "activity": "Activity 2: Simple House Price Prediction",
                "unit": "Unit II - Supervised Learning: Regression",
            },
            "dataset": {
                "name": DATASET_NAME,
                "source_url": DATASET_SOURCE_URL,
                "location": DATASET_LOCATION,
                "rows": int(self.df_raw.shape[0]),
                "columns": int(self.df_raw.shape[1]),
                "duplicates": int(self.df_raw.duplicated().sum()),
                "missing_columns_count": len(missing_cols),
                "total_missing_before": int(self.df_raw.isna().sum().sum()),
                "total_missing_after": int(self.df_clean.isna().sum().sum()),
                "missing_columns": missing_cols,
                "primary_feature": PRIMARY_FEATURE,
                "target": TARGET_COLUMN,
                "id_correlation": id_corr,
                "area_stats": {
                    "min": float(self.df_clean[PRIMARY_FEATURE].min()),
                    "max": float(self.df_clean[PRIMARY_FEATURE].max()),
                    "mean": round(float(self.df_clean[PRIMARY_FEATURE].mean()), 2),
                    "median": float(self.df_clean[PRIMARY_FEATURE].median()),
                    "std": round(float(self.df_clean[PRIMARY_FEATURE].std()), 2),
                    "train_min": float(self.X_train[PRIMARY_FEATURE].min()),
                    "train_max": float(self.X_train[PRIMARY_FEATURE].max()),
                },
                "price_stats": {
                    "min": float(self.df_clean[TARGET_COLUMN].min()),
                    "max": float(self.df_clean[TARGET_COLUMN].max()),
                    "mean": round(float(self.df_clean[TARGET_COLUMN].mean()), 2),
                    "median": float(self.df_clean[TARGET_COLUMN].median()),
                    "std": round(float(self.df_clean[TARGET_COLUMN].std()), 2),
                },
                "top_correlations": top_correlations,
                "feature_medians": feature_medians,
            },
            "split": {
                "ratio": "80:20",
                "test_size": self.test_size,
                "random_state": self.random_state,
                "train_count": int(len(self.X_train)),
                "test_count": int(len(self.X_test)),
                "train_area_mean": round(float(self.X_train[PRIMARY_FEATURE].mean()), 2),
                "test_area_mean": round(float(self.X_test[PRIMARY_FEATURE].mean()), 2),
                "train_price_mean": round(float(self.y_train.mean()), 2),
                "test_price_mean": round(float(self.y_test.mean()), 2),
            },
            "model": {
                "algorithm": "Ordinary Least Squares Linear Regression",
                "intercept": round(intercept, 4),
                "slope": round(slope, 4),
                "equation": f"SalePrice_hat = {intercept:,.2f} + {slope:,.2f} * GrLivArea",
                "r2_train": round(float(self.model.score(self.X_train, self.y_train)), 4),
            },
            "metrics": {
                "mae": round(mae, 2),
                "mse": round(mse, 2),
                "rmse": round(rmse, 2),
                "r2": round(r2, 4),
                "mae_pct_of_mean": round(mae / float(self.y_test.mean()) * 100, 2),
                "baseline_mae": round(baseline_mae, 2),
                "baseline_rmse": round(baseline_rmse, 2),
                "mae_improvement_pct": round((baseline_mae - mae) / baseline_mae * 100, 2),
                "multi_feature_benchmark": {
                    "features": MULTI_FEATURES,
                    "mae": round(multi_mae, 2),
                    "rmse": round(multi_rmse, 2),
                    "r2": round(multi_r2, 4),
                },
            },
            "test_predictions_sample": test_df.head(20).to_dict(orient="records"),
            "closest_predictions": test_df.sort_values("abs_error").head(5).to_dict(orient="records"),
            "largest_errors": test_df.sort_values("abs_error", ascending=False).head(5).to_dict(orient="records"),
            "test_points": test_df.to_dict(orient="records"),
            "scatter_points": scatter_points,
        }

    def get_summary(self) -> dict[str, Any]:
        """Return full precomputed summary and diagnostics."""
        return self._summary_cache

    def predict(
        self,
        gr_liv_area: float,
        overall_qual: int | None = None,
        total_bsmt_sf: float | None = None,
        garage_cars: int | None = None,
        full_bath: int | None = None,
        year_built: int | None = None,
    ) -> dict[str, Any]:
        """Predict property price using Simple Linear Regression and return contextual comparables."""
        input_df = pd.DataFrame({PRIMARY_FEATURE: [float(gr_liv_area)]})
        simple_pred = float(self.model.predict(input_df)[0])

        medians = self._summary_cache["dataset"]["feature_medians"]
        multi_row = pd.DataFrame(
            [
                {
                    "GrLivArea": float(gr_liv_area),
                    "OverallQual": float(overall_qual if overall_qual is not None else medians["OverallQual"]),
                    "TotalBsmtSF": float(total_bsmt_sf if total_bsmt_sf is not None else medians["TotalBsmtSF"]),
                    "GarageCars": float(garage_cars if garage_cars is not None else medians["GarageCars"]),
                    "FullBath": float(full_bath if full_bath is not None else medians["FullBath"]),
                    "YearBuilt": float(year_built if year_built is not None else medians["YearBuilt"]),
                }
            ]
        )
        multi_pred = float(self.multi_model.predict(multi_row)[0])

        rmse = self._summary_cache["metrics"]["rmse"]
        mae = self._summary_cache["metrics"]["mae"]
        train_min = self._summary_cache["dataset"]["area_stats"]["train_min"]
        train_max = self._summary_cache["dataset"]["area_stats"]["train_max"]
        in_training_range = bool(train_min <= gr_liv_area <= train_max)

        area_percentile = float((self.df_clean[PRIMARY_FEATURE] <= gr_liv_area).mean() * 100)
        price_percentile = float((self.df_clean[TARGET_COLUMN] <= simple_pred).mean() * 100)

        # Nearest comparable sales by GrLivArea
        comps_df = self.df_clean.copy()
        comps_df["area_diff"] = np.abs(comps_df[PRIMARY_FEATURE] - gr_liv_area)
        nearest = comps_df.sort_values(["area_diff", "Id"]).head(5)
        comparables = [
            {
                "id": int(row["Id"]),
                "neighborhood": str(row["Neighborhood"]),
                "gr_liv_area": float(row[PRIMARY_FEATURE]),
                "overall_qual": int(row["OverallQual"]),
                "year_built": int(row["YearBuilt"]),
                "actual_price": float(row[TARGET_COLUMN]),
            }
            for _, row in nearest.iterrows()
        ]

        intercept = float(self.model.intercept_)
        slope = float(self.model.coef_[0])

        return {
            "input": {
                "gr_liv_area": float(gr_liv_area),
                "overall_qual": int(multi_row["OverallQual"].iloc[0]),
                "total_bsmt_sf": float(multi_row["TotalBsmtSF"].iloc[0]),
                "garage_cars": int(multi_row["GarageCars"].iloc[0]),
                "full_bath": int(multi_row["FullBath"].iloc[0]),
                "year_built": int(multi_row["YearBuilt"].iloc[0]),
            },
            "simple_linear_regression": {
                "predicted_price_usd": round(simple_pred, 2),
                "units": "USD",
                "intercept": round(intercept, 2),
                "slope": round(slope, 2),
                "formula_evaluated": f"${intercept:,.2f} + (${slope:,.2f} x {gr_liv_area:,.0f}) = ${simple_pred:,.2f}",
                "mae_band": [round(max(0.0, simple_pred - mae), 2), round(simple_pred + mae, 2)],
                "rmse_band": [round(max(0.0, simple_pred - rmse), 2), round(simple_pred + rmse, 2)],
                "in_training_range": in_training_range,
                "training_range": [train_min, train_max],
                "area_percentile": round(area_percentile, 1),
                "price_percentile": round(price_percentile, 1),
            },
            "multi_feature_reference": {
                "predicted_price_usd": round(max(0.0, multi_pred), 2),
                "difference_vs_simple": round(max(0.0, multi_pred) - simple_pred, 2),
                "note": "Educational 6-feature comparison showing impact of build quality, basement, garage, baths, and age.",
            },
            "comparables": comparables,
        }
