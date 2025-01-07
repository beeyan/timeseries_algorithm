# pipelines/train_pipeline.py

import os
import argparse
import joblib
import pandas as pd
from typing import Tuple

from timeseries_algorithm.data.dataloader import load_sample_data
from timeseries_algorithm.data.dataset import TimeSeriesDataset
from timeseries_algorithm.features.create_features import (
    create_lag_features,
    create_rolling_features,
    add_time_features
)
from timeseries_algorithm.models.arima_model import ARIMAXModel
from timeseries_algorithm.utils.metrics import calculate_metrics
from timeseries_algorithm.utils.logging import setup_logger

def load_and_create_dataset(target_col: str, date_col: str) -> TimeSeriesDataset:
    """
    CSVファイルからデータを読み込み、TimeSeriesDatasetを作成する。
    """
    df = load_sample_data()
    dataset = TimeSeriesDataset(data=df, target_col=target_col, date_col=date_col)
    return dataset

def feature_engineering(data: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    データに対して、ラグ特徴量・移動平均・日時特徴量を追加する。
    """
    data_fe = create_lag_features(data, target_col=target_col, lags=[1,7,14])
    data_fe = create_rolling_features(data_fe, target_col=target_col, windows=[7,30])
    data_fe = add_time_features(data_fe, date_index=True)
    data_fe.dropna(inplace=True)  # ラグやローリングで発生したNaNを除去
    return data_fe

def split_data(data_fe: pd.DataFrame, split_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    学習用/検証用にデータを分割する。
    TimeSeriesDatasetクラスの split_train_test() を用いつつ、
    特徴量付きDataFrameとの整合を取る。
    """
    # ここでは dataset.split_train_test() と同じsplit_dateで
    # data_fe も日付で切り分ける形をとる
    train_data = data_fe.loc[:split_date]
    val_data   = data_fe.loc[split_date:]
    return train_data, val_data

def build_model(model_name: str):
    """
    指定されたモデル名に応じてモデルインスタンスを生成する。
    """
    if model_name == "arima":
        return ARIMAXModel()
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def train_model(model, X_train, y_train):
    """
    モデルを学習する (fit)。
    """
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_val, y_val):
    """
    検証データで予測し、評価指標を計算する。
    """
    y_val_pred = model.predict(X_val)
    metrics_val = calculate_metrics(y_val, y_val_pred)
    return metrics_val

def save_trained_model(model, model_name: str):
    """
    学習済みモデルを保存する。
    """
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", f"{model_name}_model.pkl")
    joblib.dump(model, model_path)
    return model_path

def main(args):
    logger = setup_logger(__name__)

    # 1. データ読み込み & Dataset作成
    logger.info("Loading data and creating dataset...")
    dataset = load_and_create_dataset(args.target_col, args.date_col)

    # 2. 特徴量エンジニアリング
    logger.info("Applying feature engineering...")
    data_fe = feature_engineering(dataset.data, target_col=args.target_col)

    # 3. データ分割
    logger.info("Splitting data into train/val...")
    train_data, val_data = split_data(data_fe, args.train_val_split_date)
    X_train, y_train = dataset.get_features_and_target(train_data)
    X_val, y_val     = dataset.get_features_and_target(val_data)

    # 4. モデル作成
    logger.info(f"Building model: {args.model}")
    model = build_model(args.model)

    # 5. 学習
    logger.info("Training model...")
    model = train_model(model, X_train, y_train)

    # 6. 評価
    logger.info("Evaluating model...")
    val_metrics = evaluate_model(model, X_val, y_val)
    logger.info(f"Validation metrics: {val_metrics}")

    # 7. 保存
    if args.save_model:
        path = save_trained_model(model, args.model)
        logger.info(f"Trained model saved to {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_col", type=str, default="sales", help="Name of the target column.")
    parser.add_argument("--date_col", type=str, default="date", help="Name of the date column.")
    parser.add_argument("--train_val_split_date", type=str, default="2022-01-01", help="Date to split train/val.")
    parser.add_argument("--model", type=str, default="arima", help="Which model to use for training.")
    parser.add_argument("--save_model", action="store_true", help="Whether to save the trained model.")
    args = parser.parse_args()

    main(args)
