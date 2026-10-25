"""
modelling.py (versi CI)  |  Hands-on 6  |  Kriteria 3

Dijalankan oleh `mlflow run MLProject`. Melatih RandomForestRegressor dengan parameter yang diberikan
MLProject, dicatat dengan autolog. Tidak memanggil set_tracking_uri atau set_experiment, karena alamat
tracking dan nama experiment ditentukan dari luar (variabel lingkungan dan opsi `mlflow run`).

Bisa juga dijalankan langsung:
    python modelling.py --data_dir influencer_marketing_roi_preprocessing
"""
import argparse
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TARGET = "engagements"
RANDOM_STATE = 42


def muat_data(data_dir):
    folder = Path(__file__).resolve().parent / data_dir
    train = pd.read_csv(folder / "train.csv")
    test = pd.read_csv(folder / "test.csv")
    X_train = train.drop(columns=[TARGET]).astype("float64")
    X_test = test.drop(columns=[TARGET]).astype("float64")
    return X_train, X_test, train[TARGET], test[TARGET]


def hitung_metrik(y_asli, y_prediksi, awalan):
    mse = mean_squared_error(y_asli, y_prediksi)
    return {
        f"{awalan}_mean_squared_error": mse,
        f"{awalan}_root_mean_squared_error": float(np.sqrt(mse)),
        f"{awalan}_mean_absolute_error": mean_absolute_error(y_asli, y_prediksi),
        f"{awalan}_r2_score": r2_score(y_asli, y_prediksi),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="influencer_marketing_roi_preprocessing")
    parser.add_argument("--n_estimators", type=int, default=50)
    parser.add_argument("--max_depth", type=int, default=3)
    parser.add_argument("--min_samples_leaf", type=int, default=100)
    parser.add_argument("--max_features", type=float, default=0.5)
    args = parser.parse_args()

    X_train, X_test, y_train, y_test = muat_data(args.data_dir)
    print(f"Data latih: {X_train.shape} | Data uji: {X_test.shape}")

    mlflow.sklearn.autolog(log_input_examples=True)

    # Di bawah `mlflow run`, run sudah dibuat oleh MLflow dan start_run() melanjutkannya
    with mlflow.start_run():
        model = RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            max_features=args.max_features,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
        model.fit(X_train, y_train)

        skor_uji = model.score(X_test, y_test)     # dicatat autolog + memunculkan metric_info.json
        metrik_uji = hitung_metrik(y_test, model.predict(X_test), "test")
        mlflow.log_metrics(metrik_uji)
        mlflow.log_metric("baseline_test_mean_absolute_error",
                          mean_absolute_error(y_test, np.full(len(y_test), y_train.mean())))

        print(f"Skor R2 data uji: {skor_uji:,.4f}")
        print(f"MAE data uji    : {metrik_uji['test_mean_absolute_error']:,.4f}")


if __name__ == "__main__":
    main()
