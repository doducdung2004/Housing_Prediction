import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import os

def doc_dl(duong_dan):
    if not os.path.exists(duong_dan):
        raise FileNotFoundError(f"Không tìm thấy file: {duong_dan}")
    return pd.read_csv(duong_dan)

def tach_dac_trung_va_nhan(df):
    y = df["Gia"]
    cot_bo = ["Gia", "Image", "ViTri", "Id"]
    df = df.drop(columns=[c for c in cot_bo if c in df.columns])
    X = df
    return X, y

def huan_luyen_xgb(X_train, y_train, X_val, y_val):
    model = xgb.XGBRegressor(
        trees=500,
        max_depth=6,
        learning_rate=0.05,
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=True,
    )
    return model

def danh_gia_model(model, X_test, y_test):
    mang_test = model.predict(X_test)
    mse = mean_squared_error(y_test, mang_test)
    rmse = np.sqrt(mse)
    print(f"RMSE trên tập test: {rmse:.4f}")

def luu_model(model, duong_dan_model):
    os.makedirs(os.path.dirname(duong_dan_model), exist_ok=True)
    model.save_model(duong_dan_model)
    print(f" Đã lưu mô hình tại: {duong_dan_model}")

if __name__ == "__main__":
    DU_LIEU = "data/processed_data.csv"
    MODEL = "models/xgboost_model.json"

    df = doc_dl(DU_LIEU)
    X, y = tach_dac_trung_va_nhan(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = huan_luyen_xgb(X_train, y_train, X_test, y_test)

    danh_gia_model(model, X_test, y_test)

    luu_model(model, MODEL)
    with open("models/features.txt", "w") as f:
        f.write("\n".join(X.columns))
