import pandas as pd
import xgboost as xgb
import numpy as np

def doc_dl(path):
    return pd.read_csv(path)

def chuan_bi(df):
    drop_cols = ["Id", "ViTri", "Image", "Gia"]
    for c in drop_cols:
        if c in df.columns:
            df = df.drop(columns=c)
    return df

def tai_model(path):
    model = xgb.XGBRegressor()
    model.load_model(path)
    return model

def du_doan(model, df, feature_names):
    X = chuan_bi(df)
    X = X[feature_names]
    y_pred = model.predict(X)
    return y_pred

if __name__ == "__main__":
    du_lieu_path = "data/processed_data.csv"
    model_path = "models/xgboost_model.json"

    dl = doc_dl(du_lieu_path)

    mdl = tai_model(model_path)

    feature_names = chuan_bi(dl).columns.tolist()

    ket_qua = du_doan(mdl, dl, feature_names)

    for i, val in enumerate(ket_qua[:10], 1):
        print(f"Nhà thứ {i}: Dự đoán giá = {val:,.0f} tỷ VNĐ")

