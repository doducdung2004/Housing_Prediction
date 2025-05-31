import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

def danh_gia_mo_hinh(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(" Đánh giá mô hình ")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R^2: {r2:.4f}")

    return rmse, mae, r2

def bieu_do_thuc_te_du_doan(y_true, y_pred, so_diem=100):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true[:so_diem], y_pred[:so_diem], alpha=0.7, edgecolor='k')
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', lw=2)
    plt.xlabel("Giá thực tế")
    plt.ylabel("Giá dự đoán")
    plt.title("Biểu đồ Giá thực tế vs Dự đoán")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import numpy as np
    y_true = np.array([100, 150, 200, 250, 300])
    y_pred = np.array([110, 140, 210, 230, 310])

    danh_gia_mo_hinh(y_true, y_pred)
    bieu_do_thuc_te_du_doan(y_true, y_pred)
