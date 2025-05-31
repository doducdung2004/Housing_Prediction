import pandas as pd

def thong_tin_df(df, ten="Dữ liệu"):
    print(f"== Thông tin {ten} ==")
    print(df.info())
    print(df.describe())
    print(f"Số dòng: {df.shape[0]}, số cột: {df.shape[1]}")
    print("-" * 40)

def cot_so(df):
    return df.select_dtypes(include=['int64', 'float64']).columns.tolist()

def cot_object(df):
    return df.select_dtypes(include=['object']).columns.tolist()

def kiem_tra_thieu(df):
    thieu = df.isnull().sum()
    thieu = thieu[thieu > 0]
    if len(thieu) == 0:
        print("Không có giá trị thiếu dữ liệu.")
    else:
        print("Giá trị thiếu dữ liệu theo cột:")
        print(thieu)

def ghi_log(msg):
    print(f"[LOG] {msg}")
