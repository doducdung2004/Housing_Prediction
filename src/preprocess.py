import pandas as pd
from sklearn.preprocessing import LabelEncoder

def doc_du_lieu(duong_dan_bang, duong_dan_anh):
    bang = pd.read_csv(duong_dan_bang)
    anh = pd.read_csv(duong_dan_anh)
    return bang, anh

def gop_du_lieu(bang, anh):
    df = pd.merge(bang, anh, on="Image", how="left")
    return df

def xu_ly_thieu(df):
    for cot in df.select_dtypes(include=['float64', 'int64']).columns:
        df[cot].fillna(df[cot].mean(), inplace=True)
    for cot in df.select_dtypes(include=['object']).columns:
        df[cot].fillna('KhongCo', inplace=True)
    return df

def ma_hoa_vi_tri(df):
    if 'ViTri' in df.columns:
        encoder = LabelEncoder()
        df['ViTri_ma'] = encoder.fit_transform(df['ViTri'])
    return df

def luu_du_lieu(df, duong_dan):
    df.to_csv(duong_dan, index=False)

if __name__ == "__main__":
    duong_dan_bang = "data/house_data.csv"
    duong_dan_anh = "data/image_features.csv"
    duong_dan_kq = "data/processed_data.csv"

    bang, anh = doc_du_lieu(duong_dan_bang, duong_dan_anh)
    df = gop_du_lieu(bang, anh)
    df = xu_ly_thieu(df)
    df = ma_hoa_vi_tri(df)
    luu_du_lieu(df, duong_dan_kq)

    print(f"Đã xử lý và lưu dữ liệu tại {duong_dan_kq}")
