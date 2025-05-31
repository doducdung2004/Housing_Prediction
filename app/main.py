import streamlit as st
import pandas as pd
import torch
from torchvision import transforms, models
from PIL import Image
import xgboost as xgb
import numpy as np
import os

def trich_dac_trung_anh(anh):
    mo_hinh = models.resnet18(pretrained=True)
    mo_hinh.eval()
    mo_hinh = torch.nn.Sequential(*list(mo_hinh.children())[:-1])

    bien_doi = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])

    tensor_anh = bien_doi(anh).unsqueeze(0)
    with torch.no_grad():
        dac_trung = mo_hinh(tensor_anh).squeeze().numpy()
    return dac_trung

danh_sach_quan = [
    'Ba Dinh', 'Hoan Kiem', 'Tay Ho', 'Long Bien', 'Cau Giay',
    'Dong Da', 'Hai Ba Trung', 'Hoang Mai', 'Thanh Xuan', 'Nam Tu Liem'
]

def hien_ban_do_google(kinh_do, vi_do):
    url_map = f"https://www.google.com/maps?q={vi_do},{kinh_do}&hl=vi&z=16"
    iframe = f"""
        <iframe width="100%" height="400" frameborder="0" style="border:0"
        src="{url_map}&output=embed" allowfullscreen></iframe>
    """
    st.markdown("### Vi tri tren Google Maps")
    st.markdown(iframe, unsafe_allow_html=True)

def ung_dung():
    st.title("Dự đoán giá nhà đất")

    dien_tich = st.number_input("Dien tich (m²)", min_value=10, max_value=1000, value=50)
    vi_tri = st.selectbox("Vi tri (quan Ha Noi)", danh_sach_quan)
    so_phong_ngu = st.slider("So phong ngu", 1, 10, 3)
    so_wc = st.slider("So phong WC", 1, 10, 2)
    so_tang = st.slider("So tang", 1, 5, 1)
    kinh_do = st.number_input("Kinh do", value=105.8, format="%.6f")
    vi_do = st.number_input("Vi do", value=21.0, format="%.6f")

    hien_ban_do_google(kinh_do, vi_do)

    tep_anh = st.file_uploader("Tai anh cua ngoi nha", type=["jpg", "jpeg", "png"])

    if st.button("Du doan gia"):
        if tep_anh is None:
            st.error("Vui long tai anh de du doan.")
            return

        anh = Image.open(tep_anh).convert("RGB")
        st.write("Dang trich dac trung tu anh...")
        dac_trung = trich_dac_trung_anh(anh)

        du_lieu = {
            'DienTich': dien_tich,
            'SoPhongNgu': so_phong_ngu,
            'SoWC': so_wc,
            'SoTang': so_tang,
            'KinhDo': kinh_do,
            'ViDo': vi_do,
            'ViTri_ma': danh_sach_quan.index(vi_tri) if vi_tri in danh_sach_quan else 0
        }

        for i in range(len(dac_trung)):
            du_lieu[f'f{i}'] = dac_trung[i]
        du_lieu_df = pd.DataFrame([du_lieu])

        try:
            with open("src/models/features.txt") as tep:
                ten_dac_trung = tep.read().splitlines()
        except FileNotFoundError:
            st.error("Khong tim thay file 'features.txt'.")
            return

        thieu_cot = [cot for cot in ten_dac_trung if cot not in du_lieu_df.columns]
        if thieu_cot:
            st.error(f"Thieu cot dac trung: {thieu_cot}")
            return

        du_lieu_df = du_lieu_df[ten_dac_trung]

        mo_hinh = xgb.XGBRegressor()
        mo_hinh.load_model("src/models/xgboost_model.json")

        st.write("Dang du doan gia...")
        gia_du_doan = mo_hinh.predict(du_lieu_df)[0]
        st.success(f"Gia nha du doan: {gia_du_doan:,.0f} ty VND")

if __name__ == "__main__":
    ung_dung()
