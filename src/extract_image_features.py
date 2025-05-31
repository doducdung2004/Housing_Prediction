import os
import torch
import torchvision.transforms as T
from torchvision.models import resnet18
from PIL import Image
import pandas as pd

thu_muc_anh = "data/images"
ds_anh = sorted([f for f in os.listdir(thu_muc_anh) if f.endswith(".jpg")])

bien_doi = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

model = resnet18(pretrained=True)
model.fc = torch.nn.Identity()
model.eval()

ds_kq = []

for ten in ds_anh:
    try:
        anh = Image.open(os.path.join(thu_muc_anh, ten)).convert("RGB")
        dau_vao = bien_doi(anh).unsqueeze(0)
        with torch.no_grad():
            dac_trung = model(dau_vao).squeeze().numpy()
        hang = [ten] + dac_trung.round(6).tolist()
        ds_kq.append(hang)
    except Exception as e:
        print(f"Lỗi xử lý {ten}: {e}")
cot = ["Image"] + [f"f{i}" for i in range(512)]
df = pd.DataFrame(ds_kq, columns=cot)
df.to_csv("data/image_features.csv", index=False)
print("Đã lưu xong file data/image_features.csv")
