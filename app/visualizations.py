import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

def gia(duong_dan):
    df = pd.read_csv(duong_dan)
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Gia'], kde=True, bins=30)
    plt.title('Phân phối giá nhà')
    plt.xlabel('Giá nhà')
    plt.ylabel('Số lượng')
    plt.tight_layout()
    plt.show()



def so_sanh(thuc_te, du_doan):
    df = pd.DataFrame({
        'Thực tế': thuc_te,
        'Dự đoán': du_doan
    })
    fig = px.scatter(df, x='Thực tế', y='Dự đoán',
                     title='Thực tế vs. Dự đoán',
                     labels={'Thực tế': 'Giá thực tế', 'Dự đoán': 'Giá dự đoán'},
                     trendline='ols')
    fig.show()

def quan_trong(dac_trung, top=10):
    df = pd.DataFrame(dac_trung.items(), columns=['Đặc trưng', 'Tầm quan trọng'])
    df = df.sort_values(by='Tầm quan trọng', ascending=False).head(top)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Tầm quan trọng', y='Đặc trưng', data=df, palette='Reds_r')
    plt.title(f'Top {top} đặc trưng quan trọng')
    plt.xlabel('Tầm quan trọng')
    plt.ylabel('Đặc trưng')
    plt.tight_layout()
    plt.show()