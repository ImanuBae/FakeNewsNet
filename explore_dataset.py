import pandas as pd

# Đọc dữ liệu
print("Đang đọc dữ liệu...")
politifact_fake = pd.read_csv('dataset/politifact_fake.csv')
politifact_real = pd.read_csv('dataset/politifact_real.csv')
gossipcop_fake = pd.read_csv('dataset/gossipcop_fake.csv')
gossipcop_real = pd.read_csv('dataset/gossipcop_real.csv')

print("\n" + "="*60)
print("TỔNG QUAN DATASET")
print("="*60)

print(f"\nPolitiFact:")
print(f"  - Tin giả: {len(politifact_fake)} bài")
print(f"  - Tin thật: {len(politifact_real)} bài")
print(f"  - Tổng: {len(politifact_fake) + len(politifact_real)} bài")

print(f"\nGossipCop:")
print(f"  - Tin giả: {len(gossipcop_fake)} bài")
print(f"  - Tin thật: {len(gossipcop_real)} bài")
print(f"  - Tổng: {len(gossipcop_fake) + len(gossipcop_real)} bài")

print(f"\nTỔNG CỘNG: {len(politifact_fake) + len(politifact_real) + len(gossipcop_fake) + len(gossipcop_real)} bài viết")

print("\n" + "="*60)
print("CẤU TRÚC DỮ LIỆU - POLITIFACT FAKE")
print("="*60)
print("\nCác cột có trong dataset:")
print(politifact_fake.columns.tolist())
print("\nMẫu dữ liệu (5 dòng đầu):")
print(politifact_fake.head())
print("\nThông tin chi tiết:")
print(politifact_fake.info())
print("\nKiểm tra giá trị null:")
print(politifact_fake.isnull().sum())

print("\n" + "="*60)
print("CẤU TRÚC DỮ LIỆU - POLITIFACT REAL")
print("="*60)
print(politifact_real.head())

# Lưu thống kê
with open('dataset_summary.txt', 'w', encoding='utf-8') as f:
    f.write("THỐNG KÊ DATASET FAKENEWSNET\n")
    f.write("="*60 + "\n\n")
    f.write(f"PolitiFact Fake: {len(politifact_fake)} bài\n")
    f.write(f"PolitiFact Real: {len(politifact_real)} bài\n")
    f.write(f"GossipCop Fake: {len(gossipcop_fake)} bài\n")
    f.write(f"GossipCop Real: {len(gossipcop_real)} bài\n")
    f.write(f"\nCác cột: {politifact_fake.columns.tolist()}\n")

print("\nĐã lưu thống kê vào file 'dataset_summary.txt'")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np

print("Đang load dữ liệu...")

# Đọc dữ liệu
politifact_fake = pd.read_csv('dataset/politifact_fake.csv')
politifact_real = pd.read_csv('dataset/politifact_real.csv')

# Thêm label (0 = fake, 1 = real)
politifact_fake['label'] = 0
politifact_real['label'] = 1

# Gộp dữ liệu
data = pd.concat([politifact_fake, politifact_real], ignore_index=True)

# Kiểm tra cột text (thay 'title' hoặc 'content' tùy dataset)
print("\nCác cột có trong dataset:")
print(data.columns.tolist())

# Giả sử có cột 'title' hoặc 'text', bạn điều chỉnh tên cột phù hợp
text_column = 'title'  # THAY ĐỔI NẾU CẦN

# Loại bỏ dòng null
data = data.dropna(subset=[text_column])

print(f"\nTổng số mẫu: {len(data)}")
print(f"Fake news: {len(data[data['label']==0])}")
print(f"Real news: {len(data[data['label']==1])}")

# Chia train/test
X = data[text_column]
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set: {len(X_train)}")
print(f"Test set: {len(X_test)}")

# Vectorization
print("\nĐang vector hóa text...")
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train models
print("\n" + "="*60)
print("TRAINING MODELS")
print("="*60)

models = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    
    print(f"\n{name} Results:")
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# Kết quả tổng hợp
print("\n" + "="*60)
print("KẾT QUẢ TỔNG HỢP")
print("="*60)
for name, acc in results.items():
    print(f"{name}: {acc:.4f}")

best_model = max(results, key=results.get)
print(f"\nModel tốt nhất: {best_model} với accuracy {results[best_model]:.4f}")

# Lưu kết quả
with open('model_results.txt', 'w') as f:
    f.write("KẾT QUẢ PHÂN LOẠI FAKE NEWS\n")
    f.write("="*60 + "\n\n")
    for name, acc in results.items():
        f.write(f"{name}: {acc:.4f}\n")
    f.write(f"\nBest Model: {best_model}\n")

print("\n✅ Đã lưu kết quả vào 'model_results.txt'")