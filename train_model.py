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

# Kiểm tra cột text - TỰ ĐỘNG NHẬN DIỆN
print("\nCác cột có trong dataset:")
print(data.columns.tolist())
print("\nMẫu dữ liệu 3 dòng đầu:")
print(data.head(3))

# Tự động tìm cột text phù hợp
if 'title' in data.columns:
    text_column = 'title'
    print("\n✓ Sử dụng cột: 'title'")
elif 'text' in data.columns:
    text_column = 'text'
    print("\n✓ Sử dụng cột: 'text'")
elif 'content' in data.columns:
    text_column = 'content'
    print("\n✓ Sử dụng cột: 'content'")
elif 'news_url' in data.columns:
    text_column = 'news_url'
    print("\n✓ Sử dụng cột: 'news_url'")
else:
    # Lấy cột đầu tiên không phải id/label
    text_column = [col for col in data.columns if col not in ['id', 'label']][0]
    print(f"\n⚠ Không tìm thấy cột thông dụng, sử dụng: '{text_column}'")

# Loại bỏ dòng null
print(f"\nSố dòng trước khi loại bỏ null: {len(data)}")
data = data.dropna(subset=[text_column])
print(f"Số dòng sau khi loại bỏ null: {len(data)}")

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
with open('model_results.txt', 'w', encoding='utf-8') as f:
    f.write("FAKE NEWS CLASSIFICATION RESULTS\n")
    f.write("="*60 + "\n\n")
    for name, acc in results.items():
        f.write(f"{name}: {acc:.4f}\n")
    f.write(f"\nBest Model: {best_model}\n")

print("\nResults saved to 'model_results.txt'")