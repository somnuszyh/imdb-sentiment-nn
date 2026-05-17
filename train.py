import os
import json
import pickle
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# 1. 自动检测并加载 data 文件夹下的 CSV 文件
data_dir = "data"
csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
if not csv_files:
    raise FileNotFoundError("在 data/ 目录中未找到任何 CSV 数据集文件！")
data_path = os.path.join(data_dir, csv_files[0])
print(f"正在加载数据集: {data_path}")

df = pd.read_csv(data_path)

# 自动匹配列名
text_col = 'review' if 'review' in df.columns else ('text' if 'text' in df.columns else df.columns[0])
label_col = 'sentiment' if 'sentiment' in df.columns else ('label' if 'label' in df.columns else df.columns[1])

X = df[text_col].astype(str).tolist()
y = df[label_col].map({'positive': 1, 'negative': 0, 1: 1, 0: 0}).tolist()

# 2. 划分数据集与 TF-IDF 向量化
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 限制最大特征数为 5000，防止模型过大
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train).toarray()
X_test_vec = vectorizer.transform(X_test).toarray()

# 转为 PyTorch 张量
X_train_tensor = torch.tensor(X_train_vec, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test_vec, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# 构建 DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 3. 定义前馈神经网络 (MLP)
class SentimentNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(SentimentNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

input_size = X_train_vec.shape[1]
model = SentimentNN(input_dim=input_size)

# 4. 训练模型
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

print("开始训练模型...")
model.train()
for epoch in range(5):  # 快速迭代 5 个 Epoch
    epoch_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}/5 - Loss: {epoch_loss/len(train_loader):.4f}")

# 5. 评估模型
model.eval()
with torch.no_grad():
    test_preds = model(X_test_tensor)
    test_preds_cls = (test_preds > 0.5).float()
    accuracy = (test_preds_cls == y_test_tensor).float().mean().item()
print(f"测试集准确率: {accuracy:.4f}")

# 6. 强制保存老师要求的 4 个产物文件
os.makedirs("model", exist_ok=True)

# (1) 模型权重
torch.save(model.state_dict(), "model/model.pt")

# (2) 向量化器
with open("model/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

# (3) 配置文件
config = {"input_dim": input_size, "hidden_dim": 64}
with open("model/config.json", "w") as f:
    json.dump(config, f)

# (4) 指标文件
metrics = {"test_accuracy": accuracy, "final_loss": epoch_loss/len(train_loader)}
with open("model/metrics.json", "w") as f:
    json.dump(metrics, f)

print("所有模型产物已成功保存至 model/ 文件夹！")