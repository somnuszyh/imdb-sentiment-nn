import torch
import pickle
import json
import torch.nn as nn
import os

# 1. 定义与训练时完全一致的网络结构
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

def predict_sentiment(text):
    # 路径配置（确保指向 model 文件夹）
    model_dir = "model"
    
    # 2. 加载必要的产物
    with open(f"{model_dir}/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    
    with open(f"{model_dir}/config.json", "r") as f:
        config = json.load(f)
    
    # 3. 初始化模型并加载权重
    model = SentimentNN(config["input_dim"], config["hidden_dim"])
    model.load_state_dict(torch.load(f"{model_dir}/model.pt"))
    model.eval()

    # 4. 预处理输入文本
    text_vec = vectorizer.transform([text]).toarray()
    text_tensor = torch.tensor(text_vec, dtype=torch.float32)

    # 5. 推理
    with torch.no_grad():
        output = model(text_tensor).item()
        sentiment = "Positive" if output > 0.5 else "Negative"
        confidence = output if output > 0.5 else 1 - output
        
    return sentiment, confidence

if __name__ == "__main__":
    # 测试一下效果
    test_text = "This movie was absolutely fantastic and I loved the plot!"
    # 如果本地还没跑过训练，这里会报错，因为没生成 model/ 文件夹，这很正常。
    # 云端 Action 跑完后，你可以在本地跑训练，然后用这个脚本测试。
    if os.path.exists("model/model.pt"):
        label, score = predict_sentiment(test_text)
        print(f"Text: {test_text}")
        print(f"Prediction: {label} (Confidence: {score:.4f})")
    else:
        print("Model files not found. Please run train.py first to generate the model.")