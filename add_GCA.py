import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from torch.utils.data import DataLoader, Dataset
import gc
import os

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 跨模态交叉注意力
class CrossAttention(nn.Module):
    def __init__(self, feature_dim, num_heads=8):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        
        # Q, K, V 的线性变换层
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        
        self.fc_out = nn.Linear(feature_dim, feature_dim)

    def forward(self, query_seq, key_value_seq):
        # query_seq: 查询模态的序列
        # key_value_seq: 被查询模态的序列
        # 输入形状: (batch, seq_len, features)
        
        batch_size = query_seq.shape[0]

        Q = self.query(query_seq)
        K = self.key(key_value_seq)
        V = self.value(key_value_seq)
        
        # 分割成多头
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # 计算注意力得分
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / (self.head_dim ** 0.5)
        attention = torch.softmax(energy, dim=-1)
        
        # 加权求和
        x = torch.matmul(attention, V)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.num_heads * self.head_dim)
        
        # 输出线性层
        x = self.fc_out(x)
        return x

# 门控交叉注意力融合
class GatedCrossAttentionFusion(nn.Module):
    def __init__(self, feature_dim, num_heads=8, dropout=0.1):
        super(GatedCrossAttentionFusion, self).__init__()
        # 交叉注意力层
        self.cross_attention = CrossAttention(feature_dim, num_heads)
        # 层归一化
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        # 门控机制
        self.gate = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq1, seq2):
        # seq1 查询 seq2
        # seq1: (batch, seq_len, features)
        # seq2: (batch, seq_len, features)
        
        # 1. 计算交叉注意力信息
        attention_out = self.cross_attention(seq1, seq2)
        
        # 2. 残差连接和归一化
        seq1_norm = self.norm1(seq1)
        fused_info = self.dropout(attention_out)
        
        # 3. 门控融合
        gate_input = torch.cat([seq1_norm, fused_info], dim=-1)
        g = self.gate(gate_input)
        
        # 使用门控权重融合原始信息和注意力信息
        enhanced_seq = g * seq1_norm + (1 - g) * fused_info
        
        # 再次进行归一化
        return self.norm2(enhanced_seq)

class CrossModalClassifier(nn.Module):
    def __init__(self, feature_dim, output_size, dropout=0.3):
        super(CrossModalClassifier, self).__init__()
        
        # 文本和图像的特征投影层
        self.text_proj = nn.Linear(feature_dim, 512)
        self.image_proj = nn.Linear(feature_dim, 512)
        
        # 使用门控融合模块
        self.fusion_module_T_attends_I = GatedCrossAttentionFusion(feature_dim=512)
        self.fusion_module_I_attends_T = GatedCrossAttentionFusion(feature_dim=512)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),  # 融合后是1024维
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, output_size)
        )
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
    
    def forward(self, text_features, image_features):
        # 输入形状: (batch, feature_dim)
        
        # 添加序列维度 (batch, 1, feature_dim)
        text_seq = text_features.unsqueeze(1)
        image_seq = image_features.unsqueeze(1)
        
        # 特征投影
        text_proj = self.text_proj(text_seq)  # (batch, 1, 512)
        image_proj = self.image_proj(image_seq)  # (batch, 1, 512)
        
        # 应用交叉注意力融合
        enhanced_text = self.fusion_module_T_attends_I(text_proj, image_proj)
        enhanced_image = self.fusion_module_I_attends_T(image_proj, text_proj)
        
        # 拼接增强后的特征并去除序列维度
        fused_features = torch.cat([enhanced_text, enhanced_image], dim=-1).squeeze(1)  # (batch, 1024)
        
        return self.classifier(fused_features)

# Feature Engineering Functions 
def get_cyclical_time_features(timestamp):
    """从时间戳提取周期性时间特征"""
    if timestamp is None:
        return np.zeros(2)
    try:
        hour = timestamp.hour
        hour_norm = 2 * np.pi * hour / 24
        hour_sin = np.sin(hour_norm)
        hour_cos = np.cos(hour_norm)
        return np.array([hour_sin, hour_cos])
    except:
        return np.zeros(2)

# 自定义数据集类 - 适配微博谣言检测数据结构
class WeiboRumorDataset(Dataset):
    def __init__(self, data):
        """
        适配微博谣言检测数据的数据集类
        每个样本是一条微博，包含文本特征和图像特征
        """
        self.data = data
        self.text_feature_dim = 1024  # CLIP文本特征维度
        self.image_feature_dim = 1024  # CLIP图像特征维度
        self.time_feature_dim = 2
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # 文本特征处理
        text_emb = sample["text_embedding"]
        if text_emb is None:
            text_emb = np.zeros(self.text_feature_dim)
        
        # 图像特征处理 - 取所有图像特征的平均
        if sample["image_embeddings"]:
            img_emb = np.mean(sample["image_embeddings"], axis=0)
        else:
            img_emb = np.zeros(self.image_feature_dim)
        
        # 时间特征
        time_feat = get_cyclical_time_features(sample["timestamp"])
        
        # 构建最终特征向量
        # 文本特征 + 时间特征
        text_final = np.concatenate([text_emb, time_feat]).astype(np.float32)
        
        # 图像特征 + 时间特征  
        image_final = np.concatenate([img_emb, time_feat]).astype(np.float32)
        
        # 标签处理 - 确保是整数类型
        label = int(sample["label"])
        
        return text_final, image_final, label

# 训练和评估函数
def train_and_evaluate(train_data, val_data, test_data, batch_size=32, num_epochs=100, patience=20):
    print("===== Creating datasets =====")
    
    # 创建数据集
    train_dataset = WeiboRumorDataset(train_data)
    val_dataset = WeiboRumorDataset(val_data)
    test_dataset = WeiboRumorDataset(test_data)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # 模型初始化 - 输入维度为1026 (1024+2)
    input_feature_dim = 1026
    output_size = 2  # 二分类：谣言 vs 非谣言
    
    model = CrossModalClassifier(
        feature_dim=input_feature_dim,
        output_size=output_size,
        dropout=0.3
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True)

    # 早停初始化
    best_f1 = 0.0
    epochs_no_improve = 0
    best_model_weights = None
    
    print(f"Starting training for {num_epochs} epochs with batch size {batch_size}...")
    print(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}, Test samples: {len(test_data)}")
    
    # 训练历史记录
    train_losses = []
    val_f1_scores = []
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        total_loss = 0.0
        total_samples = 0
        
        for text_batch, image_batch, labels_batch in train_loader:
            text_batch = text_batch.to(device, non_blocking=True)
            image_batch = image_batch.to(device, non_blocking=True)
            labels_batch = labels_batch.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            outputs = model(text_batch, image_batch)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item() * text_batch.size(0)
            total_samples += text_batch.size(0)
        
        avg_train_loss = total_loss / total_samples
        train_losses.append(avg_train_loss)
        
        # 验证阶段
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for text_batch, image_batch, labels_batch in val_loader:
                text_batch = text_batch.to(device, non_blocking=True)
                image_batch = image_batch.to(device, non_blocking=True)
                labels_batch = labels_batch.to(device, non_blocking=True)
                
                outputs = model(text_batch, image_batch)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels_batch.cpu().numpy())
        
        val_f1 = f1_score(all_labels, all_preds, average='binary')
        val_f1_scores.append(val_f1)
        
        # 更新学习率
        scheduler.step(val_f1)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val F1: {val_f1:.4f}')

        # 早停逻辑
        if val_f1 > best_f1:
            best_f1 = val_f1
            epochs_no_improve = 0
            best_model_weights = copy.deepcopy(model.state_dict())
            print(f'New best F1 score: {best_f1:.4f}. Saving model...')
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f'Early stopping triggered after {epoch+1} epochs.')
            break
            
        # 手动释放内存
        torch.cuda.empty_cache()
        gc.collect()
    
    # 绘制训练曲线
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(val_f1_scores)
    plt.title('Validation F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close()
            
    # 最终评估
    print("\nLoading best model for final evaluation...")
    model.load_state_dict(best_model_weights)
    
    # 在测试集上评估
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for text_batch, image_batch, labels_batch in test_loader:
            text_batch = text_batch.to(device, non_blocking=True)
            image_batch = image_batch.to(device, non_blocking=True)
            labels_batch = labels_batch.to(device, non_blocking=True)
            
            outputs = model(text_batch, image_batch)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # 计算指标
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='binary')
    cm = confusion_matrix(all_labels, all_preds)
    
    print(f'\n--- Final Evaluation (Best Model) ---')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print('Confusion Matrix:')
    print(cm)
    
    # 类别分布
    unique, counts = np.unique(all_labels, return_counts=True)
    print(f'Class distribution: {dict(zip(unique, counts))}')
    
    # 绘制混淆矩阵
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Rumor', 'Rumor'],
                yticklabels=['Non-Rumor', 'Rumor'])
    plt.title('Confusion Matrix - Weibo Rumor Detection')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('weibo_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return accuracy, f1

# 主函数 
def main():
    # 加载新数据集
    try:
        with open("weibo_rumor_clip_features.pkl", "rb") as f:
            data = pickle.load(f)
            train_data = data["train"]
            val_data = data["val"] 
            test_data = data["test"]
            print(f"Loaded dataset: Train size={len(train_data)}, Val size={len(val_data)}, Test size={len(test_data)}")
            
            # 打印数据统计信息
            train_labels = [sample["label"] for sample in train_data]
            val_labels = [sample["label"] for sample in val_data]
            test_labels = [sample["label"] for sample in test_data]
            
            print(f"Train class distribution: {np.unique(train_labels, return_counts=True)}")
            print(f"Val class distribution: {np.unique(val_labels, return_counts=True)}")
            print(f"Test class distribution: {np.unique(test_labels, return_counts=True)}")
            
            # 释放原始数据内存
            del data
            gc.collect()
    except FileNotFoundError:
        print("Error: Dataset file not found. Please check the path.")
        return
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # 设置batch_size
    batch_size = 64
    
    # 训练和评估模型
    accuracy, f1 = train_and_evaluate(train_data, val_data, test_data, batch_size=batch_size)
    
    # 打印最终结果
    print("\n===== Final Results =====")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")

if __name__ == "__main__":
    # 设置环境变量以减少内存碎片
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    
    # 清理内存
    torch.cuda.empty_cache()
    gc.collect()
    
    main()
