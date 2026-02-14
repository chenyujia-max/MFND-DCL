import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import copy
import os
import random
import gc
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

random.seed(129)
np.random.seed(129)
torch.manual_seed(129)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class CrossAttention(nn.Module):
    def __init__(self, feature_dim, num_heads=4):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.fc_out = nn.Linear(feature_dim, feature_dim)
        
    def forward(self, query_seq, key_value_seq):
        batch_size = query_seq.shape[0]
        Q, K, V = self.query(query_seq), self.key(key_value_seq), self.value(key_value_seq)
        
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / (self.head_dim ** 0.5)
        attention = torch.softmax(energy, dim=-1)
        
        x = torch.matmul(attention, V).permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.num_heads * self.head_dim)
        
        return self.fc_out(x)

class GatedCrossAttentionFusion(nn.Module):
    def __init__(self, feature_dim, num_heads=4, dropout=0.1):
        super(GatedCrossAttentionFusion, self).__init__()
        self.cross_attention = CrossAttention(feature_dim, num_heads)
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.gate = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, seq1, seq2):
        attention_out = self.cross_attention(self.norm1(seq1), self.norm1(seq2))
        fused_info = self.dropout(attention_out)
        
        gate_input = torch.cat([self.norm1(seq1), fused_info], dim=-1)
        g = self.gate(gate_input)
        
        enhanced_seq = g * self.norm1(seq1) + (1 - g) * fused_info
        return self.norm2(enhanced_seq)

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        batch_size = features.shape[0]
        features = nn.functional.normalize(features, p=2, dim=1)
        
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(features.device)
        identity_mask = torch.eye(batch_size, device=features.device)
        mask = mask - identity_mask  # Remove self-comparison

        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )
        
        logits = anchor_dot_contrast - torch.log(torch.exp(anchor_dot_contrast).sum(1, keepdim=True) + 1e-9)

        mask_sum = mask.sum(1)
        positive_samples_exist = mask_sum > 0
        
        if not positive_samples_exist.any():
            return torch.tensor(0.0, device=features.device)
            
        log_prob = (mask * logits).sum(1)[positive_samples_exist] / mask_sum[positive_samples_exist]
        
        loss = -log_prob.mean()
        return loss

class MultimodalConsistencyLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(MultimodalConsistencyLoss, self).__init__()
        self.temperature = temperature
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, text_features, image_features):
        text_features = nn.functional.normalize(text_features, p=2, dim=1)
        image_features = nn.functional.normalize(image_features, p=2, dim=1)
        
        logits_ti = torch.matmul(text_features, image_features.T) / self.temperature
        logits_it = torch.matmul(image_features, text_features.T) / self.temperature
        
        batch_size = text_features.shape[0]
        labels = torch.arange(batch_size, device=text_features.device)
        
        loss_t_i = self.loss_fn(logits_ti, labels)
        loss_i_t = self.loss_fn(logits_it, labels)
        
        return (loss_t_i + loss_i_t) / 2

class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, output_size, projection_dim=128, dropout=0.2):
        super(SimpleClassifier, self).__init__()
        self.text_branch = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, projection_dim)
        )
        
        self.image_branch = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, projection_dim)
        )
        
        self.fusion_module = GatedCrossAttentionFusion(projection_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(projection_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, output_size)
        )
        
        self.projection_head_text = nn.Sequential(
            nn.Linear(projection_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
        
        self.projection_head_image = nn.Sequential(
            nn.Linear(projection_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
        
        self.projection_head_user = nn.Sequential(
            nn.Linear(projection_dim * 2, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
    
    def forward(self, text_x, image_x):
        # Process text features
        text_features = self.text_branch(text_x)
        
        # Process image features
        image_features = self.image_branch(image_x)
        
        # Cross-modal fusion
        text_fused = self.fusion_module(text_features.unsqueeze(1), image_features.unsqueeze(1)).squeeze(1)
        image_fused = self.fusion_module(image_features.unsqueeze(1), text_features.unsqueeze(1)).squeeze(1)
        
        # Combine features
        fused = torch.cat([text_fused, image_fused], dim=1)
        
        # Projections for losses
        user_repr = self.projection_head_user(fused)
        text_consist_repr = self.projection_head_text(text_fused)
        image_consist_repr = self.projection_head_image(image_fused)
        
        # Final classification
        logits = self.classifier(fused)
        
        return logits, user_repr, text_consist_repr, image_consist_repr

def get_cyclical_time_features(timestamp):
    if timestamp is None: 
        return np.zeros(2)
    hour = timestamp.hour
    hour_norm = 2 * np.pi * hour / 24
    return np.array([np.sin(hour_norm), np.cos(hour_norm)])

class WeiboDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
        self.text_feature_dim = 1024
        self.time_feature_dim = 2
        self.total_feature_dim = self.text_feature_dim + self.time_feature_dim
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        item = self.data_list[idx]
        
        # Text features
        text_emb = item["text_embedding"] if item["text_embedding"] is not None else np.zeros(self.text_feature_dim)
        
        # Image features - average if multiple images
        img_emb = np.zeros(self.text_feature_dim)
        if item["image_embeddings"]:
            img_emb = np.mean(item["image_embeddings"], axis=0)
        
        # Time features
        time_feat = get_cyclical_time_features(item["timestamp"])
        
        # Combine features
        text_feature = np.concatenate((text_emb, time_feat))
        image_feature = np.concatenate((img_emb, time_feat))
        
        # Also return tweet_id for saving results
        return text_feature.astype(np.float32), image_feature.astype(np.float32), item["label"], item["tweet_id"]

def train_and_evaluate(train_data, test_data, model_name, config):
    print(f"\n===== Processing {model_name} features =====")

    train_dataset = WeiboDataset(train_data)
    test_dataset = WeiboDataset(test_data)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Model initialization
    input_dim = 1026  # 1024 features + 2 time features
    output_size = 2
    
    model = SimpleClassifier(
        input_dim=input_dim,
        output_size=output_size,
        projection_dim=config['projection_dim'],
        dropout=config['dropout']
    ).to(device)
    
    # Output model parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model initialized with {num_params/1e6:.2f}M parameters")
    
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=1e-5)
    criterion_ce = nn.CrossEntropyLoss()
    criterion_supcon = SupConLoss(temperature=config['temperature'])
    criterion_consistency = MultimodalConsistencyLoss(temperature=config['consistency_temperature'])
    
    best_f1 = 0.0
    epochs_no_improve = 0
    best_model_weights = None

    print(f"Starting training for {config['num_epochs']} epochs with batch size {config['batch_size']}...")
    for epoch in range(config['num_epochs']):
        model.train()
        total_loss, total_ce, total_sc, total_mc = 0, 0, 0, 0
        
        for text_batch, image_batch, labels_batch, _ in train_loader:
            text_batch = text_batch.to(device, non_blocking=True)
            image_batch = image_batch.to(device, non_blocking=True)
            labels_batch = labels_batch.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            logits, user_repr, text_consist_repr, image_consist_repr = model(text_batch, image_batch)
            
            loss_ce = criterion_ce(logits, labels_batch)
            loss_supcon = criterion_supcon(user_repr, labels_batch)
            loss_consistency = criterion_consistency(text_consist_repr, image_consist_repr)
            
            loss = loss_ce + config['lambda_supcon'] * loss_supcon + config['lambda_consistency'] * loss_consistency
            
            if torch.isnan(loss) or torch.isinf(loss):
                print("NaN or Inf loss detected. Skipping update.")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_ce += loss_ce.item()
            total_sc += loss_supcon.item()
            total_mc += loss_consistency.item()

        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for text_batch, image_batch, labels_batch, _ in test_loader:
                text_batch = text_batch.to(device, non_blocking=True)
                image_batch = image_batch.to(device, non_blocking=True)
                labels_batch = labels_batch.to(device, non_blocking=True)
                
                logits, _, _, _ = model(text_batch, image_batch)
                _, preds = torch.max(logits, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels_batch.cpu().numpy())
        
        # Calculate metrics
        val_f1 = f1_score(all_labels, all_preds, zero_division=0)
        
        print(f'Epoch [{epoch+1}/{config["num_epochs"]}], '
              f'Loss: {avg_loss:.4f} (CE: {total_ce/len(train_loader):.4f}, '
              f'SC: {total_sc/len(train_loader):.4f}, MC: {total_mc/len(train_loader):.4f}), '
              f'Val F1: {val_f1:.4f}')

        # Early stopping
        if val_f1 > best_f1:
            best_f1 = val_f1
            epochs_no_improve = 0
            best_model_weights = copy.deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= config['patience']:
            print(f'Early stopping triggered after {epoch+1} epochs.')
            break
    
    # Load best model for final evaluation
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
    
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    all_tweet_ids = []
    
    with torch.no_grad():
        for text_batch, image_batch, labels_batch, tweet_ids_batch in test_loader:
            text_batch = text_batch.to(device, non_blocking=True)
            image_batch = image_batch.to(device, non_blocking=True)
            labels_batch = labels_batch.to(device, non_blocking=True)
            
            logits, _, _, _ = model(text_batch, image_batch)
            probs = torch.softmax(logits, dim=1)
            _, preds = torch.max(logits, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of rumor class
            all_tweet_ids.extend(tweet_ids_batch)
    
    # 保存测试集中本身为rumor并且识别正确的样本
    correct_rumor_results = []
    for i in range(len(all_labels)):
        if all_labels[i] == 1 and all_preds[i] == 1:  # 真实标签为rumor且预测正确
            correct_rumor_results.append({
                'tweet_id': all_tweet_ids[i],
                'pred_prob': all_probs[i],
                'true_label': all_labels[i],
                'pred_label': all_preds[i]
            })
    
    # 将结果保存到txt文件
    output_filename = f'{model_name}_correct_rumor_predictions.txt'
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(f"# 测试集中真实标签为rumor且预测正确的样本\n")
        f.write(f"# 共找到 {len(correct_rumor_results)} 个符合条件的样本\n")
        f.write(f"# tweet_id, 预测概率, 真实标签, 预测标签\n")
        for result in correct_rumor_results:
            f.write(f"{result['tweet_id']}, {result['pred_prob']:.6f}, {result['true_label']}, {result['pred_label']}\n")
    
    print(f"\n已保存 {len(correct_rumor_results)} 个真实标签为rumor且预测正确的样本到 {output_filename}")
    
    # Calculate final metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    
    print(f'\n--- {model_name} Final Evaluation (Best Model) ---')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print('Confusion Matrix:\n', cm)
    
    # Plot confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Rumor', 'Rumor'], 
                yticklabels=['Non-Rumor', 'Rumor'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'{model_name.lower()}_confusion_matrix.png')
    plt.close()
    
    # Plot probability distribution
    plt.figure(figsize=(8, 6))
    sns.histplot(data={'Probability': all_probs, 'True Label': all_labels}, 
                 x='Probability', hue='True Label', bins=30, kde=True, 
                 palette={0: 'blue', 1: 'red'})
    plt.title(f'Prediction Probability Distribution - {model_name}')
    plt.xlabel('Probability of Rumor')
    plt.tight_layout()
    plt.savefig(f'{model_name.lower()}_probability_distribution.png')
    plt.close()
    
    return accuracy, f1

def main():
    config = {
        'batch_size': 128,
        'num_epochs': 100,
        'patience': 15,
        'learning_rate': 1e-4,
        'projection_dim': 128,
        'dropout': 0.3,
        'temperature': 0.2,
        'consistency_temperature': 0.1,
        'lambda_supcon': 0.5,
        'lambda_consistency': 0.3
    }

    try:
        with open("weibo_rumor_clip_features.pkl", "rb") as f:
            data = pickle.load(f)
            train_data = data["train"]
            val_data = data["val"]
            test_data = data["test"]
            print(f"Loaded data: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    except FileNotFoundError:
        print("Error: Data file not found.")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Train and evaluate
    acc, f1 = train_and_evaluate(train_data, test_data, "CLIP_SimpleClassifier", config)
    
    print("\n===== Model Results =====")
    print(f"CLIP Simple Classifier - Accuracy: {acc:.4f}, F1: {f1:.4f}")

if __name__ == "__main__":
    main()
