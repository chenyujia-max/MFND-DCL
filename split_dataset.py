import os
import pickle
import numpy as np
from datetime import datetime
from PIL import Image
from tqdm import tqdm
import torch
import cn_clip.clip as clip
from sklearn.model_selection import train_test_split

# 设备配置
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 加载Chinese-CLIP模型
model, preprocess = clip.load_from_name("ViT-H-14", device=device, download_root='./cnclipmodel')
model.eval()

def parse_weibo_time(time_str):
    """解析微博时间格式"""
    try:
        # 微博时间格式: "2013-04-16 21:34"
        return datetime.strptime(time_str, '%Y-%m-%d %H:%M')
    except:
        return datetime.now()

def truncate_text(text, max_length=50):
    """截断文本到指定长度"""
    if len(text) > max_length:
        return text[:max_length] + "..."
    return text

def parse_tweet_blocks(file_path, label):
    """解析三行一组的微博数据"""
    tweets = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        # 每三行一组
        for i in range(0, len(lines), 3):
            if i + 2 >= len(lines):
                break
                
            # 第一行：元数据
            line1 = lines[i].strip()
            # 第二行：图片URL
            line2 = lines[i+1].strip()
            # 第三行：文本内容
            line3 = lines[i+2].strip()
            
            # 解析第一行
            parts = line1.split('|')
            if len(parts) < 5:  # 至少需要包含ID、用户名、时间等基本信息
                continue
                
            tweet_id = parts[0]
            username = parts[1]
            timestamp = parts[4]
            
            # 解析第二行：图片URL（可能多个，用|分隔）
            image_urls = []
            if line2 and line2 != 'null':
                url_parts = line2.split('|')
                for url in url_parts:
                    url = url.strip()
                    if url.startswith('http') and url != 'null':
                        image_urls.append(url)
            
            # 第三行就是文本内容
            text = line3
            
            tweets.append({
                "tweet_id": tweet_id,
                "username": username,
                "timestamp": timestamp,
                "text": text,
                "image_urls": image_urls,
                "label": label
            })
            
    except FileNotFoundError:
        print(f"Warning: File {file_path} not found")
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
    
    return tweets

def load_weibo_dataset(base_path):
    """
    加载微博谣言检测数据集
    
    Args:
        base_path: 数据集根目录路径
    """
    # 路径设置
    rumor_images_dir = os.path.join(base_path, "rumor_images")
    nonrumor_images_dir = os.path.join(base_path, "nonrumor_images")
    tweets_dir = os.path.join(base_path, "tweets")
    
    # 检查图片目录是否存在
    print(f"Rumor images dir exists: {os.path.exists(rumor_images_dir)}")
    print(f"Nonrumor images dir exists: {os.path.exists(nonrumor_images_dir)}")
    
    # 读取微博文本数据
    print("Loading tweet texts...")
    
    # 加载所有微博数据
    all_tweets = []
    
    # 谣言微博
    rumor_files = [
        os.path.join(tweets_dir, "train_rumor.txt"),
        os.path.join(tweets_dir, "test_rumor.txt")
    ]
    for file_path in rumor_files:
        tweets = parse_tweet_blocks(file_path, label=1)  # 1表示谣言
        all_tweets.extend(tweets)
        print(f"Loaded {len(tweets)} rumor tweets from {os.path.basename(file_path)}")
    
    # 非谣言微博
    nonrumor_files = [
        os.path.join(tweets_dir, "train_nonrumor.txt"),
        os.path.join(tweets_dir, "test_nonrumor.txt")
    ]
    for file_path in nonrumor_files:
        tweets = parse_tweet_blocks(file_path, label=0)  # 0表示非谣言
        all_tweets.extend(tweets)
        print(f"Loaded {len(tweets)} nonrumor tweets from {os.path.basename(file_path)}")
    
    print(f"Loaded {len(all_tweets)} tweets in total")
    
    # 直接按照txt文件的train和test进行划分
    train_tweets = []
    test_tweets = []
    
    # 从文件名判断是训练集还是测试集
    for tweet in all_tweets:
        # 这里我们根据数据来源判断，实际应该根据你的数据组织方式调整
        # 假设train_rumor.txt和train_nonrumor.txt是训练集，test_rumor.txt和test_nonrumor.txt是测试集
        # 由于parse_tweet_blocks没有记录来源，我们需要重新组织数据
        
        # 重新加载并分别存储训练集和测试集
        pass
    
    # 重新组织数据：分别加载训练集和测试集
    print("Reorganizing data by train/test split...")
    
    # 训练集
    train_rumor = parse_tweet_blocks(os.path.join(tweets_dir, "train_rumor.txt"), label=1)
    train_nonrumor = parse_tweet_blocks(os.path.join(tweets_dir, "train_nonrumor.txt"), label=0)
    train_data = train_rumor + train_nonrumor
    print(f"Training set: {len(train_data)} tweets ({len(train_rumor)} rumor, {len(train_nonrumor)} nonrumor)")
    
    # 测试集
    test_rumor = parse_tweet_blocks(os.path.join(tweets_dir, "test_rumor.txt"), label=1)
    test_nonrumor = parse_tweet_blocks(os.path.join(tweets_dir, "test_nonrumor.txt"), label=0)
    test_data = test_rumor + test_nonrumor
    print(f"Test set: {len(test_data)} tweets ({len(test_rumor)} rumor, {len(test_nonrumor)} nonrumor)")
    
    # 从训练集中划分验证集 (8:2 -> 训练集80%, 验证集20%)
    print("Splitting training set into train/validation (80%/20%)...")
    train_ids = [tweet["tweet_id"] for tweet in train_data]
    train_labels = [tweet["label"] for tweet in train_data]
    
    # 分层抽样，保持类别比例
    train_ids_final, val_ids = train_test_split(
        train_ids, test_size=0.2, random_state=42, stratify=train_labels
    )
    
    print(f"Final train set: {len(train_ids_final)} tweets")
    print(f"Validation set: {len(val_ids)} tweets")
    print(f"Test set: {len(test_data)} tweets")
    
    # 构建ID到微博数据的映射
    tweet_dict = {tweet["tweet_id"]: tweet for tweet in train_data + test_data}
    
    # 获取图片目录中的所有图片文件名
    print("Scanning image directories...")
    rumor_images = set(os.listdir(rumor_images_dir)) if os.path.exists(rumor_images_dir) else set()
    nonrumor_images = set(os.listdir(nonrumor_images_dir)) if os.path.exists(nonrumor_images_dir) else set()
    
    print(f"Found {len(rumor_images)} rumor images, {len(nonrumor_images)} nonrumor images")
    
    # 根据ID划分数据集
    def build_dataset(id_list, name):
        dataset = []
        missing_count = 0
        image_found_count = 0
        
        for tweet_id in tqdm(id_list, desc=f"Building {name} set"):
            if tweet_id in tweet_dict:
                tweet_data = tweet_dict[tweet_id]
                
                # 构建图片路径
                image_paths = []
                for image_url in tweet_data["image_urls"]:
                    # 从URL中提取文件名
                    image_filename = os.path.basename(image_url)
                    
                    # 根据标签确定在哪个目录查找
                    if tweet_data["label"] == 1:  # 谣言
                        image_path = os.path.join(rumor_images_dir, image_filename)
                    else:  # 非谣言
                        image_path = os.path.join(nonrumor_images_dir, image_filename)
                    
                    # 检查文件是否存在
                    if os.path.exists(image_path):
                        image_paths.append(image_path)
                        image_found_count += 1
                    else:
                        # 如果直接路径不存在，尝试在对应的图片集合中查找
                        if tweet_data["label"] == 1 and image_filename in rumor_images:
                            image_paths.append(os.path.join(rumor_images_dir, image_filename))
                            image_found_count += 1
                        elif tweet_data["label"] == 0 and image_filename in nonrumor_images:
                            image_paths.append(os.path.join(nonrumor_images_dir, image_filename))
                            image_found_count += 1
                
                # 添加到数据集
                dataset.append({
                    "tweet_id": tweet_id,
                    "username": tweet_data["username"],
                    "text": tweet_data["text"],
                    "image_paths": image_paths,
                    "timestamp": parse_weibo_time(tweet_data["timestamp"]),
                    "label": tweet_data["label"]
                })
            else:
                missing_count += 1
        
        print(f"{name}: {len(dataset)} samples, {missing_count} missing tweets, {image_found_count} images found")
        
        # 打印一些有图片的样本示例
        samples_with_images = [d for d in dataset if d["image_paths"]]
        if samples_with_images:
            print(f"  Examples with images (first 3):")
            for sample in samples_with_images[:3]:
                print(f"    Tweet {sample['tweet_id']}: {len(sample['image_paths'])} images")
        
        return dataset
    
    # 构建训练、验证、测试集
    train_data_final = build_dataset(train_ids_final, "train")
    val_data_final = build_dataset(val_ids, "validation")
    test_data_final = build_dataset([tweet["tweet_id"] for tweet in test_data], "test")
    
    return train_data_final, val_data_final, test_data_final

def extract_features(data):
    """提取文本和图像特征"""
    processed_data = []
    
    for tweet in tqdm(data, desc="Extracting features"):
        tweet_features = {
            "tweet_id": tweet["tweet_id"],
            "username": tweet["username"],
            "label": tweet["label"],
            "timestamp": tweet["timestamp"],
            "text_embedding": None,
            "image_embeddings": []
        }
        
        # 文本特征提取
        if tweet["text"] and tweet["text"] != "null":
            try:
                truncated_text = truncate_text(tweet["text"], max_length=50)
                with torch.no_grad():
                    text_tokens = clip.tokenize([truncated_text], context_length=64).to(device)
                    text_features = model.encode_text(text_tokens)
                    tweet_features["text_embedding"] = text_features.cpu().numpy()[0]
            except Exception as e:
                print(f"Text processing error for tweet {tweet['tweet_id']}: {e}")
        
        # 图像特征提取
        image_embeddings_count = 0
        for img_path in tweet["image_paths"]:
            try:
                image = Image.open(img_path).convert("RGB")
                image_tensor = preprocess(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    image_features = model.encode_image(image_tensor)
                    tweet_features["image_embeddings"].append(image_features.cpu().numpy()[0])
                    image_embeddings_count += 1
            except Exception as e:
                print(f"Image processing error for {img_path}: {e}")
                continue
        
        # 仅保存有有效特征的微博
        if tweet_features["text_embedding"] is not None or tweet_features["image_embeddings"]:
            processed_data.append(tweet_features)
    
    print(f"Processed {len(processed_data)} tweets with valid features")
    
    # 统计有图片特征的微博数量
    tweets_with_images = sum(1 for tweet in processed_data if tweet["image_embeddings"])
    print(f"Tweets with image features: {tweets_with_images}")
    
    return processed_data

def process_weibo_dataset(base_path, output_file):
    """处理微博数据集并保存为pkl文件"""
    
    # 加载数据集
    print("Loading Weibo rumor detection dataset...")
    train_data, val_data, test_data = load_weibo_dataset(base_path)
    
    # 提取特征
    print("Extracting features...")
    processed_train = extract_features(train_data)
    processed_val = extract_features(val_data)
    processed_test = extract_features(test_data)
    
    # 保存为pickle文件
    with open(output_file, "wb") as f:
        pickle.dump({
            "train": processed_train,
            "val": processed_val,
            "test": processed_test,
            "feature_dim": 512  # CLIP特征维度
        }, f)
    
    print(f"Saved processed data to {output_file}")
    print(f"Train size: {len(processed_train)} tweets")
    print(f"Validation size: {len(processed_val)} tweets")
    print(f"Test size: {len(processed_test)} tweets")
    
    # 打印图片特征统计
    def count_image_features(data, name):
        with_images = sum(1 for tweet in data if tweet["image_embeddings"])
        print(f"{name}: {with_images} tweets have image features ({with_images/len(data)*100:.1f}%)")
    
    count_image_features(processed_train, "Train")
    count_image_features(processed_val, "Validation")
    count_image_features(processed_test, "Test")
    
    return processed_train, processed_val, processed_test

# 使用示例
if __name__ == "__main__":
    # 设置路径
    base_path = "weibo"  # 修改为实际的weibo数据集路径
    output_file = "weibo_rumor_clip_features.pkl"
    
    # 处理数据集
    train_data, val_data, test_data = process_weibo_dataset(base_path, output_file)
