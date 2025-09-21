import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import numpy as np
import random
from pathlib import Path
import matplotlib.pyplot as plt

# 设置随机种子以确保可重现性
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed()

# 音频数据处理配置
SAMPLE_RATE = 22050  # 音频采样率
N_FFT = 1024         # FFT点数
HOP_LENGTH = 256     # 帧移
N_MELS = 128         # 梅尔频谱维度
MAX_LENGTH = 512     # 最大序列长度

# 自定义数据集类
class AudioDataset(Dataset):
    def __init__(self, data_dir, sample_rate=SAMPLE_RATE, n_mels=N_MELS, max_length=MAX_LENGTH):
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.max_length = max_length
        
        # 获取所有音频文件（包括子目录）
        self.file_list = []
        for ext in ["*.wav", "*.WAV", "*.mp3", "*.flac"]:
            self.file_list.extend(list(self.data_dir.rglob(ext)))
        
        if len(self.file_list) == 0:
            raise ValueError(f"在目录 {data_dir} 中未找到任何音频文件")
        
        print(f"找到 {len(self.file_list)} 个音频文件")
        
        # 计算梅尔频谱转换
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=n_mels
        )
        
        # 收集所有音频文件的频谱统计信息用于标准化
        self.compute_stats()
    
    def compute_stats(self):
        all_mels = []
        valid_files = 0
        
        for i, file_path in enumerate(self.file_list):
            try:
                waveform, sr = torchaudio.load(file_path)
                print(f"处理文件 {i+1}/{len(self.file_list)}: {file_path.name}, 采样率: {sr}")
                
                if sr != self.sample_rate:
                    waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
                
                # 转换为单声道
                if waveform.size(0) > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                
                # 计算梅尔频谱
                mel = self.mel_transform(waveform)
                mel = torchaudio.functional.amplitude_to_DB(mel, multiplier=10, amin=1e-10, db_multiplier=0)
                all_mels.append(mel)
                valid_files += 1
            except Exception as e:
                print(f"处理文件 {file_path} 时出错: {e}, 跳过该文件")
        
        if valid_files == 0:
            raise ValueError("未能成功处理任何音频文件")
        
        print(f"成功处理 {valid_files}/{len(self.file_list)} 个文件")
        
        # 计算均值和标准差
        all_mels = torch.cat(all_mels, dim=1)
        self.mel_mean = all_mels.mean()
        self.mel_std = all_mels.std()
        
        print(f"梅尔频谱均值: {self.mel_mean:.4f}, 标准差: {self.mel_std:.4f}")
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        
        # 尝试加载文件，如果失败则返回空数据
        try:
            waveform, sr = torchaudio.load(file_path)
            
            # 重采样到目标采样率
            if sr != self.sample_rate:
                waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
            
            # 转换为单声道
            if waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # 计算梅尔频谱
            mel = self.mel_transform(waveform)
            mel = torchaudio.functional.amplitude_to_DB(mel, multiplier=10, amin=1e-10, db_multiplier=0)
            
            # 标准化
            mel = (mel - self.mel_mean) / self.mel_std
            
            # 转置并截取合适长度 (时间步, 特征维度)
            mel = mel.squeeze(0).T
            
            # 确保序列长度合适
            if mel.size(0) > self.max_length:
                # 随机截取一段
                start = random.randint(0, mel.size(0) - self.max_length)
                mel = mel[start:start + self.max_length, :]
            else:
                # 填充到最大长度
                padding = self.max_length - mel.size(0)
                mel = torch.nn.functional.pad(mel, (0, 0, 0, padding))
            
            # 创建输入和目标（预测下一个时间步）
            input_seq = mel[:-1, :]  # 除最后一个时间步外的所有时间步
            target_seq = mel[1:, :]  # 除第一个时间步外的所有时间步
            
            return input_seq, target_seq
            
        except Exception as e:
            print(f"加载文件 {file_path} 时出错: {e}, 返回空数据")
            # 返回空数据
            empty_input = torch.zeros(self.max_length-1, self.n_mels)
            empty_target = torch.zeros(self.max_length-1, self.n_mels)
            return empty_input, empty_target

# 基于Transformer的音频续写模型
class AudioTransformer(nn.Module):
    def __init__(self, input_dim, model_dim=256, num_heads=8, num_layers=6, dropout=0.1):
        super(AudioTransformer, self).__init__()
        self.input_dim = input_dim
        self.model_dim = model_dim
        
        # 输入投影层
        self.input_projection = nn.Linear(input_dim, model_dim)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(model_dim, dropout)
        
        # Transformer编码器
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # 输出层
        self.output_layer = nn.Linear(model_dim, input_dim)
        
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # 投影输入到模型维度
        src = self.input_projection(src)
        
        # 添加位置编码
        src = self.pos_encoder(src)
        
        # 通过Transformer编码器
        output = self.transformer_encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        
        # 投影回原始维度
        output = self.output_layer(output)
        
        return output

# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# 训练函数
def train_audio_model():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建数据集和数据加载器
    try:
        dataset = AudioDataset("ds")
        print(f"数据集大小: {len(dataset)}")
        
        # 创建数据加载器
        dataloader = DataLoader(
            dataset, 
            batch_size=4, 
            shuffle=True, 
            num_workers=2,
            drop_last=True  # 丢弃最后一个不完整的批次
        )
        
    except Exception as e:
        print(f"创建数据集时出错: {e}")
        print("请确保 'ds' 目录存在且包含音频文件 (.wav, .mp3, .flac)")
        return
    
    # 初始化模型
    input_dim = N_MELS
    model = AudioTransformer(input_dim=input_dim).to(device)
    
    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数数量: {total_params}")
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    
    # 训练参数
    num_epochs = 20
    print_interval = 5
    
    # 训练循环
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        batch_count = 0
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            # 跳过空批次
            if inputs.nelement() == 0:
                continue
                
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 前向传播
            outputs = model(inputs)
            
            # 计算损失
            loss = criterion(outputs, targets)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            
            if batch_idx % print_interval == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.6f}")
        
        if batch_count == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], 没有有效数据批次")
            continue
            
        # 计算平均损失
        avg_loss = total_loss / batch_count
        
        # 更新学习率
        scheduler.step()
        
        # 打印每个epoch的平均损失
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.6f}")
        
        # 每10个epoch保存一次模型
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'mel_mean': dataset.mel_mean,
                'mel_std': dataset.mel_std
            }, f"audio_transformer_epoch_{epoch+1}.pth")
    
    # 保存最终模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'mel_mean': dataset.mel_mean,
        'mel_std': dataset.mel_std
    }, "audio_transformer_final.pth")
    
    print("训练完成，模型已保存!")

if __name__ == "__main__":
    train_audio_model()