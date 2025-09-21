import torch
import torchaudio
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

# 使用与训练时相同的配置
SAMPLE_RATE = 22050
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 128
MAX_LENGTH = 512

# 加载训练好的模型进行推理
class AudioInference:
    def __init__(self, model_path, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        print(f"使用设备: {self.device}")
        
        # 加载模型检查点
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            print("模型加载成功!")
        except Exception as e:
            print(f"加载模型时出错: {e}")
            return
        
        # 初始化模型
        self.model = AudioTransformer(input_dim=N_MELS).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # 加载标准化参数
        self.mel_mean = checkpoint['mel_mean']
        self.mel_std = checkpoint['mel_std']
        
        print(f"梅尔频谱均值: {self.mel_mean:.4f}, 标准差: {self.mel_std:.4f}")
        
        # 梅尔频谱转换
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS
        )
        
        # 逆梅尔频谱转换（改进版本）
        self.inverse_mel_transform = torchaudio.transforms.GriffinLim(
            n_fft=N_FFT,
            n_iter=64,  # 增加迭代次数
            hop_length=HOP_LENGTH,
            power=1.0,
            rand_init=True  # 随机初始化相位
        )
    
    def preprocess_audio(self, audio_path):
        """预处理输入音频文件"""
        try:
            waveform, sr = torchaudio.load(audio_path)
            print(f"加载音频: {audio_path}, 采样率: {sr}, 形状: {waveform.shape}")
            
            # 重采样
            if sr != SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
                waveform = resampler(waveform)
            
            # 转换为单声道
            if waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            return waveform
        except Exception as e:
            print(f"预处理音频时出错: {e}")
            return None
    
    def waveform_to_mel(self, waveform):
        """将波形转换为梅尔频谱"""
        mel = self.mel_transform(waveform)
        mel = torchaudio.functional.amplitude_to_DB(mel, multiplier=10, amin=1e-10, db_multiplier=0)
        mel = (mel - self.mel_mean) / self.mel_std
        return mel.squeeze(0).T  # (时间步, 特征维度)
    
    def mel_to_waveform(self, mel):
        """将梅尔频谱转换回波形（改进版本）"""
        # 反标准化
        mel = mel * self.mel_std + self.mel_mean
        
        # 将DB转换回振幅
        mel = torchaudio.functional.DB_to_amplitude(mel, ref=1.0, power=1.0)
        
        # 将梅尔频谱转换为幅度谱
        # 创建梅尔滤波器组
        mel_filters = torchaudio.functional.melscale_fbanks(
            n_freqs=N_FFT//2 + 1,
            f_min=0.0,
            f_max=SAMPLE_RATE/2,
            n_mels=N_MELS,
            sample_rate=SAMPLE_RATE,
            norm='slaney'
        ).to(mel.device)
        
        # 计算梅尔滤波器组的伪逆
        mel_filters_inv = torch.linalg.pinv(mel_filters)
        
        # 将梅尔频谱转换为幅度谱
        # mel 的形状是 (时间步, 128)
        # mel_filters_inv 的形状是 (128, 513)
        # 结果是 (时间步, 513)
        mag_spec = torch.matmul(mel, mel_filters_inv)
        
        # 使用Griffin-Lim算法重建波形
        # GriffinLim 期望输入形状为 (频率, 时间步)，所以需要转置
        waveform = self.inverse_mel_transform(mag_spec.T.unsqueeze(0))
        
        # 归一化波形
        waveform = waveform / waveform.abs().max()
        
        return waveform
    
    def continue_audio(self, input_audio_path, output_length_seconds, output_path="continued_audio.wav"):
        """续写音频"""
        # 预处理输入音频
        waveform = self.preprocess_audio(input_audio_path)
        if waveform is None:
            return
        
        original_length = waveform.size(1) / SAMPLE_RATE
        
        # 转换为梅尔频谱
        mel = self.waveform_to_mel(waveform)
        print(f"梅尔频谱形状: {mel.shape}")
        
        # 可视化原始梅尔频谱
        plt.figure(figsize=(12, 4))
        plt.imshow(mel.T.cpu().numpy(), aspect='auto', origin='lower')
        plt.title("Original Mel Spectrogram")
        plt.colorbar()
        plt.savefig("original_mel.png")
        plt.close()
        
        # 计算需要生成的时间步数
        time_steps_per_second = SAMPLE_RATE / HOP_LENGTH
        total_time_steps_needed = int(output_length_seconds * time_steps_per_second)
        time_steps_to_generate = total_time_steps_needed - mel.size(0)
        
        if time_steps_to_generate <= 0:
            print(f"输入音频已经长达 {original_length:.2f} 秒，无需续写。")
            return
        
        print(f"输入音频长度: {original_length:.2f} 秒")
        print(f"目标长度: {output_length_seconds:.2f} 秒")
        print(f"需要生成 {time_steps_to_generate} 个时间步...")
        
        # 使用输入音频的最后一部分作为生成起点
        start_idx = max(0, mel.size(0) - MAX_LENGTH)
        current_mel = mel[start_idx:].unsqueeze(0).to(self.device)
        
        # 自回归生成
        generated_mel = current_mel.clone()
        
        with torch.no_grad():
            for i in range(time_steps_to_generate):
                # 获取当前序列的最后MAX_LENGTH个时间步
                if generated_mel.size(1) > MAX_LENGTH:
                    model_input = generated_mel[:, -MAX_LENGTH:, :]
                else:
                    model_input = generated_mel
                
                # 预测下一个时间步
                next_step = self.model(model_input)
                next_step = next_step[:, -1:, :]  # 只取最后一个时间步的预测
                
                # 添加到生成序列
                generated_mel = torch.cat([generated_mel, next_step], dim=1)
                
                if (i + 1) % 100 == 0:
                    print(f"已生成 {i + 1}/{time_steps_to_generate} 时间步...")
        
        # 提取生成的部分（去掉输入部分）
        generated_only = generated_mel[:, current_mel.size(1):, :]
        
        # 转换回CPU并去除批量维度
        generated_only = generated_only.squeeze(0).cpu()
        
        # 将生成的部分与原始输入结合
        full_mel = torch.cat([mel, generated_only], dim=0)
        
        # 可视化生成的梅尔频谱
        plt.figure(figsize=(12, 4))
        plt.imshow(full_mel.T.cpu().numpy(), aspect='auto', origin='lower')
        plt.title("Full Mel Spectrogram (Original + Generated)")
        plt.colorbar()
        plt.savefig("full_mel.png")
        plt.close()
        
        # 转换回波形
        full_waveform = self.mel_to_waveform(full_mel)
        
        # 保存结果
        torchaudio.save(output_path, full_waveform, SAMPLE_RATE)
        print(f"续写完成！结果已保存到 {output_path}")
        
        # 可视化波形
        plt.figure(figsize=(12, 4))
        plt.plot(full_waveform.squeeze().numpy())
        plt.title("Generated Waveform")
        plt.savefig("waveform.png")
        plt.close()
        
        return full_waveform

# 模型定义（需要与训练时相同）
class AudioTransformer(torch.nn.Module):
    def __init__(self, input_dim, model_dim=256, num_heads=8, num_layers=6, dropout=0.1):
        super(AudioTransformer, self).__init__()
        self.input_dim = input_dim
        self.model_dim = model_dim
        
        self.input_projection = torch.nn.Linear(input_dim, model_dim)
        
        self.pos_encoder = PositionalEncoding(model_dim, dropout)
        
        encoder_layers = torch.nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.output_layer = torch.nn.Linear(model_dim, input_dim)
        
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src = self.input_projection(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        output = self.output_layer(output)
        return output

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        
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

def main():
    parser = argparse.ArgumentParser(description="音频续写推理")
    parser.add_argument("input_audio", type=str, help="输入音频文件路径")
    parser.add_argument("output_length", type=float, help="输出音频总长度（秒）")
    parser.add_argument("--model_path", type=str, default="audio_transformer_final.pth", help="训练好的模型路径")
    parser.add_argument("--output_path", type=str, default="continued_audio.wav", help="输出音频文件路径")
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not Path(args.input_audio).exists():
        print(f"错误: 输入文件 {args.input_audio} 不存在!")
        return
    
    # 检查模型文件是否存在
    if not Path(args.model_path).exists():
        print(f"错误: 模型文件 {args.model_path} 不存在!")
        print("请先运行训练代码或提供正确的模型路径")
        return
    
    # 初始化推理器
    inference = AudioInference(args.model_path)
    
    # 进行音频续写
    inference.continue_audio(args.input_audio, args.output_length, args.output_path)

if __name__ == "__main__":
    main()