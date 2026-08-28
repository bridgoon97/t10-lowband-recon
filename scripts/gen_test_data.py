"""生成 data/test_speech/ 下的合成测试语音（让 verify.py 开箱可跑）。

这是 L0 退化模拟用的"干净宽带语音"源——简单的谐波合成，结构上
足够让退化管线和训练冒烟测试跑通。真实语音请用 gen_data.py 或
DatasetAdapter 接入。

    python scripts/gen_test_data.py
"""
import os
import numpy as np
import soundfile as sf

os.makedirs("data/test_speech", exist_ok=True)
np.random.seed(42)
SR = 16000

for i in range(10):
    T = SR * 2  # 2 秒
    t = np.arange(T) / SR
    f0 = 100 + 50 * np.sin(2 * np.pi * 2 * t)        # F0 轨迹
    phase = np.cumsum(2 * np.pi * f0 / SR)
    sig = np.zeros(T)
    for k in range(1, 20):                            # 谐波
        sig += (1.0 / k) * np.sin(k * phase)
    env = 0.3 + 0.7 * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t))  # AM 包络
    sig = sig * env * 0.3
    sig += np.random.randn(T) * 0.01                  # 轻微噪声
    sf.write(f"data/test_speech/clean_{i:02d}.wav", sig.astype(np.float32), SR)

print(f"生成 {len(os.listdir('data/test_speech'))} 个文件到 data/test_speech/")
