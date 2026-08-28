"""生成更接近真实语音频谱的合成数据：
- F0 轨迹（音高轮廓）
- 谐波结构（基频+高次谐波）
- 共振峰（F1/F2/F3 模拟元音）
- 清浊音交替（voiced/unvoiced 段）
- 音节级能量包络
"""
import numpy as np
import soundfile as sf
import os

SR = 16000
np.random.seed(42)

def hz_to_mel(f): return 1127 * np.log1p(f / 700.0)
def mel_to_hz(m): return 700 * (np.exp(m / 1127.0) - 1)

VOWEL_FORMANTS = [
    # (F1, F2, F3) 模拟 /a/ /i/ /u/ /e/ /o/
    (730, 1090, 2440),  # a
    (270, 2290, 3010),  # i
    (300, 870, 2240),   # u
    (530, 1840, 2480),  # e
    (570, 840, 2410),   # o
]

def formant_filter(sig, f0, sr, formants, Q=12):
    """对谐波信号施加共振峰滤波（模拟声道传递函数）。"""
    from scipy.signal import lfilter
    b = np.array([1.0])
    a = np.array([1.0])
    for F in formants:
        bw = F / Q
        r = np.exp(-np.pi * bw / sr)
        theta = 2 * np.pi * F / sr
        # 二阶共振
        a = np.convolve(a, [1, -2*r*np.cos(theta), r**2])
    return lfilter(b, a, sig)

def gen_syllable(duration, sr, f0_base, vowel_idx, voiced=True):
    t = np.arange(int(sr * duration)) / sr
    n = len(t)
    if voiced:
        # F0 微小抖动（自然语音有 jitter）
        f0 = f0_base * (1 + 0.03 * np.sin(2*np.pi*5*t) + 0.02*np.random.randn(n))
        phase = 2*np.pi*np.cumsum(f0) / sr
        sig = np.zeros(n)
        # 谐波 1..30，幅度递减
        for k in range(1, 31):
            sig += (0.5/k**0.8) * np.sin(k * phase)
        sig = sig / 20.0
        # 共振峰滤波
        sig = formant_filter(sig, f0, sr, VOWEL_FORMANTS[vowel_idx])
    else:
        # 清音：白噪过高通滤波
        noise = np.random.randn(n)
        from scipy.signal import butter, filtfilt
        b, a = butter(4, 1500/(sr/2), btype='high')
        sig = filtfilt(b, a, noise) * 0.3
    # 音节级能量包络（起音+衰减）
    env = np.exp(-t / (duration * 0.6)) * (1 - np.exp(-t / 0.01))
    return sig * env

def gen_utterance(sr=16000, duration=3.0):
    """生成一段模拟语音：多个音节，F0 轮廓，清浊交替。"""
    n_total = int(sr * duration)
    sig = np.zeros(n_total)
    pos = 0
    syllable_dur = 0.2 + 0.15 * np.random.rand()
    f0_base = 90 + 60 * np.random.rand()  # 90-150 Hz
    # F0 整体轮廓（下降，模拟自然语调）
    f0_drift = np.linspace(1.2, 0.85, int(duration/syllable_dur)+2)
    i = 0
    while pos < n_total:
        d = min(syllable_dur, (n_total - pos) / sr)
        voiced = np.random.rand() > 0.25  # 75% 浊音
        vowel = np.random.randint(0, 5)
        f0 = f0_base * f0_drift[min(i, len(f0_drift)-1)]
        seg = gen_syllable(d, sr, f0, vowel, voiced)
        sig[pos:pos+len(seg)] += seg
        pos += len(seg)
        syllable_dur = 0.15 + 0.2 * np.random.rand()
        i += 1
    # 归一化
    sig = sig / (np.max(np.abs(sig)) + 1e-8) * 0.8
    return sig

os.makedirs("data/synth_speech", exist_ok=True)
for i in range(80):
    sig = gen_utterance(SR, duration=2.0 + np.random.rand())
    sf.write(f"data/synth_speech/utt_{i:03d}.wav", sig.astype(np.float32), SR)
print(f"Generated {len(os.listdir('data/synth_speech'))} utterances")
