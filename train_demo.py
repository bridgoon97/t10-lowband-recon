"""用真实语音(LJSpeech 29条)做 L0 退化模拟训练，画效果对比图。

- 数据: LJSpeech → 重采样到4kHz → 退化模拟(300-1200Hz带限) 作为输入
        原始重采样到4kHz 作为目标
- 训练: Arm B (CRN) 和 Arm A (DDSP) 各2000步
- 可视化: 输入谱 / 目标谱 / 预测谱 对比
"""
import os, sys, time, glob
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '.')

from lowband import build_model
from lowband.data.degradation import DegradationConfig, apply_degradation
from lowband.data.lowpass_sim import _resample
from lowband.dsp.stft import StftConfig, causal_stft
from lowband.losses.spectral import SpectralLoss
import soundfile as sf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SR = 4000
SEG = 4000  # 1秒
STFT_CFG = StftConfig(n_fft=128, hop=32, win=128)
torch.manual_seed(42); np.random.seed(42)

# ---- 加载真实语音，重采样到4kHz ----
wavs = sorted(glob.glob('data/real_speech/*.wav'))
print(f'加载 {len(wavs)} 条 LJSpeech')
clean_4k = []
for w in wavs:
    sig, sr_in = sf.read(w)
    if sig.ndim>1: sig = sig.mean(axis=1)
    sig = _resample(sig.astype(np.float32), sr_in, SR)
    # 切成1秒段
    for s in range(0, len(sig)-SEG, SEG//2):
        clean_4k.append(sig[s:s+SEG])
clean_4k = [c/(np.abs(c).max()+1e-8) for c in clean_4k if np.abs(c).max()>0.01]
print(f'  切出 {len(clean_4k)} 个1秒段')
# 分训练/验证
n_val = 6
val_clean = clean_4k[:n_val]
train_clean = clean_4k[n_val:]
print(f'  训练{len(train_clean)}段, 验证{len(val_clean)}段')

DEG = DegradationConfig(cutoff_min=300, cutoff_max=800, rolloff_min=12, rolloff_max=30,
                         noise_floor_min_db=-65, noise_floor_max_db=-45,
                         time_vary=True, spectral_tilt=True, formants=True,
                         body_noise=True, clipping=True, sample_rate=SR)

def make_batch(clean_list, batch_size, train=True):
    """从干净语音生成 (退化输入, 目标) 批。"""
    idx = np.random.choice(len(clean_list), batch_size)
    sensors, refs = [], []
    for i in idx:
        c = clean_list[i]
        # 目标 = 干净宽带(4kHz全频带)
        ref = torch.from_numpy(c.copy()).float()
        # 输入 = 退化(带限到300-800Hz)
        x = torch.from_numpy(c.copy()).float()
        rng = np.random.default_rng() if train else np.random.default_rng(i+1000)
        x_deg = apply_degradation(x, DEG, rng=rng, n_fft=128)
        sensors.append(x_deg); refs.append(ref)
    return torch.stack(sensors), torch.stack(refs)

def train_arm(arm_name, n_steps=2000, lr=3e-4):
    print(f'\n{"="*50}\n训练 {arm_name} ({n_steps}步)\n{"="*50}')
    cfg = {'arm':arm_name,'sample_rate':SR,'stft_n_fft':128,'stft_hop':32,'stft_win':128,'f0_mode':'estimated'}
    model = build_model(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = SpectralLoss()
    losses = []
    B = 8
    t0 = time.time()
    for step in range(n_steps):
        x, ref = make_batch(train_clean, B)
        _, ref_mag = causal_stft(ref, STFT_CFG)
        opt.zero_grad()
        out = model(x, None)
        N = min(out['mag'].shape[-1], ref_mag.shape[-1])
        loss = loss_fn(out['mag'][...,:N], ref_mag[...,:N])['loss']
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 200 == 0:
            el = time.time()-t0; sps = (step+1)/el
            print(f'  step {step:4d} loss={loss.item():.4f} steps/s={sps:.1f}')
            losses.append(loss.item())
    return model, losses

# 训练两个臂
models = {}
for arm in ['arm_b_crn', 'arm_a_ddsp']:
    m, ls = train_arm(arm, n_steps=1500)
    models[arm] = m
    torch.save(m.state_dict(), f'checkpoints/{arm}_real.pt')

# ---- 可视化 ----
print('\n生成频谱对比图...')
x_val, ref_val = make_batch(val_clean, 4, train=False)
_, ref_mag = causal_stft(ref_val, STFT_CFG)

fig, axes = plt.subplots(4, 4, figsize=(16, 12))
for i in range(4):
    # 输入谱
    _, in_mag = causal_stft(x_val[i:i+1], STFT_CFG)
    axes[0,i].imshow(in_mag[0].log1p().numpy(), aspect='auto', origin='lower',
                      extent=[0,1,0,2000], cmap='magma')
    axes[0,i].set_title(f'输入(带限) #{i}'); axes[0,i].set_ylabel('Hz')
    if i==0: axes[0,0].text(0.02,0.9,'INPUT',color='w',transform=axes[0,0].transAxes)
    # 目标谱
    axes[1,i].imshow(ref_mag[i].log1p().numpy(), aspect='auto', origin='lower',
                      extent=[0,1,0,2000], cmap='viridis')
    axes[1,i].set_title(f'目标(宽带) #{i}')
    if i==0: axes[1,0].text(0.02,0.9,'TARGET',color='w',transform=axes[1,0].transAxes)
    # Arm B 预测
    for row, (arm, cmap) in enumerate([('arm_b_crn','cividis'),('arm_a_ddsp','plasma')], start=2):
        with torch.no_grad():
            out = models[arm](x_val[i:i+1], None)
        N = min(out['mag'].shape[-1], ref_mag.shape[-1])
        pred = out['mag'][0,...,:N].log1p().numpy()
        axes[row,i].imshow(pred, aspect='auto', origin='lower',
                            extent=[0,1,0,2000], cmap=cmap)
        axes[row,i].set_title(f'{arm} #{i}')
        if i==0:
            axes[row,0].text(0.02,0.9,arm.upper(),color='w',transform=axes[row,0].transAxes)

plt.suptitle('低频重建效果对比 (LJSpeech真实语音, L0退化模拟, 4kHz)\n'
             '上→下: 输入(300-800Hz带限) / 目标(0-2kHz) / CRN预测 / DDSP预测', fontsize=13)
plt.tight_layout()
plt.savefig('reports/recon_demo.png', dpi=120)
print('保存 reports/recon_demo.png')

# 也画一个频谱切片对比(某一帧的频谱曲线)
fig2, ax = plt.subplots(1,1, figsize=(10,5))
frame_idx = 60
in_slice = in_mag[0,:,min(frame_idx,in_mag.shape[-1]-1)].numpy()
ref_slice = ref_mag[0,:,min(frame_idx,ref_mag.shape[-1]-1)].numpy()
freqs = np.linspace(0, 2000, len(in_slice))
ax.semilogy(freqs, in_slice, 'r-', label='输入(带限)', alpha=0.7)
freqs_ref = np.linspace(0, 2000, len(ref_slice))
ax.semilogy(freqs_ref, ref_slice, 'k-', label='目标', linewidth=2)
for arm, c in [('arm_b_crn','b'),('arm_a_ddsp','g')]:
    with torch.no_grad(): out = models[arm](x_val[0:1], None)
    N = min(out['mag'].shape[-1], ref_mag.shape[-1])
    p = out['mag'][0,:,min(frame_idx,N-1)].numpy()
    ax.semilogy(np.linspace(0,2000,len(p)), p, c+'--', label=f'{arm}预测', alpha=0.8)
ax.set_xlabel('频率 (Hz)'); ax.set_ylabel('幅度'); ax.set_title(f'单帧频谱对比 (第{frame_idx}帧)')
ax.legend(); ax.grid(True, alpha=0.3); ax.set_xlim(0,2000)
plt.tight_layout(); plt.savefig('reports/recon_spectrum.png', dpi=120)
print('保存 reports/recon_spectrum.png')
