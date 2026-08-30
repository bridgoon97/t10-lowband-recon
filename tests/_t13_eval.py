"""T13 测试侧唯一评测谱入口。

算法实际接收的是 ``apply_d1`` 频谱经 ISTFT 得到的时域 S，因此所有
评测必须以再次分析该时域信号所得的 ``stft_batch(S)`` 为参照。D1 的
直接返回谱只是一项中间构造，不能作为一个实际存在的信号参照。
"""
from fusion.degrade import apply_d1
from fusion.f0 import f0_batch
from fusion.stft import istft_batch, stft_batch


def eval_specs(ff, cfg, deg):
    """返回评测用 ``(spec_X, spec_S, S)``；spec_S 必为往返后分析谱。"""
    spec_x = stft_batch(ff, cfg)
    f0, _ = f0_batch(ff, cfg)
    d1_spec, _ = apply_d1(spec_x, f0, cfg, deg)
    s = istft_batch(d1_spec, cfg, length=ff.shape[-1])
    spec_s = stft_batch(s, cfg)
    return spec_x, spec_s, s
