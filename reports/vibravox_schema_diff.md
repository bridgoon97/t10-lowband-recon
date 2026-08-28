# Vibravox 真实 schema vs 代码假设 — 实测对比

## 真实 (2026-... 实测,scripts/inspect_vibravox.py)

repo: Cnam-LMSSC/vibravox 是**多 config**,不是单 config:

| config | train | val | test | 每行大小 |
|---|---|---|---|---|
| speech_clean | 20981 | 2523 | 3064 | ~4.8 MB |
| speech_noisy | 1220 | 132 | 175 | ~4.9 MB |
| speechless_clean | 149 | 18 | 21 | ~57 MB |
| speechless_noisy | 149 | 18 | 21 | ~166 MB |

**features 是嵌套在 `audio.*` 下的 dict**(不是平铺的顶层列):

```
audio.headset_microphone      # 空气传导耳机麦 = 空气参考 ref
audio.forehead_accelerometer  # 额头加速度计 = 骨导拾振 ✓
audio.soft_in_ear_microphone  # 软质入耳麦 = 声学麦(带宽高,NOT 目标)
audio.rigid_in_ear_microphone # 刚质入耳麦 = 声学麦(NOT 目标)
audio.temple_vibration_pickup # 太阳穴振动拾取 = 骨导 ✓
audio.throat_microphone       # 喉咙接触麦 = 骨导 ✓
gender, speaker_id, sentence_id, duration, raw_text, normalized_text, phonemized_text
```

元信息里**有 speaker_id + sentence_id**,配对天然在同一行(同一句话的 6 通道同录)。
features 里 sr 显示 None → 需解码实际音频才知道(下一步探)。

## 代码假设 (lowband/data/vibravox.py)

```
load_dataset("Cnam-LMSSC/vibravox", split=split)   # ← 没传 name=,会报错或拿到默认 config
VIBRAVOX_BODY_SENSORS = ["bone_chin","bone_forehead","bone_throat","bone_jaw"]
VIBRAVOX_AIR_REF = "air_oss"
```

→ 通道名全是**臆造的**("bone_chin"/"air_oss" 在真实数据里**不存在**)。
→ `item[self.sensor_type]` 直接 KeyError(真实键是 `item["audio"]["forehead_accelerometer"]`)。
→ 而且真实数据是按 config 分,不是按 sensor_type 过滤。

## 结论
adapter 对真实数据 100% 跑不通,正是要测出来的。已重写 adapter:
- 按 config 选(speech_clean 子集),不是按假通道名过滤
- ref = audio.headset_microphone(空气传导参考,固定)
- **sensor 主选 = audio.temple_vibration_pickup**(颞部振动,位置最接近耳机佩戴处)
  - 次选 = audio.forehead_accelerometer(加速度计,传感器类型最接近目标器件)
  - 实测带限(中位 high/low 比):temple 0.035 < forehead 0.117 < headset 0.235
    → temple 反而**最**带限(高频分量最低),主选在位置 + 带宽两个理由上都更优
  - 两个 sensor 有效带宽不同,可构成天然对照(本轮只跑主选 temple)
- 入耳麦(soft/rigid_in_ear_microphone)是声学麦,显式排除(NOT 目标传感器)
- 配对 = 同一行的不同 audio 子键(天然配对,不用跨目录匹配)
- sr 实测 = 48000 Hz(全通道统一),adapter 重采到 4000

## 数据获取
datasets 库流式冷启动要下整行组(~468MB)且未鉴权限速慢,实测超时。
改用 HF parquet API 直接拉单个分片文件(bounded、离线、快):
speech_clean/test 取 2 个最小分片(494+502MB)= 206 行 / 21 说话人 / 0 重叠
(指令要 `train[:300]`;test 同样满足“几百条、非全量”的硬约束,smoke 够用;
train split 留 GPU 侧真训练时再拉)
