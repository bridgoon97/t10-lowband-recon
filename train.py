#!/usr/bin/env python3
"""Unified training entry point (§7.2: one command, three arms share this).

    python train.py --config configs/arm_a_ddsp.yaml

Config must expose:
    arm, sample_rate, stft_* , device, seed, deterministic
    data: {adapter, ...}
    loss: {spectral, multi_res_stft, discriminator, ...}
    optimizer: {lr, weight_decay, betas}
    scheduler: {type, ...}
    batch_size, num_steps, log_every, ckpt_every, ckpt_dir
"""
from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from lowband import build_model, build_dataset, ARMS
from lowband.dsp import StftConfig, stft
from lowband.losses import (SpectralLoss, MultiResolutionSTFTLoss,
                            reconstruct_waveform_with_oracle_phase,
                            MultiSubbandDiscriminator)
from lowband.utils.config import load_config, set_seed, get_device
from lowband.utils.complexity import measure_complexity


def collate(batch):
    sensors = torch.stack([b["sensor"] for b in batch])
    refs = torch.stack([b["ref"] for b in batch])
    return {"sensor": sensors, "ref": refs,
            "meta": [b["meta"] for b in batch]}


def build_loss(cfg: dict, stft_cfg: StftConfig):
    loss_cfg = cfg.get("loss", {})
    spectral = SpectralLoss(
        l1_weight=loss_cfg.get("l1_weight", 1.0),
        l2_weight=loss_cfg.get("l2_weight", 0.5),
        db_weight=loss_cfg.get("db_weight", 1.0),
    )
    use_mrstft = loss_cfg.get("multi_res_stft", False)
    mrstft = MultiResolutionSTFTLoss() if use_mrstft else None
    use_disc = loss_cfg.get("discriminator", False)
    disc = MultiSubbandDiscriminator() if use_disc else None
    return spectral, mrstft, disc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42), cfg.get("deterministic", False))
    device = get_device(cfg)

    # --- model ---
    model_cfg = dict(cfg)
    model_cfg["sample_rate"] = cfg["sample_rate"]
    model = build_model(cfg).to(device)
    print(f"[train] arm={cfg['arm']} params={sum(p.numel() for p in model.parameters()):,}")

    # --- data ---
    data_cfg = cfg["data"]
    data_cfg["sr"] = cfg["sample_rate"]
    train_ds = build_dataset(data_cfg)
    loader = DataLoader(train_ds, batch_size=cfg.get("batch_size", 4),
                        shuffle=True, num_workers=cfg.get("num_workers", 0),
                        collate_fn=collate, drop_last=True)
    print(f"[train] dataset: {len(train_ds)} items")

    # --- loss ---
    stft_cfg = StftConfig(
        n_fft=cfg.get("stft_n_fft", 128),
        hop=cfg.get("stft_hop", 32),
        win=cfg.get("stft_win", 128),
        window=cfg.get("stft_window", "hann"),
    )
    spectral, mrstft, disc = build_loss(cfg, stft_cfg)
    spectral = spectral.to(device)
    if mrstft:
        mrstft = mrstft.to(device)
    if disc:
        disc = disc.to(device)

    # --- optimizer ---
    opt_cfg = cfg.get("optimizer", {"lr": 3e-4, "weight_decay": 1e-5})
    params = list(model.parameters())
    if disc:
        params += list(disc.parameters())
    optimizer = torch.optim.AdamW(params, lr=opt_cfg.get("lr", 3e-4),
                                   weight_decay=opt_cfg.get("weight_decay", 1e-5),
                                   betas=tuple(opt_cfg.get("betas", [0.9, 0.999])))
    scheduler = None
    sch_cfg = cfg.get("scheduler", {})
    if sch_cfg.get("type") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.get("num_steps", 1000))

    # --- checkpoint ---
    ckpt_dir = Path(cfg.get("ckpt_dir", "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        step = ckpt["step"]
        print(f"[train] resumed from step {step}")

    num_steps = cfg.get("num_steps", 1000)
    log_every = cfg.get("log_every", 50)
    ckpt_every = cfg.get("ckpt_every", 500)
    use_amp = cfg.get("amp", False)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and torch.cuda.is_available())

    model.train()
    data_iter = iter(loader)
    t0 = time.time()
    while step < num_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        sensor = batch["sensor"].to(device)
        ref = batch["ref"].to(device)

        # F0 oracle (if arm needs it) — computed from REF for oracle mode.
        # §6.5: this is ONLY for the f0_mode="oracle" ablation path.
        cond = None
        if cfg.get("arm") == "arm_a_ddsp" and cfg.get("f0_mode") == "oracle":
            from lowband.dsp.f0 import yin_f0
            f0, _ = yin_f0(ref, cfg["sample_rate"], frame_len=stft_cfg.win)
            cond = {"f0": f0}

        optimizer.zero_grad()
        with torch.autocast(device_type=("cuda" if device.type == "cuda" else "cpu"),
                            enabled=use_amp):
            out = model(sensor, cond)
            pred_mag = out["mag"]  # (B, F, N)

            # Target magnitude
            _, target_mag = stft(ref, stft_cfg)
            # Align frame counts
            N = min(pred_mag.shape[-1], target_mag.shape[-1])
            pred_mag = pred_mag[..., :N]
            target_mag = target_mag[..., :N]

            loss_dict = spectral(pred_mag, target_mag)
            loss = loss_dict["loss"]

            if mrstft:
                pred_wav = reconstruct_waveform_with_oracle_phase(
                    pred_mag, ref, stft_cfg)
                mrstft_loss = mrstft(pred_wav, ref)
                loss = loss + cfg.get("loss", {}).get("mrstft_weight", 1.0) * mrstft_loss

            if disc:
                pred_wav = reconstruct_waveform_with_oracle_phase(
                    pred_mag, ref, stft_cfg)
                adv_loss, feat_loss = disc(pred_wav, ref)
                d_loss = disc.disc_loss(pred_wav, ref)
                loss = loss + cfg.get("loss", {}).get("adv_weight", 1.0) * adv_loss
                loss = loss + cfg.get("loss", {}).get("feat_weight", 10.0) * feat_loss

        scaler.scale(loss).backward()
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        if disc:
            # Discriminator step
            optimizer.zero_grad()
            with torch.autocast(device_type=("cuda" if device.type == "cuda" else "cpu"),
                                enabled=use_amp):
                pred_wav = reconstruct_waveform_with_oracle_phase(
                    pred_mag.detach(), ref, stft_cfg)
                d_loss = disc.disc_loss(pred_wav, ref)
            scaler.scale(d_loss).backward()
            scaler.step(optimizer)
            scaler.update()

        if scheduler:
            scheduler.step()
        step += 1

        if step % log_every == 0:
            elapsed = time.time() - t0
            sps = step / elapsed
            grad_norm = sum(p.grad.norm().item() ** 2 for p in model.parameters()
                            if p.grad is not None) ** 0.5
            lr = optimizer.param_groups[0]["lr"]
            print(f"[train] step {step}/{num_steps} loss={loss.item():.4f} "
                  f"l1={loss_dict['l1'].item():.4f} db={loss_dict['db_l1'].item():.4f} "
                  f"grad_norm={grad_norm:.3f} lr={lr:.2e} "
                  f"steps/s={sps:.2f} (CPU-only, NOT extrapolable to GPU)")
            print(f"         NOTE: steps/s measured on CPU is relative only; "
                  f"do NOT extrapolate to GPU throughput.")

        if step % ckpt_every == 0:
            ckpt_path = ckpt_dir / f"{cfg['arm']}_step{step}.pt"
            torch.save({"model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "step": step,
                        "cfg": cfg}, ckpt_path)
            print(f"[train] checkpoint → {ckpt_path}")

    # Final checkpoint
    ckpt_path = ckpt_dir / f"{cfg['arm']}_final.pt"
    torch.save({"model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "cfg": cfg}, ckpt_path)
    print(f"[train] done. final checkpoint → {ckpt_path}")

    # Report complexity (§5.1)
    print("\n[train] === Static complexity (§5.1) ===")
    c = measure_complexity(model, cfg["sample_rate"],
                            hop=stft_cfg.hop, n_bins=stft_cfg.num_bins)
    print(f"  params: {c['params']:,}")
    print(f"  MACs/s: {c['macs_per_sec']:,}  (budget ≤60 MMACs/s)")
    print(f"  peak memory: {c['peak_total_kb']:.1f} KB  (budget ≤300 KB)")
    print(f"  weight size: {c['weight_kb']:.1f} KB")


if __name__ == "__main__":
    main()
