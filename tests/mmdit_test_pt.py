#!/usr/bin/env python3
"""
测试脚本: 独立测试 MMDiT 模型 (PyTorch 版本)
用于对比 MindSpore 版本，调试 SD3.5M 的 MMDiT 模型
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# 注意: 这个脚本需要在标准的 ComfyUI (PyTorch 版本) 环境中运行
# 请确保你有 PyTorch 版本的 ComfyUI

# 添加 ComfyUI 路径 (根据你的实际路径修改)
COMFYUI_PATH = "/home/zyd/ComfyUI"  # 修改为你的 PyTorch ComfyUI 路径
sys.path.insert(0, COMFYUI_PATH)

try:
    from comfy.ldm.modules.diffusionmodules.mmdit import MMDiT, OpenAISignatureMMDITWrapper
    import comfy.ops
except ImportError as e:
    print(f"错误: 无法导入 ComfyUI 模块")
    print(f"请确保 COMFYUI_PATH 设置正确: {COMFYUI_PATH}")
    print(f"错误详情: {e}")
    sys.exit(1)


def create_test_mmdit(
    input_size=(96, 160),  # SD3.5M 的 latent size (H/8, W/8) for 768x1280
    patch_size=2,
    in_channels=16,
    depth=24,
    mlp_ratio=4.0,
    learn_sigma=False,
    device='cuda' if torch.cuda.is_available() else 'cpu',
):
    """
    创建一个测试用的 MMDiT 模型 (PyTorch 版本)
    
    参数与 SD3.5M 配置对应
    """
    
    # SD3.5M 的配置
    adm_in_channels = 2048  # pooled text embedding dimension
    context_embedder_config = {
        "target": "comfy.ldm.modules.encoders.modules.ContextEmbedder",
        "params": {
            "in_channels": 4096,  # CLIP-L (768) + CLIP-G (1280) + T5 (2048) = 4096
            "out_channels": 1536,  # hidden dimension
        }
    }
    
    print("=" * 80)
    print("创建 MMDiT 模型 (PyTorch)")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Input size: {input_size}")
    print(f"Patch size: {patch_size}")
    print(f"In channels: {in_channels}")
    print(f"Depth: {depth}")
    print(f"ADM in channels: {adm_in_channels}")
    print()
    
    model = OpenAISignatureMMDITWrapper(
        input_size=input_size,
        patch_size=patch_size,
        in_channels=in_channels,
        depth=depth,
        mlp_ratio=mlp_ratio,
        learn_sigma=learn_sigma,
        adm_in_channels=adm_in_channels,
        context_embedder_config=context_embedder_config,
        operations=comfy.ops.disable_weight_init,
        device=device,
        dtype=torch.float32,
    )
    
    model.eval()
    model = model.to(device)
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"✓ 模型创建成功")
    print(f"  参数量: {param_count / 1e6:.2f}M")
    print()
    
    return model


def create_test_inputs(
    batch_size=1,
    latent_channels=16,
    latent_h=96,
    latent_w=160,
    context_tokens=77,
    context_dim=4096,
    pooled_dim=2048,
    seed=42,
    device='cuda' if torch.cuda.is_available() else 'cpu',
):
    """
    创建测试输入 tensors (PyTorch 版本)
    
    返回:
        x: noisy latent [B, C, H, W]
        t: timestep [B]
        context: text embeddings [B, N, D]
        y: pooled text embeddings [B, D_pooled]
    """
    
    print("=" * 80)
    print("创建测试输入 (PyTorch)")
    print("=" * 80)
    
    # 设置随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed(seed)
    
    # 1. Noisy latent (从标准正态分布采样)
    x = torch.randn(batch_size, latent_channels, latent_h, latent_w, 
                    dtype=torch.float32, device=device)
    print(f"✓ Noisy latent x: {x.shape}")
    print(f"  - dtype: {x.dtype}")
    print(f"  - device: {x.device}")
    print(f"  - min: {x.min().item():.4f}, max: {x.max().item():.4f}")
    print(f"  - mean: {x.mean().item():.4f}, std: {x.std().item():.4f}")
    print()
    
    # 2. Timestep (SD3 使用 flow matching, timestep 在 [0, 1] 范围)
    t = torch.tensor([1000.0] * batch_size, dtype=torch.float32, device=device)
    print(f"✓ Timestep t: {t.shape}")
    print(f"  - values: {t.cpu().numpy()}")
    print()
    
    # 3. Context (CLIP + T5 text embeddings)
    context = torch.randn(batch_size, context_tokens, context_dim, 
                         dtype=torch.float32, device=device)
    print(f"✓ Context: {context.shape}")
    print(f"  - dtype: {context.dtype}")
    print(f"  - device: {context.device}")
    print(f"  - min: {context.min().item():.4f}, max: {context.max().item():.4f}")
    print(f"  - mean: {context.mean().item():.4f}, std: {context.std().item():.4f}")
    print()
    
    # 4. Pooled output (来自 CLIP 的 pooled embedding)
    y = torch.randn(batch_size, pooled_dim, dtype=torch.float32, device=device)
    print(f"✓ Pooled output y: {y.shape}")
    print(f"  - dtype: {y.dtype}")
    print(f"  - device: {y.device}")
    print(f"  - min: {y.min().item():.4f}, max: {y.max().item():.4f}")
    print(f"  - mean: {y.mean().item():.4f}, std: {y.std().item():.4f}")
    print()
    
    return x, t, context, y


def test_mmdit_forward(model, x, t, context, y):
    """
    测试 MMDiT 的前向传播 (PyTorch 版本)
    """
    
    print("=" * 80)
    print("执行 MMDiT 前向传播 (PyTorch)")
    print("=" * 80)
    
    try:
        # 前向传播
        print("调用 model.forward()...")
        with torch.no_grad():  # 不需要梯度
            output = model(
                x=x,
                timesteps=t,
                context=context,
                y=y,
                control=None,
                transformer_options={},
            )
        
        print(f"✓ 前向传播成功!")
        print()
        print(f"输出 shape: {output.shape}")
        print(f"输出 dtype: {output.dtype}")
        print(f"输出 device: {output.device}")
        print(f"输出统计:")
        print(f"  - min: {output.min().item():.4f}")
        print(f"  - max: {output.max().item():.4f}")
        print(f"  - mean: {output.mean().item():.4f}")
        print(f"  - std: {output.std().item():.4f}")
        print()
        
        # 检查输出是否包含 NaN 或 Inf
        has_nan = torch.isnan(output).any().item()
        has_inf = torch.isinf(output).any().item()
        
        if has_nan:
            print("⚠️  警告: 输出包含 NaN!")
        if has_inf:
            print("⚠️  警告: 输出包含 Inf!")
        
        if not has_nan and not has_inf:
            print("✓ 输出数值正常 (无 NaN 或 Inf)")
        
        print()
        return output
        
    except Exception as e:
        print(f"✗ 前向传播失败!")
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return None


def compare_with_expected(output, x):
    """
    与预期输出进行简单对比
    """
    
    print("=" * 80)
    print("输出验证")
    print("=" * 80)
    
    # 检查输出 shape 是否正确
    expected_shape = x.shape  # 输出应该与输入 latent 相同 shape
    if output.shape == expected_shape:
        print(f"✓ 输出 shape 正确: {output.shape}")
    else:
        print(f"✗ 输出 shape 错误!")
        print(f"  期望: {expected_shape}")
        print(f"  实际: {output.shape}")
    
    # 检查输出数值范围是否合理
    print()
    print("数值范围对比:")
    print(f"  输入 x - mean: {x.mean().item():.4f}, std: {x.std().item():.4f}")
    print(f"  输出   - mean: {output.mean().item():.4f}, std: {output.std().item():.4f}")
    
    # 简单的合理性检查
    if abs(output.mean().item()) < 10.0 and output.std().item() < 10.0:
        print("✓ 输出数值范围看起来合理")
    else:
        print("⚠️  输出数值范围可能异常")
    
    print()


def save_results_for_comparison(x, t, context, y, output, filename="pytorch_results.npz"):
    """
    保存结果用于与 MindSpore 版本对比
    """
    
    print("=" * 80)
    print("保存结果")
    print("=" * 80)
    
    # 转换为 numpy 并保存
    results = {
        'x': x.cpu().numpy(),
        't': t.cpu().numpy(),
        'context': context.cpu().numpy(),
        'y': y.cpu().numpy(),
        'output': output.cpu().numpy(),
    }
    
    np.savez(filename, **results)
    print(f"✓ 结果已保存到: {filename}")
    print()
    print("你可以在 MindSpore 版本中加载这些结果进行对比:")
    print(f"  data = np.load('{filename}')")
    print(f"  x_pytorch = data['x']")
    print(f"  output_pytorch = data['output']")
    print()


def main():
    """
    主函数
    """
    
    print("\n" + "=" * 80)
    print("MMDiT 独立测试脚本 (PyTorch 版本)")
    print("=" * 80)
    print()
    
    # 检查设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # 1. 创建模型
    model = create_test_mmdit(
        input_size=(96, 160),  # 对应 768x1280 图像
        patch_size=2,
        in_channels=16,
        depth=24,  # SD3.5M 使用 24 层
        mlp_ratio=4.0,
        learn_sigma=False,
        device=device,
    )
    
    # 2. 创建测试输入
    x, t, context, y = create_test_inputs(
        batch_size=1,
        latent_channels=16,
        latent_h=96,
        latent_w=160,
        context_tokens=77,
        context_dim=4096,
        pooled_dim=2048,
        seed=42,
        device=device,
    )
    
    # 3. 执行前向传播
    output = test_mmdit_forward(model, x, t, context, y)
    
    # 4. 验证输出
    if output is not None:
        compare_with_expected(output, x)
        
        # 5. 保存结果用于对比
        save_results_for_comparison(x, t, context, y, output, "pytorch_mmdit_results.npz")
    
    print("=" * 80)
    print("测试完成")
    print("=" * 80)
    print()
    
    return model, x, t, context, y, output


if __name__ == "__main__":
    # 提示用户修改路径
    if COMFYUI_PATH == "/path/to/ComfyUI":
        print("=" * 80)
        print("⚠️  警告: 请先修改脚本中的 COMFYUI_PATH")
        print("=" * 80)
        print()
        print("请编辑脚本，将 COMFYUI_PATH 设置为你的 PyTorch ComfyUI 路径")
        print("例如: COMFYUI_PATH = '/home/user/ComfyUI'")
        print()
        sys.exit(1)
    
    model, x, t, context, y, output = main()
