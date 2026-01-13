# Qwen-Image

This repository provides the inference server support for [Qwen-Image](https://arxiv.org/abs/2508.02324).

-----

## ✨ Key Features

* **Superior Text Rendering:** Qwen-Image excels at complex text rendering, including multiline layouts, paragraph-level semantics, and fine-grained details. It supports both alphabetic languages (e.g., English) and logographic languages (e.g., Chinese) with high fidelity.

* **Consistent Image Editing:** Through our enhanced multi-task training paradigm, Qwen-Image achieves exceptional performance in preserving both semantic meaning and visual realism during editing operations.

* **Strong Cross-Benchmark Performance:** Evaluated on multiple benchmarks, Qwen-Image consistently outperforms existing models across diverse generation and editing tasks, establishing a strong foundation model for image generation.


## 📑 Todo List
- Qwen-Image (Text-to-Image Model)
  - [x] Inference server support


## 📦 Requirements

| mindspore | ascend driver | cann               |
| :-------: | :-----------: | :----------------: |
| >=2.7.0   |  >=25.2.0     | >=8.2.RC1          |


## 🚀 Quick Start

### Installation
Clone the repo:
```sh
git clone https://github.com/mindspore-lab/mindone.git
cd mindone/examples/diffusers/qwenimage
```

Download Model Weights:
```bash
# Download from HuggingFace
hf download Qwen/Qwen-Image
```

### Run Qwen-Image Inference Server

Try the inference example:
```bash
cd comfyui_mindspore/research/qwenimage/
sh run_infer.sh
```

Start the inference server:
```bash
cd comfyui_mindspore/research/qwenimage/
sh run_api.sh
```

When the server is ready, showing "* Running on http://...", enjoy the comfyui_mindspore!
```bash
cd comfyui_mindspore/

# run on ascend 310p, forcing fp16 precision
python main.py --listen 0.0.0.0 --port 9001 --force-fp16 --fp16-vae
```

#### Note: 
1. Start a qwenimage server api requires the distributed environment (`4 * ascend 310p`).
2. Wait for all NPUs ready. If you use `4 * ascend 310p`, four urls will be opened for calling.
3. Run comfyui_mindspore, add `AsyncQwenImageGenerator` to calling the qwenimage server api.

#### Some configurations you may be interested in `qwenimage_generator.py`: 

| configurations          | Description                                                  | Default     |
| ----------------------- | ------------------------------------------------------------ | ----------- |
| `prompt`                | Prompt to guide the image generation                         | (Required)  |
| `width` and `height`    | Image resolution.                                            | `1024`      |
| `num_inference_steps`   | Diffusion infer steps                                        | `50`        |
| `true_cfg_scale`        | Scale for true classifier-free guidance with negative_prompt | `4.0`       |
| `num_workers`           | Number of used NPUs for distributed environment              | `4`         |
| `negative_prompt`       | The prompt not to guide the image generation                 | None        |
| `base_seed`             | Random seed for image generation                             | `42`        |
| `base_port`             | Initial Port for starting server api for calling             | `5000`      |
| `timeout`               | Wait seconds for generating an image (> infer_steps * cost)  | `15000`     |
