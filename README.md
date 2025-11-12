<div align="center">

# Run ComfyUI on MindSpore
**The most powerful and modular visual AI engine and application.**


[![Website][website-shield]][website-url]
[![Dynamic JSON Badge][discord-shield]][discord-url]
[![Twitter][twitter-shield]][twitter-url]
[![Matrix][matrix-shield]][matrix-url]
<br>
[![][github-release-shield]][github-release-link]


[matrix-shield]: https://img.shields.io/badge/Matrix-000000?style=flat&logo=matrix&logoColor=white
[matrix-url]: https://app.element.io/#/room/%23comfyui_space%3Amatrix.org
[website-shield]: https://img.shields.io/badge/ComfyOrg-4285F4?style=flat
[website-url]: https://www.comfy.org/
<!-- Workaround to display total user from https://github.com/badges/shields/issues/4500#issuecomment-2060079995 -->
[discord-shield]: https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fdiscord.com%2Fapi%2Finvites%2Fcomfyorg%3Fwith_counts%3Dtrue&query=%24.approximate_member_count&logo=discord&logoColor=white&label=Discord&color=green&suffix=%20total
[discord-url]: https://www.comfy.org/discord
[twitter-shield]: https://img.shields.io/twitter/follow/ComfyUI
[twitter-url]: https://x.com/ComfyUI

[github-release-shield]: https://img.shields.io/github/v/release/comfyanonymous/ComfyUI?style=flat&sort=semver
[github-release-link]: https://github.com/comfyanonymous/ComfyUI/releases

![ComfyUI Screenshot](https://github.com/user-attachments/assets/7ccaf2c1-9b72-41ae-9a89-5688c94b7abe)
</div>

ComfyUI lets you design and execute advanced stable diffusion pipelines using a graph/nodes/flowchart based interface. Available on Windows, Linux, and macOS.

## 🔧 Dependencies and Installation

| mindspore  | ascend driver  |  firmware   |cann toolkit/kernel |
|:----------:|:--------------:|:-----------:|:------------------:|
|   2.7.1    |    24.1.RC2    | 7.3.0.1.231 |   8.2.RC1          |

```shell
git clone https://github.com/zhanghuiyao/comfyui_mindspore.git -b master
git clone https://github.com/mindspore-lab/mindone.git -b master

export PYTHONPATH=$PYTHONPATH:/path_to/mindone:/path_to/comfyui_mindspore

cd comfyui_mindspore
pip install -r requirements.txt
```

## Quick Strat

```shell
python main.py --listen 0.0.0.0 --port 9001
```

### performance, example as flux:

| models   | image size | infer_steps  | time cost |
|:-------:|:-------:|:-------:|:-------:|
| flux | 512*512 | 20 | ~6s |
| qwen_image | 512*512 | 20 | ~13s
