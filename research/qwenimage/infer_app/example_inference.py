import argparse
from functools import partial

import numpy as np
from models.pipeline import QwenImagePipeline
from models.transformer import QwenImageTransformer2DModel
from models.vae import AutoencoderKLQwenImage

import mindspore as ms
import mindspore.mint.distributed as dist
from mindspore.communication import GlobalComm

from mindone.trainers.zero import prepare_network


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a inference script.")
    parser.add_argument(
        "--model_id",
        type=str,
        default="Qwen/Qwen-Image",
        help="The model id in huggingface or the local path of the model's weights.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="qwenimage_inference_output.png",
        help="The output path where the model prediction will be written.",
    )
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible generation.")

    return parser.parse_args()


def main():
    args = parse_args()

    dist.init_process_group()
    ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL)

    local_rank = dist.get_rank()

    model_id = args.model_id
    transformer = QwenImageTransformer2DModel.from_pretrained(
        model_id, subfolder="transformer", mindspore_dtype=ms.float32
    )
    vae = AutoencoderKLQwenImage.from_pretrained(model_id, subfolder="vae", mindspore_dtype=ms.float16)
    pipe = QwenImagePipeline.from_pretrained(
        model_id,
        transformer=transformer,
        vae=vae,
        mindspore_dtype=ms.float16,
    )

    prompt = (
        'A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup," with a neon light '
        'beside it displaying "通义千问". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the '
        'poster is written "π≈3.1415926-53589793-23846264-33832795-02384197".'
    )
    negative_prompt = " "

    shard_fn = partial(prepare_network, zero_stage=3, optimizer_parallel_group=GlobalComm.WORLD_COMM_GROUP)
    pipe.transformer = shard_fn(pipe.transformer)
    pipe.text_encoder = shard_fn(pipe.text_encoder)

    dist.barrier()

    image = pipe(
        prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=50,
        true_cfg_scale=4.0,
        generator=np.random.Generator(np.random.PCG64(seed=args.seed)),
    )[0][0]

    if local_rank == 0:
        image.save(args.output_path)


if __name__ == "__main__":
    main()
