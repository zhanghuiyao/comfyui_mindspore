import argparse
import base64
import os
import threading
from functools import partial
from io import BytesIO

import numpy as np
from flask import Blueprint, Flask, request
from flask_restful import Api, Resource
from models.pipeline import QwenImagePipeline
from models.transformer import QwenImageTransformer2DModel
from models.vae import AutoencoderKLQwenImage

import mindspore as ms
import mindspore.mint.distributed as dist
from mindspore.communication import GlobalComm

from mindone.trainers.zero import prepare_network


def parsed_args():
    parser = argparse.ArgumentParser(description="QwenImage API Functions")
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    return args


class QwenImageAppPipeline(Resource):
    def __init__(self, model_id):
        # perpare components with given dtype
        transformer = QwenImageTransformer2DModel.from_pretrained(
            model_id, subfolder="transformer", mindspore_dtype=ms.float32
        )
        vae = AutoencoderKLQwenImage.from_pretrained(model_id, subfolder="vae", mindspore_dtype=ms.float16)
        self.model = QwenImagePipeline.from_pretrained(
            model_id,
            transformer=transformer,
            vae=vae,
            mindspore_dtype=ms.float16,
        )

        # apply zero3
        shard_fn = partial(prepare_network, zero_stage=3, optimizer_parallel_group=GlobalComm.WORLD_COMM_GROUP)
        self.model.transformer = shard_fn(self.model.transformer)
        self.model.text_encoder = shard_fn(self.model.text_encoder)

        # wait for all NPUs
        dist.barrier()

        print("Loaded QwenImage pipeline")

    def generate(self, prompts, *args, **kwargs):
        seed = kwargs.get("seed", 42)
        image = self.model(
            prompts,
            negative_prompt=kwargs.get("negative_prompt", " "),
            num_inference_steps=kwargs.get("num_inference_steps", 50),
            true_cfg_scale=kwargs.get("true_cfg_scale", 4.0),
            generator=np.random.Generator(np.random.PCG64(seed=seed)),
        )[0][0]

        return image


lock = threading.Lock()


class QwenImageAPI(Resource):
    def __init__(self, qwenimage_pipeline):
        self.qwenimage_pipeline = qwenimage_pipeline

    def post(self):
        with lock:
            try:
                data = request.get_json()

                if not data:
                    return {"error": "No data provided"}, 400

                feature = {}
                allowed_keys = {"prompts", "num_inference_steps", "negative_prompt", "true_cfg_scale"}
                for key, value in data.items():
                    if key in allowed_keys and value is not None:
                        feature[key] = value

                image = self.qwenimage_pipeline.generate(**feature)

                buffered = BytesIO()
                image.save(buffered, format="PNG")
                buffered.seek(0)
                img_str = base64.b64encode(buffered.getvalue()).decode()

                response_data = {"status": "success", "image_data": img_str, "format": "png"}

                return response_data, 200

            except Exception as e:
                return {"error": str(e)}, 500


class RemoteServer(object):
    def __init__(self, args) -> None:
        self.app = Flask(__name__)
        root = Blueprint("root", __name__)
        self.app.register_blueprint(root)
        api = Api(self.app)

        self.qwenimage_pipeline = QwenImageAppPipeline(model_id=os.path.join(args.model_dir))
        api.add_resource(
            QwenImageAPI,
            "/qwenimage-api",
            resource_class_args=[self.qwenimage_pipeline],
        )

    def run(self, host="127.0.0.1", port=5000):
        self.app.run(host, port=port, threaded=True, debug=False)


if __name__ == "__main__":
    args = parsed_args()

    ms.set_context(
        mode=ms.PYNATIVE_MODE,
        device_target="Ascend",
        jit_config={"jit_level": "O0"},
        deterministic="ON",
        pynative_synchronize=True,
        memory_optimize_level="O1",
        max_device_memory="59GB",
        # jit_syntax_level=ms.STRICT,
    )

    dist.init_process_group()
    ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL)
    ms.launch_blocking()

    flask_server = RemoteServer(args)
    port_for_rank = int(args.port) + dist.get_rank()
    flask_server.run(host="127.0.0.1", port=port_for_rank)
