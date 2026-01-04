import asyncio
import base64
from io import BytesIO

import aiohttp
from PIL import Image


# calling function
def call_api_gen(url):
    try:

        async def _fn(samples, *args, **kwargs):
            async with aiohttp.ClientSession() as sess:
                data = {
                    "prompts": samples,
                    "num_inference_steps": kwargs.get("num_inference_steps", 50),
                    "negative_prompt": kwargs.get("negative_prompt", " "),
                    "true_cfg_scale": kwargs.get("true_cfg_scale", 4.0),
                }

                timeout = aiohttp.ClientTimeout(total=15000)

                async with sess.post(url, json=data, timeout=timeout) as response:
                    if response.status == 200:
                        result = await response.json()

                        if result.get("status") == "success":
                            if "image_data" in result:
                                img_data = base64.b64decode(result["image_data"])
                                return Image.open(BytesIO(img_data))
                            else:
                                raise Exception("No image data in response")
                        else:
                            raise Exception(f"API error: {result.get('error', 'Unknown error')}")
                    else:
                        raise Exception(f"HTTP {response.status}")

    except Exception as e:
        print(f"Error calling API: {e}")
        raise

    return _fn


# default parameters
port = 5000
worker_num = 4

# get pipes for different ports
urls = [f"http://127.0.0.1:{port + i}/qwenimage-api" for i in range(worker_num)]
pipes = [call_api_gen(url) for url in urls]


# function for passing inference requests
async def run_all(pipes, prompt):
    results = await asyncio.gather(*[pipe(prompt) for pipe in pipes])
    return results


# inference parameters
prompt = (
    'A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup," with a neon light '
    'beside it displaying "通义千问". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the '
    'poster is written "π≈3.1415926-53589793-23846264-33832795-02384197".'
)

# call and get results from server
results = asyncio.run(run_all(pipes, prompt))

# save image
results[0].save("generated_image.png")
