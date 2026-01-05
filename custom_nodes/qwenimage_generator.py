import mindspore as ms
import numpy as np
from PIL import Image
from io import BytesIO
import base64
import asyncio
import aiohttp
import concurrent.futures


class AsyncQwenImageGenerator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "a beautiful landscape with mountains and lake"}),
                "width": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 64}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 64}),
                "num_inference_steps": ("INT", {"default": 50, "min": 1, "max": 200}),
                "true_cfg_scale": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 20.0, "step": 0.1}),
                "num_workers": ("INT", {"default": 4, "min": 1, "max": 8, "step": 1}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "base_seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "base_port": ("INT", {"default": 5000, "min": 1000, "max": 9999}),
                "timeout": ("INT", {"default": 15000, "min": 1000, "max": 300000, "step": 1000}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "info")
    FUNCTION = "generate_async"
    CATEGORY = "QwenImage"
    
    def __init__(self):
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)
    
    def call_api_sync(self, url, payload, timeout):
        """Synchronously calling the API in a new thread"""
        try:

            async def _call_api():
                async with aiohttp.ClientSession() as session:
                    aiohttp_timeout = aiohttp.ClientTimeout(total=timeout)
                    
                    async with session.post(
                        url, 
                        json=payload, 
                        timeout=aiohttp_timeout,
                        headers={'Content-Type': 'application/json'}
                    ) as response:
                        
                        if response.status == 200:
                            result = await response.json()
                            
                            if result.get("status") == "success" and "image_data" in result:
                                img_data = base64.b64decode(result["image_data"])
                                return Image.open(BytesIO(img_data))
                            else:
                                raise Exception(f"API error: {result.get('error', 'Unknown error')}")
                        else:
                            error_text = await response.text()
                            raise Exception(f"HTTP {response.status}: {error_text}")
                            
            # Running asynchronous code in a new thread
            return asyncio.run(_call_api())
            
        except asyncio.TimeoutError:
            raise Exception(f"Request timeout after {timeout}s")
        except Exception as e:
            raise Exception(f"API call failed: {str(e)}")
    
    def _generate_parallel_sync(self, prompt, negative_prompt, width, height,
                                num_inference_steps, true_cfg_scale, num_workers,
                                base_port, base_seed, timeout):
        # Create worker URLs and seed
        urls = [f"http://127.0.0.1:{base_port + i}/qwenimage-api" for i in range(num_workers)]
        # Payload for each worker
        payloads = []
        for i in range(num_workers):

            payload = {
                "prompts": prompt.strip(),
                "num_inference_steps": num_inference_steps,
                "negative_prompt": negative_prompt.strip() if negative_prompt.strip() else " ",
                "true_cfg_scale": true_cfg_scale,
                "width": width,
                "height": height,
                "seed": base_seed
            }

            payloads.append(payload)
        
        # Use a thread pool to execute all requests in parallel
        futures = []
        for url, payload in zip(urls, payloads):
            future = self.executor.submit(self.call_api_sync, url, payload, timeout)
            futures.append(future)
        
        # Collect results
        successful_images = []
        errors = []
        
        for i, future in enumerate(futures):
            try:
                result = future.result(timeout=timeout)
                if result is not None:
                    successful_images.append(result)
                else:
                    errors.append(f"Worker {i}: No result returned")
            except Exception as e:
                errors.append(f"Worker {i}: {str(e)}")
        
        return successful_images, errors
    
    def generate_async(self, prompt: str, width: int, height: int, 
                      num_inference_steps: int, true_cfg_scale: float,
                      num_workers: int = 4, negative_prompt: str = "",
                      base_seed: int = 42, base_port: int = 5000,
                      timeout: int = 15000):

        try:
            print(f"[AsyncQwen] Starting generation with prompt: {prompt[:50]}...")

            successful_images, errors = self._generate_parallel_sync(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                true_cfg_scale=true_cfg_scale,
                num_workers=num_workers,
                base_port=base_port,
                base_seed=base_seed,
                timeout=timeout
            )

            if not successful_images:
                error_msg = "All workers failed: " + "; ".join(errors)
                print(f"[AsyncQwen] {error_msg}")
                return (self.create_error_tensor(height, width), error_msg)

            # Convert to tensor
            image_tensors = []
            for image in successful_images:
                # to RGB
                if image.mode != 'RGB':
                    image = image.convert('RGB')

                # to numpy
                image_np = np.array(image).astype(np.float32) / 255.0

                # (H, W, C) -> (1, H, W, C)
                if len(image_np.shape) == 3:
                    image_tensor = ms.from_numpy(image_np).unsqueeze(0)
                # (B, H, W, C)
                elif len(image_np.shape) == 4:
                    image_tensor = ms.from_numpy(image_np)
                
                image_tensors.append(image_tensor)

            # Collect tensors
            if len(image_tensors) > 1:
                final_tensor = image_tensors[0]
                # final_tensor = ms.mint.cat(image_tensors, dim=0)  # image_num = num_workers
            else:
                final_tensor = image_tensors

            # Create Information String
            info = f"Generated {len(successful_images)} images | Steps: {num_inference_steps} | Workers: {num_workers}"
            if errors:
                info += f" | Errors: {len(errors)}"

            return (final_tensor, info)
            
        except Exception as e:
            error_msg = f"Generation failed: {str(e)}"
            print(f"[AsyncQwen] {error_msg}")
            import traceback
            traceback.print_exc()
            return (self.create_error_tensor(height, width), error_msg)
    
    def create_error_tensor(self, height: int, width: int) -> ms.Tensor:
        """Create an error placeholder tensor"""
        error_image = np.zeros((3, height, width), dtype=np.float32)
        error_image[0, :, :] = 1.0  # red channel set to 1
        
        return ms.from_numpy(error_image).unsqueeze(0)  # (1, 3, height, width)

# Register node
NODE_CLASS_MAPPINGS = {
    "AsyncQwenImageGenerator": AsyncQwenImageGenerator,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AsyncQwenImageGenerator": "Async Qwen Image Generator",
}
print(f"Available nodes: {list(NODE_CLASS_MAPPINGS.keys())}")
