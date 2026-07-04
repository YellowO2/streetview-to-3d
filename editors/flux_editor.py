import numpy as np
import torch
from diffusers import Flux2KleinPipeline
from diffusers.utils import load_image
from PIL import Image

DEFAULT_STEPS = 4
DEFAULT_GUIDANCE = 1.0

PANO_ASPECT_THRESHOLD = 1.8


class FluxEditor:
    def __init__(
        self,
        model_id="black-forest-labs/FLUX.2-klein-9B",
        device="cuda",
        offload=False,
    ):
        self.device = device
        self.pipe = Flux2KleinPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
        )
        if offload:
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(device)

    def _run_pipe(self, image, prompt):
        result = self.pipe(
            prompt=prompt,
            image=image,
            guidance_scale=DEFAULT_GUIDANCE,
            num_inference_steps=DEFAULT_STEPS,
        ).images[0]
        if (result.width, result.height) != (image.width, image.height):
            result = result.resize((image.width, image.height), Image.LANCZOS)
        return result

    def _chunked_pano_edit(self, image, prompt):
        W, H = image.size
        chunk_w = H
        stride = chunk_w // 2
        n = max(1, W // stride)

        chunks = []
        for i in range(n):
            x = i * stride
            if x + chunk_w <= W:
                c = image.crop((x, 0, x + chunk_w, H))
            else:
                left = image.crop((x, 0, W, H))
                right = image.crop((0, 0, (x + chunk_w) - W, H))
                c = Image.new("RGB", (chunk_w, H))
                c.paste(left, (0, 0))
                c.paste(right, (W - x, 0))
            chunks.append(c)

        edited_arrs = []
        for i, c in enumerate(chunks):
            print(f"  pano chunk {i+1}/{n} editing...")
            out = self._run_pipe(c, prompt)
            edited_arrs.append(np.array(out, dtype=np.float32))

        w_1d = 1.0 - np.abs(2.0 * (np.arange(chunk_w) + 0.5) / chunk_w - 1.0)
        w_1d = np.clip(w_1d, 0.01, 1.0).astype(np.float32)
        w_2d = np.tile(w_1d, (H, 1))

        ext_w = W + chunk_w
        acc = np.zeros((H, ext_w, 3), dtype=np.float32)
        wsum = np.zeros((H, ext_w), dtype=np.float32)
        for i, arr in enumerate(edited_arrs):
            x = i * stride
            acc[:, x:x + chunk_w, :] += arr * w_2d[:, :, None]
            wsum[:, x:x + chunk_w] += w_2d

        tail = ext_w - W
        if tail > 0:
            acc[:, :tail, :] += acc[:, W:W + tail, :]
            wsum[:, :tail] += wsum[:, W:W + tail]

        result = acc[:, :W, :] / wsum[:, :W, None]
        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

    def edit(self, image_path, prompt, mode="general", output_path=None):
        image = load_image(image_path)
        aspect = image.width / image.height
        if aspect >= PANO_ASPECT_THRESHOLD:
            print(f"Editing pano {image_path} (mode={mode}, chunked)...")
            result = self._chunked_pano_edit(image, prompt)
        else:
            print(f"Editing image {image_path} (mode={mode})...")
            result = self._run_pipe(image, prompt)
        if output_path:
            result.save(output_path)
        return result
