import torch
from diffusers import Flux2KleinPipeline
from diffusers.utils import load_image
from PIL import Image

DEFAULT_STEPS = 4
DEFAULT_GUIDANCE = 1.0


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

    def edit(self, image_path, prompt, mode="general", output_path=None):
        image = load_image(image_path)
        result = self.pipe(
            prompt=prompt,
            image=image,
            guidance_scale=DEFAULT_GUIDANCE,
            num_inference_steps=DEFAULT_STEPS,
        ).images[0]
        if (result.width, result.height) != (image.width, image.height):
            result = result.resize((image.width, image.height), Image.LANCZOS)
        if output_path:
            result.save(output_path)
        return result
