from share import *
import config

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.midas import MidasDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler


apply_midas = MidasDetector()

model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict('./models/control_sd15_depth.pth', location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)


def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, input_is_depthmap=False, process_images_in_series=False):
    with torch.no_grad():
        
        input_image = HWC3(input_image)
        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape
        
        if input_is_depthmap:
            detected_map = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LINEAR)
        else:
            detected_map, _ = apply_midas(resize_image(input_image, detect_resolution))
            detected_map = HWC3(detected_map)
            detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

        if process_images_in_series:
            num_images = num_samples
            num_samples = 1

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if process_images_in_series:
            seeds = [random.randint(0, 2147483647) for _ in range(num_images)]
        elif seed == -1:
            seed = random.randint((0, 2147483647))

        # seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        
        if process_images_in_series:
            x_samples = None
            for i in range(num_images):
                seed_everything(seeds[i])
                s, _ = ddim_sampler.sample(ddim_steps, num_samples,
                                            shape, cond, verbose=False, eta=eta,
                                            unconditional_guidance_scale=scale,
                                            unconditional_conditioning=un_cond)
                xs = model.decode_first_stage(s)
                xs = (einops.rearrange(xs, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
                if x_samples is None:
                    x_samples = xs
                else:
                    x_samples = np.concatenate([x_samples, xs], axis=0)

            num_samples = num_images
        else:
            seed_everything(seed)
            samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                        shape, cond, verbose=False, eta=eta,
                                                        unconditional_guidance_scale=scale,
                                                        unconditional_conditioning=un_cond)
            
            x_samples = model.decode_first_stage(samples)
            x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        results = [x_samples[i] for i in range(num_samples)]
    return [detected_map] + results


block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("## Control Stable Diffusion with Depth Maps")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source='upload', type="numpy")
            prompt = gr.Textbox(label="Prompt")
            run_button = gr.Button(label="Run")
            with gr.Accordion("Advanced options", open=False):
                input_is_depthmap = gr.Checkbox(label='Insert Depthmap Directly', value=True)
                num_samples = gr.Slider(label="Images", minimum=1, maximum=20, value=1, step=1)
                process_images_in_series = gr.Checkbox(label='Process Images In a Serie', value=True)
                image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=64)
                strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
                guess_mode = gr.Checkbox(label='Guess Mode', value=False)
                detect_resolution = gr.Slider(label="Depth Resolution", minimum=128, maximum=1024, value=384, step=1)
                ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
                scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
                seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
                eta = gr.Number(label="eta (DDIM)", value=0.0)
                a_prompt = gr.Textbox(label="Added Prompt", value='best quality, extremely detailed, realistic, extremely realistic, photo, photorealistic, human, person')
                n_prompt = gr.Textbox(label="Negative Prompt",
                                      value='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, drawing, draw, animated')
        with gr.Column():
            result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=2, height='auto')
    ips = [input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, input_is_depthmap, process_images_in_series]
    run_button.click(fn=process, inputs=ips, outputs=[result_gallery])


block.launch(server_name='0.0.0.0')
