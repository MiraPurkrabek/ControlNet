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
from annotator.openpose import OpenposeDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler


apply_openpose = OpenposeDetector()

model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict('./models/control_sd15_openpose.pth', location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)

SKELETON = [
    [0, 1],
    [1, 2],
    [1, 5],
    [2, 3],
    [3, 4],
    [5, 6],
    [6, 7],
    [2, 8],
    [5, 11],
    [8, 11],
    [8, 9],
    [9, 10],
    [11, 12],
    [12, 13],
    [0, 14],
    [0, 15],
    [14, 16],
    [15, 17],
]


def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta):
    with torch.no_grad():
        input_image = HWC3(input_image)
        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape
        
        detected_map_full, kpts = apply_openpose(resize_image(input_image, detect_resolution))
        detected_map_full = HWC3(detected_map_full)
        detected_map = cv2.resize(detected_map_full, (W, H), interpolation=cv2.INTER_NEAREST)

        # Process the keypoints
        kpts = np.array(kpts["candidate"])
        kpts[:, 0] /= detected_map_full.shape[1]
        kpts[:, 1] /= detected_map_full.shape[0]
        kpts[:, 0] *= W
        kpts[:, 1] *= H

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        # results = [x_samples[i] for i in range(num_samples)]
        results = []
        print(kpts)
        for i in range(num_samples):
            src = x_samples[i].astype(np.uint8).copy()
            print("src.shape", src.shape)
            print("src type", type(src))
            for kpt in kpts:
                src = cv2.circle(
                    src,
                    (int(kpt[0]), int(kpt[1])),
                    radius=4,
                    color=(255, 0, 0),
                    thickness=-1,
                )

            for bone in SKELETON:
                if kpt.shape[0] < 19:
                    k0 = kpts[bone[0]]
                    k1 = kpts[bone[1]]
                else:
                    k0 = kpts[bone[0]+1]
                    k1 = kpts[bone[1]+1]
                src = cv2.line(
                    src,
                    (int(k0[0]), int(k0[1])),
                    (int(k1[0]), int(k1[1])),
                    color=(255, 0, 0),
                    thickness=2,
                )

            results.append(src)
        # alpha = 0.3
        # results = [cv2.addWeighted(
        #     x_samples[i],
        #     alpha,
        #     detected_map,
        #     1-alpha, 
        #     0.0,
        # ) for i in range(num_samples)]
            
        print("Results:")
        print(results[0].shape)
        print(type(results[0]))
        print("Detected map:")
        print(detected_map.shape)
        print(type(detected_map))

    return [detected_map] + results


block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("## Control Stable Diffusion with Human Pose")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source='upload', type="numpy")
            prompt = gr.Textbox(label="Prompt")
            run_button = gr.Button(label="Run")
            with gr.Accordion("Advanced options", open=True):
                num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
                image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=256, step=64)
                strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
                guess_mode = gr.Checkbox(label='Guess Mode', value=False)
                detect_resolution = gr.Slider(label="OpenPose Resolution", minimum=128, maximum=1024, value=512, step=1)
                ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=10, step=1)
                scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
                seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, value=-1, randomize=True)
                eta = gr.Number(label="eta (DDIM)", value=0.0)
                a_prompt = gr.Textbox(label="Added Prompt", value='best quality, extremely detailed')
                n_prompt = gr.Textbox(label="Negative Prompt",
                                      value='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality')
        with gr.Column():
            result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=2, height='auto')
    ips = [input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta]
    run_button.click(fn=process, inputs=ips, outputs=[result_gallery])


block.launch(server_name='0.0.0.0')
