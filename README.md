# Stable Diffusion WebUI Forge/reForge

Stable Diffusion WebUI Forge/reForge is a platform on top of [Stable Diffusion WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) (based on [Gradio](https://www.gradio.app/)) to make development easier, optimize resource management, speed up inference, and study experimental features.

The name "Forge" is inspired from "Minecraft Forge". This project is aimed at becoming SD WebUI's Forge.

# Important: Branches

* main: Has all the possible upstream changes from A1111, should be more stable if coming from stock Forge/A1111. It may be missing some new features related to the comfy backend (from 2024-01 and onwards when it's not samplers or model managament).
* dev_upstream: Has all the possible upstream changes from A1111 and all possible backend upstream changes from Comfy. For now it's a bit faster than main. It can be unstable. It has some new features, optimizations, etc.
* main_new_forge: Has all the possible upstream changes from A1111 and all possible backend upstream changes from new OG Forge. It is gradio 4.x and experimental. It will be pretty simialr to OG Forge with some reForge custom features on top of it (like samplers, schedulers, extensions, etc). It probably will support new models (SD3, Flux, etc)

## If you were using dev_upstream_a1111 or dev_upstream_a1111_customschedulers branches.
You will get an issue when trying to do git pull, so do this on your reForge folder:
```bash
git fetch origin
git checkout main
```
Or if you want to go to dev_upstream
```bash
git fetch origin
git checkout dev_upstream
```
# reForge comes with updates from A1111 and others' features, and for now, we have:

* Scheduler Selection
* DoRA Support
* Small Performance Optimizations (based on small tests on txt2img, it is a bit faster than Forge on a RTX 4090 and SDXL)
* Refiner bugfixes
* Optimized cache
* Soft Inpainting
* Multiple checkpoints loaded at the same time (even if using pin-shared-memory!)
* A lot of new samplers!
* NMS (Negative Guidance minimum sigma all steps)
* Skip Early CFG (Ignore negative prompt during early sampling)
* NGMS all steps
* Fixed png_info for API
* Fixed built*in controlnet not accepting ControlMode/Resizemode as int
* Restored forge custom tree*view for loras, instead of the A1111 one
* More Upscalers: COMPACT, GRL, OmniSR, SPAN, and SRFormer and fixes from https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/14794
* Maybe more!

# Installing Forge/reForge

Tutorial from: https://github.com/continue-revolution/sd-webui-animatediff/blob/forge/master/docs/how-to-use.md#you-have-a1111-and-you-know-git
### You have A1111/Stock Forge and you know Git
I suggest 2 paths here. Seems after a lot of changes, git reset --hard introduces some issues. So for now, we will try with git stash instead.
Option 1:
If you have already had OG A1111 and you are familiar with git, I highly recommend running the following commands in your terminal in `/path/to/stable-diffusion-webui`
```bash
git remote add reForge https://github.com/Panchovix/stable-diffusion-webui-reForge
git branch Panchovix/main
git checkout Panchovix/main
git fetch reForge
git branch -u reForge/main
git stash
git pull
```
To go back to OG A1111, just do `git checkout master` or `git checkout dev`.

If you got stuck in a merge to resolve conflicts, you can go back with `git merge --abort`

Option 2: If instructions above don't work, I suggest doing a clean install with the instructions below, and then moving the folders (extensions, models, etc) into the reForge folder.

### You don't have A111 or doing a clean install.

If you know what you are doing, you can install Forge/reForge using same method as SD-WebUI. (Install Git, Python, Git Clone the reForge repo `https://github.com/Panchovix/stable-diffusion-webui-reForge.git` and then run webui-user.bat):

```bash
git clone https://github.com/Panchovix/stable-diffusion-webui-reForge.git
cd stable-diffusion-webui-reForge
git checkout main
```
Then run webui-user.bat or webui-user.sh.

When you want to update:
```bash
cd stable-diffusion-webui-reForge
git pull
```

Pre-done package is WIP.

# Performance comparison of main branch vs A1111 vs Stock Forge (2024-07-18).

I did these comparisons with newer main branch. Both using the same venv.

* A1111 flags: --xformers --precision half --opt-channelslast

* ReForge flags: --xformers --always-gpu --disable-nan-check -cuda-malloc --cuda-stream --pin-shared-memory

* Forge flags: --xformers --always-gpu --disable-nan-check -cuda-malloc --cuda-stream --pin-shared-memory

* DPM++ 2M, AYS, 25 steps, 10 hi-res step with Restart, Adetailer, RTX 4090.

reForge:

* No LoRA:
```
https://pastebin.com/55QkuKpL

Total time moving the model -> 0.01+0.03+0.01+0.00+0.02+0.01+0.01+0.03+0.01 = 0.13 seconds in total.

Total inference time: 16 seconds.
```

* With 220MB LoRA:
```
https://pastebin.com/QNC3hNhR

Total time moving the model -> 0.24+0.34+0.04+0.00+0.28+0.04+0.10+0.35+0.05 = 1.44 seconds in total.

Total inference time: 17 seconds.
```

* With 1.4GB LoRA:
```
https://pastebin.com/GJ34a4yv

Total time moving the model -> 0.45+0.69+0.05+0.00+0.44+0.05+0.15+0.46+0.04 = 2.33 seconds in total.

Total inference time: 18 seconds.
```

* With 3.8GB LoRA:
```
https://pastebin.com/12Lx6QdM

Total time moving the model ->  0.45+0.64+0.06+0.00+0.62+0.05+0.21+0.65+0.04 = 2.72 seconds in total.

Total inference time: 18 seconds.
```

Forge:

* No LoRA:
```
Time taken: 16.6 sec. (0.6s more vs reForge)
```

* With 220MB LoRA:
```
Time taken: 17.2 sec. (0.2s more vs reForge)
```

* With 1.4GB LoRA:
```
Time taken: 18.0 sec. (same as reForge)
```

* With 3.8GB LoRA:
```
Time taken: 18.4 sec. (0.4s more vs reForge)
```

A1111:

* No LoRA:
```
Time taken: 19.2 sec. (3.2s more vs reForge)
```

* With 220MB LoRA:
```
Time taken: 20.9 sec. (3.9s more vs reForge)
```

* With 1.4GB LoRA:
```
Time taken: 26.3 sec. (8.3s more vs reForge)
```

* With 3.8GB LoRA:
```
Time taken: 34.4 sec. (16.4s more vs reForge)
```

# Screenshots of Comparison (by Illyasviel)

I tested with several devices, and this is a typical result from 8GB VRAM (3070ti laptop) with SDXL.

**This is original WebUI:**

![image](https://github.com/lllyasviel/stable-diffusion-webui-forge/assets/19834515/16893937-9ed9-4f8e-b960-70cd5d1e288f)

![image](https://github.com/lllyasviel/stable-diffusion-webui-forge/assets/19834515/7bbc16fe-64ef-49e2-a595-d91bb658bd94)

![image](https://github.com/lllyasviel/stable-diffusion-webui-forge/assets/19834515/de1747fd-47bc-482d-a5c6-0728dd475943)

![image](https://github.com/lllyasviel/stable-diffusion-webui-forge/assets/19834515/96e5e171-2d74-41ba-9dcc-11bf68be7e16)

(average about 7.4GB/8GB, peak at about 7.9GB/8GB)

**This is WebUI Forge/reForge:**

![image](https://github.com/lllyasviel/stable-diffusion-webui-forge/assets/19834515/ca5e05ed-bd86-4ced-8662-f41034648e8c)

![image](https://github.com/lllyasviel/stable-diffusion-webui-forge/assets/19834515/3629ee36-4a99-4d9b-b371-12efb260a283)

![image](https://github.com/lllyasviel/stable-diffusion-webui-forge/assets/19834515/6d13ebb7-c30d-4aa8-9242-c0b5a1af8c95)

![image](https://github.com/lllyasviel/stable-diffusion-webui-forge/assets/19834515/c4f723c3-6ea7-4539-980b-0708ed2a69aa)

(average and peak are all 6.3GB/8GB)

You can see that Forge/reForge does not change WebUI results. Installing Forge/reForge is not a seed breaking change. 

Forge/reForge can perfectly keep WebUI unchanged even for most complicated prompts like `fantasy landscape with a [mountain:lake:0.25] and [an oak:a christmas tree:0.75][ in foreground::0.6][ in background:0.25] [shoddy:masterful:0.5]`.

All your previous works still work in Forge/reForge!

# Forge/reForge Backend

Forge/reForge backend removes all WebUI's codes related to resource management and reworked everything. All previous CMD flags like `medvram, lowvram, medvram-sdxl, precision full, no half, no half vae, attention_xxx, upcast unet`, ... are all **REMOVED**. Adding these flags will not cause error but they will not do anything now. **We highly encourage Forge/reForge users to remove all cmd flags and let Forge/reForge to decide how to load models.**

Without any cmd flag, Forge/reForge can run SDXL with 4GB vram and SD1.5 with 2GB vram.

**Some flags that you may still pay attention to:** 

1. `--always-offload-from-vram` (This flag will make things **slower** but less risky). This option will let Forge/reForge always unload models from VRAM. This can be useful if you use multiple software together and want Forge/reForge to use less VRAM and give some VRAM to other software, or when you are using some old extensions that will compete vram with Forge/reForge, or (very rarely) when you get OOM.

2. `--cuda-malloc` (This flag will make things **faster** but more risky). This will ask pytorch to use *cudaMallocAsync* for tensor malloc. On some profilers I can observe performance gain at millisecond level, but the real speed up on most my devices are often unnoticed (about or less than 0.1 second per image). This cannot be set as default because many users reported issues that the async malloc will crash the program. Users need to enable this cmd flag at their own risk.

3. `--cuda-stream` (This flag will make things **faster** but more risky). This will use pytorch CUDA streams (a special type of thread on GPU) to move models and compute tensors simultaneously. This can almost eliminate all model moving time, and speed up SDXL on 30XX/40XX devices with small VRAM (eg, RTX 4050 6GB, RTX 3060 Laptop 6GB, etc) by about 15\% to 25\%. However, this unfortunately cannot be set as default because I observe higher possibility of pure black images (Nan outputs) on 2060, and higher chance of OOM on 1080 and 2060. When the resolution is large, there is a chance that the computation time of one single attention layer is longer than the time for moving entire model to GPU. When that happens, the next attention layer will OOM since the GPU is filled with the entire model, and no remaining space is available for computing another attention layer. Most overhead detecting methods are not robust enough to be reliable on old devices (in my tests). Users need to enable this cmd flag at their own risk.

4. `--pin-shared-memory` (This flag will make things **faster** but more risky). Effective only when used together with `--cuda-stream`. This will offload modules to Shared GPU Memory instead of system RAM when offloading models. On some 30XX/40XX devices with small VRAM (eg, RTX 4050 6GB, RTX 3060 Laptop 6GB, etc), I can observe significant (at least 20\%) speed-up for SDXL. However, this unfortunately cannot be set as default because the OOM of Shared GPU Memory is a much more severe problem than common GPU memory OOM. Pytorch does not provide any robust method to unload or detect Shared GPU Memory. Once the Shared GPU Memory OOM, the entire program will crash (observed with SDXL on GTX 1060/1050/1066), and there is no dynamic method to prevent or recover from the crash. Users need to enable this cmd flag at their own risk.

CMD flags are on ldm_patches/modules/args_parser.py and on the normal A1111 path (modules/cmd_args.py)

    --disable-xformers
        Disables xformers, to use other attentions like SDP.
    --attention-split
        Use the split cross attention optimization. Ignored when xformers is used.
    --attention-quad
        Use the sub-quadratic cross attention optimization . Ignored when xformers is used.
    --attention-pytorch
        Use the new pytorch 2.0 cross attention function.
    --disable-attention-upcast
        Disable all upcasting of attention. Should be unnecessary except for debugging.
    --gpu-device-id
        Set the id of the cuda device this instance will use.

(VRAM related)

    --always-gpu
        Store and run everything (text encoders/CLIP models, etc... on the GPU).
    --always-high-vram
        By default models will be unloaded to CPU memory after being used. This option keeps them in GPU memory.
    --always-normal-vram
        Used to force normal vram use if lowvram gets automatically enabled.
    --always-low-vram
        Split the unet in parts to use less vram.
    --always-no-vram
        When lowvram isn't enough.
    --always-cpu
        To use the CPU for everything (slow).

(float point type)

    --all-in-fp32
    --all-in-fp16
    --unet-in-bf16
    --unet-in-fp16
    --unet-in-fp8-e4m3fn
    --unet-in-fp8-e5m2
    --vae-in-fp16
    --vae-in-fp32
    --vae-in-bf16
    --clip-in-fp8-e4m3fn
    --clip-in-fp8-e5m2
    --clip-in-fp16
    --clip-in-fp32

(rare platforms)

    --directml
    --disable-ipex-hijack
    --pytorch-deterministic

Again, Forge/reForge do not recommend users to use any cmd flags unless you are very sure that you really need these.

# Contribution

I have forked the original Forge repo https://github.com/lllyasviel/stable-diffusion-webui-forge and then, applied most of the updates I could that are from the dev branch of A1111 webui https://github.com/AUTOMATIC1111/stable-diffusion-webui, re-visiting and seeing to apply them each by each by hand, so this way, we can have the speed and vram maganements advtanges with (some) new features.

All PRs that can be implemented in https://github.com/AUTOMATIC1111/stable-diffusion-webui/tree/dev could submit PRs here as well.

The list of what doesn't work/I couldn't/didn't know how to merge/fix:
* SD3 (Since forge has it's own unet implementation, I didn't tinker on implementing it). Any help to implement this would be really appreciated!
* Callback order (https://github.com/AUTOMATIC1111/stable-diffusion-webui/commit/5bd27247658f2442bd4f08e5922afff7324a357a), specifically because the forge implementation of modules doesn't have script_callbacks. So it broke the included controlnet extension and ui_settings.py.
* Didn't tinker much about changes that affect extensions-builtin\Lora, since forge does it mostly on ldm_patched\modules. But some of them could be applied thay I have may skipped.
* precision-half (forge has this by default)
* New "is_sdxl" flag (sdxl works fine, but there are some new things that don't work without this flag)
* DDIM CFG++ (because the edit on sd_samplers_cfg_denoiser.py). Researching on how to modify sd_samplers_cfg_denoiser.py to make it work. EDIT: In the progress to make it work.
* Newer controlnet updates into the built-in extension.
* Probably others things, you can check and discuss here [https://github.com/Panchovix/stable-diffusion-webui-reForge/issues/1](https://github.com/Panchovix/stable-diffusion-webui-reForge/discussions/11)

Feel free to submit PRs related to the functionality of Forge here. Any help will be really appreciated!

# UNet Patcher

The full name of the backend is `Stable Diffusion WebUI with Forge/reForge backend`, or for simplicity, the `Forge backend`. The API and python symbols are made similar to previous software only for reducing the learning cost of developers. Backend has a high percentage of Comfy code, about 80-85% or so.

Now developing an extension is super simple. We finally have a patchable UNet.

Below is using one single file with 80 lines of codes to support FreeU:

`extensions-builtin/sd_forge_freeu/scripts/forge_freeu.py`

```python
import torch
import gradio as gr
from modules import scripts


def Fourier_filter(x, threshold, scale):
    x_freq = torch.fft.fftn(x.float(), dim=(-2, -1))
    x_freq = torch.fft.fftshift(x_freq, dim=(-2, -1))
    B, C, H, W = x_freq.shape
    mask = torch.ones((B, C, H, W), device=x.device)
    crow, ccol = H // 2, W //2
    mask[..., crow - threshold:crow + threshold, ccol - threshold:ccol + threshold] = scale
    x_freq = x_freq * mask
    x_freq = torch.fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = torch.fft.ifftn(x_freq, dim=(-2, -1)).real
    return x_filtered.to(x.dtype)


def set_freeu_v2_patch(model, b1, b2, s1, s2):
    model_channels = model.model.model_config.unet_config["model_channels"]
    scale_dict = {model_channels * 4: (b1, s1), model_channels * 2: (b2, s2)}

    def output_block_patch(h, hsp, *args, **kwargs):
        scale = scale_dict.get(h.shape[1], None)
        if scale is not None:
            hidden_mean = h.mean(1).unsqueeze(1)
            B = hidden_mean.shape[0]
            hidden_max, _ = torch.max(hidden_mean.view(B, -1), dim=-1, keepdim=True)
            hidden_min, _ = torch.min(hidden_mean.view(B, -1), dim=-1, keepdim=True)
            hidden_mean = (hidden_mean - hidden_min.unsqueeze(2).unsqueeze(3)) / \
                          (hidden_max - hidden_min).unsqueeze(2).unsqueeze(3)
            h[:, :h.shape[1] // 2] = h[:, :h.shape[1] // 2] * ((scale[0] - 1) * hidden_mean + 1)
            hsp = Fourier_filter(hsp, threshold=1, scale=scale[1])
        return h, hsp

    m = model.clone()
    m.set_model_output_block_patch(output_block_patch)
    return m


class FreeUForForge(scripts.Script):
    def title(self):
        return "FreeU Integrated"

    def show(self, is_img2img):
        # make this extension visible in both txt2img and img2img tab.
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            freeu_enabled = gr.Checkbox(label='Enabled', value=False)
            freeu_b1 = gr.Slider(label='B1', minimum=0, maximum=2, step=0.01, value=1.01)
            freeu_b2 = gr.Slider(label='B2', minimum=0, maximum=2, step=0.01, value=1.02)
            freeu_s1 = gr.Slider(label='S1', minimum=0, maximum=4, step=0.01, value=0.99)
            freeu_s2 = gr.Slider(label='S2', minimum=0, maximum=4, step=0.01, value=0.95)

        return freeu_enabled, freeu_b1, freeu_b2, freeu_s1, freeu_s2

    def process_before_every_sampling(self, p, *script_args, **kwargs):
        # This will be called before every sampling.
        # If you use highres fix, this will be called twice.
        
        freeu_enabled, freeu_b1, freeu_b2, freeu_s1, freeu_s2 = script_args

        if not freeu_enabled:
            return

        unet = p.sd_model.forge_objects.unet

        unet = set_freeu_v2_patch(unet, freeu_b1, freeu_b2, freeu_s1, freeu_s2)

        p.sd_model.forge_objects.unet = unet

        # Below codes will add some logs to the texts below the image outputs on UI.
        # The extra_generation_params does not influence results.
        p.extra_generation_params.update(dict(
            freeu_enabled=freeu_enabled,
            freeu_b1=freeu_b1,
            freeu_b2=freeu_b2,
            freeu_s1=freeu_s1,
            freeu_s2=freeu_s2,
        ))

        return
```

It looks like this:

![image](https://github.com/lllyasviel/stable-diffusion-webui-forge/assets/19834515/277bac6e-5ea7-4bff-b71a-e55a60cfc03c)

Similar components like HyperTile, KohyaHighResFix, SAG, can all be implemented within 100 lines of codes (see also the codes).

![image](https://github.com/lllyasviel/stable-diffusion-webui-forge/assets/19834515/06472b03-b833-4816-ab47-70712ac024d3)

ControlNets can finally be called by different extensions.

Implementing Stable Video Diffusion and Zero123 are also super simple now (see also the codes). 

*Stable Video Diffusion:*

`extensions-builtin/sd_forge_svd/scripts/forge_svd.py`

```python
import torch
import gradio as gr
import os
import pathlib

from modules import script_callbacks
from modules.paths import models_path
from modules.ui_common import ToolButton, refresh_symbol
from modules import shared

from modules_forge.forge_util import numpy_to_pytorch, pytorch_to_numpy
from ldm_patched.modules.sd import load_checkpoint_guess_config
from ldm_patched.contrib.external_video_model import VideoLinearCFGGuidance, SVD_img2vid_Conditioning
from ldm_patched.contrib.external import KSampler, VAEDecode


opVideoLinearCFGGuidance = VideoLinearCFGGuidance()
opSVD_img2vid_Conditioning = SVD_img2vid_Conditioning()
opKSampler = KSampler()
opVAEDecode = VAEDecode()

svd_root = os.path.join(models_path, 'svd')
os.makedirs(svd_root, exist_ok=True)
svd_filenames = []


def update_svd_filenames():
    global svd_filenames
    svd_filenames = [
        pathlib.Path(x).name for x in
        shared.walk_files(svd_root, allowed_extensions=[".pt", ".ckpt", ".safetensors"])
    ]
    return svd_filenames


@torch.inference_mode()
@torch.no_grad()
def predict(filename, width, height, video_frames, motion_bucket_id, fps, augmentation_level,
            sampling_seed, sampling_steps, sampling_cfg, sampling_sampler_name, sampling_scheduler,
            sampling_denoise, guidance_min_cfg, input_image):
    filename = os.path.join(svd_root, filename)
    model_raw, _, vae, clip_vision = \
        load_checkpoint_guess_config(filename, output_vae=True, output_clip=False, output_clipvision=True)
    model = opVideoLinearCFGGuidance.patch(model_raw, guidance_min_cfg)[0]
    init_image = numpy_to_pytorch(input_image)
    positive, negative, latent_image = opSVD_img2vid_Conditioning.encode(
        clip_vision, init_image, vae, width, height, video_frames, motion_bucket_id, fps, augmentation_level)
    output_latent = opKSampler.sample(model, sampling_seed, sampling_steps, sampling_cfg,
                                      sampling_sampler_name, sampling_scheduler, positive,
                                      negative, latent_image, sampling_denoise)[0]
    output_pixels = opVAEDecode.decode(vae, output_latent)[0]
    outputs = pytorch_to_numpy(output_pixels)
    return outputs


def on_ui_tabs():
    with gr.Blocks() as svd_block:
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label='Input Image', source='upload', type='numpy', height=400)

                with gr.Row():
                    filename = gr.Dropdown(label="SVD Checkpoint Filename",
                                           choices=svd_filenames,
                                           value=svd_filenames[0] if len(svd_filenames) > 0 else None)
                    refresh_button = ToolButton(value=refresh_symbol, tooltip="Refresh")
                    refresh_button.click(
                        fn=lambda: gr.update(choices=update_svd_filenames),
                        inputs=[], outputs=filename)

                width = gr.Slider(label='Width', minimum=16, maximum=8192, step=8, value=1024)
                height = gr.Slider(label='Height', minimum=16, maximum=8192, step=8, value=576)
                video_frames = gr.Slider(label='Video Frames', minimum=1, maximum=4096, step=1, value=14)
                motion_bucket_id = gr.Slider(label='Motion Bucket Id', minimum=1, maximum=1023, step=1, value=127)
                fps = gr.Slider(label='Fps', minimum=1, maximum=1024, step=1, value=6)
                augmentation_level = gr.Slider(label='Augmentation Level', minimum=0.0, maximum=10.0, step=0.01,
                                               value=0.0)
                sampling_steps = gr.Slider(label='Sampling Steps', minimum=1, maximum=200, step=1, value=20)
                sampling_cfg = gr.Slider(label='CFG Scale', minimum=0.0, maximum=50.0, step=0.1, value=2.5)
                sampling_denoise = gr.Slider(label='Sampling Denoise', minimum=0.0, maximum=1.0, step=0.01, value=1.0)
                guidance_min_cfg = gr.Slider(label='Guidance Min Cfg', minimum=0.0, maximum=100.0, step=0.5, value=1.0)
                sampling_sampler_name = gr.Radio(label='Sampler Name',
                                                 choices=['euler', 'euler_ancestral', 'heun', 'heunpp2', 'dpm_2',
                                                          'dpm_2_ancestral', 'lms', 'dpm_fast', 'dpm_adaptive',
                                                          'dpmpp_2s_ancestral', 'dpmpp_sde', 'dpmpp_sde_gpu',
                                                          'dpmpp_2m', 'dpmpp_2m_sde', 'dpmpp_2m_sde_gpu',
                                                          'dpmpp_3m_sde', 'dpmpp_3m_sde_gpu', 'ddpm', 'lcm', 'ddim',
                                                          'uni_pc', 'uni_pc_bh2'], value='euler')
                sampling_scheduler = gr.Radio(label='Scheduler',
                                              choices=['normal', 'karras', 'exponential', 'sgm_uniform', 'simple',
                                                       'ddim_uniform'], value='karras')
                sampling_seed = gr.Number(label='Seed', value=12345, precision=0)

                generate_button = gr.Button(value="Generate")

                ctrls = [filename, width, height, video_frames, motion_bucket_id, fps, augmentation_level,
                         sampling_seed, sampling_steps, sampling_cfg, sampling_sampler_name, sampling_scheduler,
                         sampling_denoise, guidance_min_cfg, input_image]

            with gr.Column():
                output_gallery = gr.Gallery(label='Gallery', show_label=False, object_fit='contain',
                                            visible=True, height=1024, columns=4)

        generate_button.click(predict, inputs=ctrls, outputs=[output_gallery])
    return [(svd_block, "SVD", "svd")]


update_svd_filenames()
script_callbacks.on_ui_tabs(on_ui_tabs)
```

Note that although the above codes look like independent codes, they actually will automatically offload/unload any other models. For example, below is me opening webui, load SDXL, generated an image, then go to SVD, then generated image frames. You can see that the GPU memory is perfectly managed and the SDXL is moved to RAM then SVD is moved to GPU. 

Note that this management is fully automatic. This makes writing extensions super simple.

![image](https://github.com/lllyasviel/stable-diffusion-webui-forge/assets/19834515/de1a2d05-344a-44d7-bab8-9ecc0a58a8d3)

![image](https://github.com/lllyasviel/stable-diffusion-webui-forge/assets/19834515/14bcefcf-599f-42c3-bce9-3fd5e428dd91)

Similarly, Zero123:

![image](https://github.com/lllyasviel/stable-diffusion-webui-forge/assets/19834515/7685019c-7239-47fb-9cb5-2b7b33943285)

### Write a simple ControlNet:

Below is a simple extension to have a completely independent pass of ControlNet that never conflicts any other extensions:

`extensions-builtin/sd_forge_controlnet_example/scripts/sd_forge_controlnet_example.py`

Note that this extension is hidden because it is only for developers. To see it in UI, use `--show-controlnet-example`.

The memory optimization in this example is fully automatic. You do not need to care about memory and inference speed, but you may want to cache objects if you wish.

```python
# Use --show-controlnet-example to see this extension.

import cv2
import gradio as gr
import torch

from modules import scripts
from modules.shared_cmd_options import cmd_opts
from modules_forge.shared import supported_preprocessors
from modules.modelloader import load_file_from_url
from ldm_patched.modules.controlnet import load_controlnet
from modules_forge.controlnet import apply_controlnet_advanced
from modules_forge.forge_util import numpy_to_pytorch
from modules_forge.shared import controlnet_dir


class ControlNetExampleForge(scripts.Script):
    model = None

    def title(self):
        return "ControlNet Example for Developers"

    def show(self, is_img2img):
        # make this extension visible in both txt2img and img2img tab.
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            gr.HTML('This is an example controlnet extension for developers.')
            gr.HTML('You see this extension because you used --show-controlnet-example')
            input_image = gr.Image(source='upload', type='numpy')
            funny_slider = gr.Slider(label='This slider does nothing. It just shows you how to transfer parameters.',
                                     minimum=0.0, maximum=1.0, value=0.5)

        return input_image, funny_slider

    def process(self, p, *script_args, **kwargs):
        input_image, funny_slider = script_args

        # This slider does nothing. It just shows you how to transfer parameters.
        del funny_slider

        if input_image is None:
            return

        # controlnet_canny_path = load_file_from_url(
        #     url='https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/sai_xl_canny_256lora.safetensors',
        #     model_dir=model_dir,
        #     file_name='sai_xl_canny_256lora.safetensors'
        # )
        controlnet_canny_path = load_file_from_url(
            url='https://huggingface.co/lllyasviel/fav_models/resolve/main/fav/control_v11p_sd15_canny_fp16.safetensors',
            model_dir=controlnet_dir,
            file_name='control_v11p_sd15_canny_fp16.safetensors'
        )
        print('The model [control_v11p_sd15_canny_fp16.safetensors] download finished.')

        self.model = load_controlnet(controlnet_canny_path)
        print('Controlnet loaded.')

        return

    def process_before_every_sampling(self, p, *script_args, **kwargs):
        # This will be called before every sampling.
        # If you use highres fix, this will be called twice.

        input_image, funny_slider = script_args

        if input_image is None or self.model is None:
            return

        B, C, H, W = kwargs['noise'].shape  # latent_shape
        height = H * 8
        width = W * 8
        batch_size = p.batch_size

        preprocessor = supported_preprocessors['canny']

        # detect control at certain resolution
        control_image = preprocessor(
            input_image, resolution=512, slider_1=100, slider_2=200, slider_3=None)

        # here we just use nearest neighbour to align input shape.
        # You may want crop and resize, or crop and fill, or others.
        control_image = cv2.resize(
            control_image, (width, height), interpolation=cv2.INTER_NEAREST)

        # Output preprocessor result. Now called every sampling. Cache in your own way.
        p.extra_result_images.append(control_image)

        print('Preprocessor Canny finished.')

        control_image_bchw = numpy_to_pytorch(control_image).movedim(-1, 1)

        unet = p.sd_model.forge_objects.unet

        # Unet has input, middle, output blocks, and we can give different weights
        # to each layers in all blocks.
        # Below is an example for stronger control in middle block.
        # This is helpful for some high-res fix passes. (p.is_hr_pass)
        positive_advanced_weighting = {
            'input': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
            'middle': [1.0],
            'output': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
        }
        negative_advanced_weighting = {
            'input': [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05, 1.15, 1.25],
            'middle': [1.05],
            'output': [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05, 1.15, 1.25]
        }

        # The advanced_frame_weighting is a weight applied to each image in a batch.
        # The length of this list must be same with batch size
        # For example, if batch size is 5, the below list is [0.2, 0.4, 0.6, 0.8, 1.0]
        # If you view the 5 images as 5 frames in a video, this will lead to
        # progressively stronger control over time.
        advanced_frame_weighting = [float(i + 1) / float(batch_size) for i in range(batch_size)]

        # The advanced_sigma_weighting allows you to dynamically compute control
        # weights given diffusion timestep (sigma).
        # For example below code can softly make beginning steps stronger than ending steps.
        sigma_max = unet.model.model_sampling.sigma_max
        sigma_min = unet.model.model_sampling.sigma_min
        advanced_sigma_weighting = lambda s: (s - sigma_min) / (sigma_max - sigma_min)

        # You can even input a tensor to mask all control injections
        # The mask will be automatically resized during inference in UNet.
        # The size should be B 1 H W and the H and W are not important
        # because they will be resized automatically
        advanced_mask_weighting = torch.ones(size=(1, 1, 512, 512))

        # But in this simple example we do not use them
        positive_advanced_weighting = None
        negative_advanced_weighting = None
        advanced_frame_weighting = None
        advanced_sigma_weighting = None
        advanced_mask_weighting = None

        unet = apply_controlnet_advanced(unet=unet, controlnet=self.model, image_bchw=control_image_bchw,
                                         strength=0.6, start_percent=0.0, end_percent=0.8,
                                         positive_advanced_weighting=positive_advanced_weighting,
                                         negative_advanced_weighting=negative_advanced_weighting,
                                         advanced_frame_weighting=advanced_frame_weighting,
                                         advanced_sigma_weighting=advanced_sigma_weighting,
                                         advanced_mask_weighting=advanced_mask_weighting)

        p.sd_model.forge_objects.unet = unet

        # Below codes will add some logs to the texts below the image outputs on UI.
        # The extra_generation_params does not influence results.
        p.extra_generation_params.update(dict(
            controlnet_info='You should see these texts below output images!',
        ))

        return


# Use --show-controlnet-example to see this extension.
if not cmd_opts.show_controlnet_example:
    del ControlNetExampleForge

```

![image](https://github.com/lllyasviel/stable-diffusion-webui-forge/assets/19834515/822fa2fc-c9f4-4f58-8669-4b6680b91063)


### Add a preprocessor

Below is the full codes to add a normalbae preprocessor with perfect memory managements.

You can use arbitrary independent extensions to add a preprocessor.

Your preprocessor will be read by all other extensions using `modules_forge.shared.preprocessors`

Below codes are in `extensions-builtin\forge_preprocessor_normalbae\scripts\preprocessor_normalbae.py`

```python
from modules_forge.supported_preprocessor import Preprocessor, PreprocessorParameter
from modules_forge.shared import preprocessor_dir, add_supported_preprocessor
from modules_forge.forge_util import resize_image_with_pad
from modules.modelloader import load_file_from_url

import types
import torch
import numpy as np

from einops import rearrange
from annotator.normalbae.models.NNET import NNET
from annotator.normalbae import load_checkpoint
from torchvision import transforms


class PreprocessorNormalBae(Preprocessor):
    def __init__(self):
        super().__init__()
        self.name = 'normalbae'
        self.tags = ['NormalMap']
        self.model_filename_filters = ['normal']
        self.slider_resolution = PreprocessorParameter(
            label='Resolution', minimum=128, maximum=2048, value=512, step=8, visible=True)
        self.slider_1 = PreprocessorParameter(visible=False)
        self.slider_2 = PreprocessorParameter(visible=False)
        self.slider_3 = PreprocessorParameter(visible=False)
        self.show_control_mode = True
        self.do_not_need_model = False
        self.sorting_priority = 100  # higher goes to top in the list

    def load_model(self):
        if self.model_patcher is not None:
            return

        model_path = load_file_from_url(
            "https://huggingface.co/lllyasviel/Annotators/resolve/main/scannet.pt",
            model_dir=preprocessor_dir)

        args = types.SimpleNamespace()
        args.mode = 'client'
        args.architecture = 'BN'
        args.pretrained = 'scannet'
        args.sampling_ratio = 0.4
        args.importance_ratio = 0.7
        model = NNET(args)
        model = load_checkpoint(model_path, model)
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.model_patcher = self.setup_model_patcher(model)

    def __call__(self, input_image, resolution, slider_1=None, slider_2=None, slider_3=None, **kwargs):
        input_image, remove_pad = resize_image_with_pad(input_image, resolution)

        self.load_model()

        self.move_all_model_patchers_to_gpu()

        assert input_image.ndim == 3
        image_normal = input_image

        with torch.no_grad():
            image_normal = self.send_tensor_to_model_device(torch.from_numpy(image_normal))
            image_normal = image_normal / 255.0
            image_normal = rearrange(image_normal, 'h w c -> 1 c h w')
            image_normal = self.norm(image_normal)

            normal = self.model_patcher.model(image_normal)
            normal = normal[0][-1][:, :3]
            normal = ((normal + 1) * 0.5).clip(0, 1)

            normal = rearrange(normal[0], 'c h w -> h w c').cpu().numpy()
            normal_image = (normal * 255.0).clip(0, 255).astype(np.uint8)

        return remove_pad(normal_image)


add_supported_preprocessor(PreprocessorNormalBae())

```

# New features (that are not available in original WebUI)

Thanks to Unet Patcher, many new things are possible now and supported in Forge/reForge, including SVD, Z123, masked Ip-adapter, masked controlnet, photomaker, etc.

Masked Ip-Adapter

![image](https://github.com/lllyasviel/stable-diffusion-webui-forge/assets/19834515/d26630f9-922d-4483-8bf9-f364dca5fd50)

![image](https://github.com/lllyasviel/stable-diffusion-webui-forge/assets/19834515/03580ef7-235c-4b03-9ca6-a27677a5a175)

![image](https://github.com/lllyasviel/stable-diffusion-webui-forge/assets/19834515/d9ed4a01-70d4-45b4-a6a7-2f765f158fae)

Masked ControlNet

![image](https://github.com/lllyasviel/stable-diffusion-webui-forge/assets/19834515/872d4785-60e4-4431-85c7-665c781dddaa)

![image](https://github.com/lllyasviel/stable-diffusion-webui-forge/assets/19834515/335a3b33-1ef8-46ff-a462-9f1b4f2c49fc)

![image](https://github.com/lllyasviel/stable-diffusion-webui-forge/assets/19834515/b3684a15-8895-414e-8188-487269dfcada)

PhotoMaker

(Note that photomaker is a special control that need you to add the trigger word "photomaker". Your prompt should be like "a photo of photomaker")

![image](https://github.com/lllyasviel/stable-diffusion-webui-forge/assets/19834515/07b0b626-05b5-473b-9d69-3657624d59be)

Marigold Depth

![image](https://github.com/lllyasviel/stable-diffusion-webui-forge/assets/19834515/bdf54148-892d-410d-8ed9-70b4b121b6e7)

# New Sampler (that is not in origin)

    DDPM

# Others samplers may be available, but after the schedulers merge, they shouldn't be needed.

# About Extensions

ControlNet and TiledVAE are integrated, and you should uninstall these two extensions:

    sd-webui-controlnet
    multidiffusion-upscaler-for-automatic1111

Note that **AnimateDiff** is under construction by [continue-revolution](https://github.com/continue-revolution) at [sd-webui-animatediff forge/master branch](https://github.com/continue-revolution/sd-webui-animatediff/tree/forge/master) and [sd-forge-animatediff](https://github.com/continue-revolution/sd-forge-animatediff) (they are in sync). (continue-revolution original words: prompt travel, inf t2v, controlnet v2v have been proven to work well; motion lora, i2i batch still under construction and may be finished in a week")

Other extensions should work without problems, like:

    canvas-zoom
    translations/localizations
    Dynamic Prompts
    Adetailer
    Ultimate SD Upscale
    Reactor

However, if newer extensions use Forge/reForge, their codes can be much shorter. 

Usually if an old extension rework using Forge/reForge's unet patcher, 80% codes can be removed, especially when they need to call controlnet.

# Support

Some people have been asking how to donate or support the project, and I'm really grateful for that! I did this buymeacoffe link from some suggestions!

[!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/Panchovix)
