import modules.scripts as scripts
import gradio as gr
import os
import copy
from modules import images, devices, ui
from modules.processing import process_images, Processed
from modules.processing import Processed
from modules.shared import opts, cmd_opts, state
from functools import partial
from itertools import chain
import argparse
from scripts.adapter.inference_base import get_adapters
from scripts.adapter.modules.extra_condition.api import ExtraCondition, get_cond_model
from scripts.adapter.modules.extra_condition import api
from scripts.adapter.modules.encoders.adapter import CoAdapterFuser
from scripts.hook import UnetHook, ControlParams
import torch
from modules import scripts
import cv2
from basicsr.utils import tensor2img
import gc
from huggingface_hub import hf_hub_url
import subprocess
import shlex
from scripts.adapter.util import get_hw

urls = {
    'TencentARC/T2I-Adapter':[
        'third-party-models/body_pose_model.pth', 'third-party-models/table5_pidinet.pth',
        'models/coadapter-canny-sd15v1.pth',
        'models/coadapter-color-sd15v1.pth',
        'models/coadapter-sketch-sd15v1.pth',
        'models/coadapter-style-sd15v1.pth',
        'models/coadapter-depth-sd15v1.pth',
        'models/coadapter-fuser-sd15v1.pth',

    ],
    'runwayml/stable-diffusion-v1-5': ['v1-5-pruned-emaonly.ckpt'],
    'andite/anything-v4.0': ['anything-v4.5-pruned.ckpt', 'anything-v4.0.vae.pt'],
}

if os.path.exists('models/T2I-Adapter') == False:
    os.mkdir('models/T2I-Adapter')
for repo in urls:
    files = urls[repo]
    for file in files:
        url = hf_hub_url(repo, file)
        name_ckp = url.split('/')[-1]
        save_path = os.path.join('models/T2I-Adapter',name_ckp)
        if os.path.exists(save_path) == False:
            subprocess.run(shlex.split(f'wget {url} -O {save_path}'))

DEFAULT_NEGATIVE_PROMPT = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, ' \
                          'fewer digits, cropped, worst quality, low quality'

supported_cond = ['style', 'color', 'sketch', 'depth', 'canny']

# config
class Params:
    def __init__(self):
        self.sd_ckpt = 'models/T2I-Adapter/v1-5-pruned-emaonly.ckpt'
        self.vae_ckpt = None
global_opt = Params()
global_opt.config = os.path.join(scripts.basedir(),'configs/stable-diffusion/sd-v1-inference.yaml')
for cond_name in supported_cond:
    setattr(global_opt, f'{cond_name}_adapter_ckpt', f'models/T2I-Adapter/coadapter-{cond_name}-sd15v1.pth')
global_opt.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
global_opt.max_resolution = 512 * 512
global_opt.resize_short_edge = None
global_opt.sampler = 'ddim'
global_opt.cond_weight = 1.0
global_opt.C = 4
global_opt.f = 8
#TODO: expose style_cond_tau to users
global_opt.style_cond_tau = 1.0

def change_visible(im1, im2, val):
    outputs = {}
    if val == "Image":
        outputs[im1] = gr.update(visible=True)
        outputs[im2] = gr.update(visible=False)
    elif val == "Nothing":
        outputs[im1] = gr.update(visible=False)
        outputs[im2] = gr.update(visible=False)
    else:
        outputs[im1] = gr.update(visible=False)
        outputs[im2] = gr.update(visible=True)
    return outputs

class Script(scripts.Script):  
    def __init__(self):
        super().__init__()
        self.adapters = {}
        self.cond_models = {}
        self.coadapter_fuser = CoAdapterFuser(unet_channels=[320, 640, 1280, 1280], width=768, num_head=8, n_layes=3)
        self.coadapter_fuser.load_state_dict(torch.load(f'models/T2I-Adapter/coadapter-fuser-sd15v1.pth'))
        self.coadapter_fuser = self.coadapter_fuser.to(devices.get_device_for('T2I-Adapter'))
        self.network_cur = None

# The title of the script. This is what will be displayed in the dropdown menu.
    def title(self):

        return "CoAdapter"


# Determines when the script should be shown in the dropdown menu via the 
# returned value. As an example:
# is_img2img is True if the current tab is img2img, and False if it is txt2img.
# Thus, return is_img2img to only show the script on the img2img tab.

    def show(self, is_img2img):

        return scripts.AlwaysVisible #is_img2img

# How the script's is displayed in the UI. See https://gradio.app/docs/#components
# for the different UI components you can use and how to create them.
# Most UI components can return a value, such as a boolean for a checkbox.
# The returned values are passed to the run method as parameters.

    def ui(self, is_img2img):
        btns = []
        ims1 = []
        ims2 = []
        cond_weights = []
        with gr.Group():
            with gr.Accordion("CoAdapter", open=False):
                enabled = gr.Checkbox(label='Enable (Light it up if you want to use this function.)', value=False)
                resize_mode = gr.Radio(choices=['Consistent with the condition map', 'Controlled by resize sliders'], value='Consistent with the condition map', label="Resize Mode")
                with gr.Row():
                    for cond_name in supported_cond:
                        with gr.Box():
                            with gr.Column():
                                if cond_name == 'style':
                                    btn1 = gr.Radio(
                                    choices=["Image", "Nothing"],
                                    label=f"Input type for {cond_name}",
                                    interactive=True,
                                    value="Nothing",
                                )
                                else:
                                    btn1 = gr.Radio(
                                        choices=["Image", cond_name, "Nothing"],
                                        label=f"Input type for {cond_name}",
                                        interactive=True,
                                        value="Nothing",
                                    )
                                im1 = gr.Image(source='upload', label="Image", interactive=True, visible=False, type="numpy")
                                im2 = gr.Image(source='upload', label=cond_name, interactive=True, visible=False, type="numpy")
                                cond_weight = gr.Slider(
                                    label="Condition weight", minimum=0, maximum=5, step=0.05, value=1, interactive=True)

                                fn = partial(change_visible, im1, im2)
                                btn1.change(fn=fn, inputs=[btn1], outputs=[im1, im2], queue=False)

                                btns.append(btn1)
                                ims1.append(im1)
                                ims2.append(im2)
                                cond_weights.append(cond_weight)

                with gr.Column():
                    cond_tau = gr.Slider(
                        label="timestamp parameter that determines until which step the adapter is applied",
                        value=1.0,
                        minimum=0.1,
                        maximum=1.0,
                        step=0.05)

                inps = list(chain(btns, ims1, ims2, cond_weights))
                inps.extend([cond_tau, enabled, resize_mode])

        return inps

    def process(self, p, *args):
        unet = p.sd_model.model.diffusion_model
        if self.network_cur is not None:
            self.network_cur.restore(unet)

        inps = []
        for i in range(0, len(args) - 3, len(supported_cond)):
            inps.append(args[i:i + len(supported_cond)])

        opt = copy.deepcopy(global_opt)
        opt.cond_tau = args[-3]
        opt.enabled = args[-2]
        opt.resize_mode = args[-1]
        if len(inps) == 0 or opt.enabled==False:
           self.network_cur = None
           return
        h, w, bsz = p.height, p.width, p.batch_size

        ims1 = []
        ims2 = []

        # resize all the images to the same size
        if opt.resize_mode == 'Consistent with the condition map':
            for idx, (b, im1, im2, cond_weight) in enumerate(zip(*inps)):
                if idx > 0:
                    if b != 'Nothing' and (im1 is not None or im2 is not None):
                        if im1 is not None:
                            h, w, _ = im1.shape
                        else:
                            h, w, _ = im2.shape
            h, w = get_hw(h, w, max_resolution=opt.max_resolution, resize_short_edge=opt.resize_short_edge)
            p.height, p.width = h, w
        # else:

        for idx, (b, im1, im2, cond_weight) in enumerate(zip(*inps)):
            if b != 'Nothing':
                if im1 is not None:
                    im1 = cv2.resize(im1, (w, h), interpolation=cv2.INTER_CUBIC)
                if im2 is not None:
                    im2 = cv2.resize(im2, (w, h), interpolation=cv2.INTER_CUBIC)
            ims1.append(im1)
            ims2.append(im2)

        conds = []
        activated_conds = []
        for idx, (b, im1, im2, cond_weight) in enumerate(zip(*inps)):
            cond_name = supported_cond[idx]
            if b == 'Nothing':
                if cond_name in self.adapters: # save gpu memory
                    self.adapters[cond_name]['model'] = self.adapters[cond_name]['model'].cpu()
            else:
                activated_conds.append(cond_name)
                if cond_name in self.adapters:
                    self.adapters[cond_name]['model'] = self.adapters[cond_name]['model'].to(devices.get_device_for('T2I-Adapter'))
                else:
                    self.adapters[cond_name] = get_adapters(opt, getattr(ExtraCondition, cond_name))
                self.adapters[cond_name]['cond_weight'] = cond_weight

                process_cond_module = getattr(api, f'get_cond_{cond_name}')

                if b == 'Image':
                    if cond_name not in self.cond_models:
                        self.cond_models[cond_name] = get_cond_model(opt, getattr(ExtraCondition, cond_name))
                    conds.append(process_cond_module(opt, ims1[idx], 'image', self.cond_models[cond_name]))
                else:
                    conds.append(process_cond_module(opt, ims2[idx], cond_name, None))

        features = dict()
        for idx, cond_name in enumerate(activated_conds):
            cur_feats = self.adapters[cond_name]['model'](conds[idx])
            if isinstance(cur_feats, list):
                for i in range(len(cur_feats)):
                    cur_feats[i] *= self.adapters[cond_name]['cond_weight']
            else:
                cur_feats *= self.adapters[cond_name]['cond_weight']
            features[cond_name] = cur_feats

        adapter_features, append_to_context = self.coadapter_fuser(features)
        self.output_conds = []
        for cond in conds:
            self.output_conds.append(tensor2img(cond, rgb2bgr=False))
        forward_params = ControlParams(
            cond_tau = opt.cond_tau,
            enabled = opt.enabled,
            style_features = append_to_context, 
            adapter_features = adapter_features,
            guidance_stopped=False,
        )

        self.network_cur = UnetHook()    
        self.network_cur.hook(unet)
        self.network_cur.notify(forward_params, p.sampler_name in ["DDIM", "PLMS", "UniPC"])

    def postprocess(self, p, processed, *args):
        if self.network_cur is None:
            return

        for detect_map in self.output_conds:
            processed.images.extend([detect_map])
        self.input_image = None
        self.network_cur.restore(p.sd_model.model.diffusion_model)
        self.network_cur = None

        gc.collect()
        devices.torch_gc()

