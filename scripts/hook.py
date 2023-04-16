# hook function inspired by https://github.com/Mikubill/sd-webui-controlnet/blob/main/scripts/hook.py
import torch
import torch.nn as nn
from modules import devices, lowvram, shared, scripts

cond_cast_unet = getattr(devices, 'cond_cast_unet', lambda x: x)

from ldm.modules.diffusionmodules.util import timestep_embedding
from ldm.modules.diffusionmodules.openaimodel import UNetModel


class TorchHijackForUnet:
    """
    This is torch, but with cat that resizes tensors to appropriate dimensions if they do not match;
    this makes it possible to create pictures with dimensions that are multiples of 8 rather than 64
    """

    def __getattr__(self, item):
        if item == 'cat':
            return self.cat

        if hasattr(torch, item):
            return getattr(torch, item)

        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, item))

    def cat(self, tensors, *args, **kwargs):
        if len(tensors) == 2:
            a, b = tensors
            if a.shape[-2:] != b.shape[-2:]:
                a = torch.nn.functional.interpolate(a, b.shape[-2:], mode="nearest")

            tensors = (a, b)

        return torch.cat(tensors, *args, **kwargs)

th = TorchHijackForUnet()

class ControlParams:
    def __init__(self, cond_tau, enabled, style_features, adapter_features, guidance_stopped):
        self.cond_tau = cond_tau
        self.enabled = enabled
        self.style_features = style_features
        self.adapter_features = adapter_features
        self.guidance_stopped = guidance_stopped


class UnetHook(nn.Module):
    def __init__(self, lowvram=False) -> None:
        super().__init__()
        self.lowvram = lowvram
        self.batch_cond_available = True
        self.only_mid_control = shared.opts.data.get("control_net_only_mid_control", False)
        
    def hook(self, model):
        outer = self
        
        def guidance_schedule_handler(x):
            current_sampling_percent = (x.sampling_step / x.total_sampling_steps)
            self.control_params.guidance_stopped = (current_sampling_percent < self.control_params.cond_tau) and (self.control_params.enabled)
   
        def cfg_based_adder(base, x, require_autocast, is_adapter=False):
            if isinstance(x, float):
                return base + x
            
            if require_autocast:
                zeros = torch.zeros_like(base)
                zeros[:, :x.shape[1], ...] = x
                x = zeros
                
            if base.shape[0] % 2 == 0 and (self.guess_mode or shared.opts.data.get("control_net_cfg_based_guidance", False)):
                if self.is_vanilla_samplers:  
                    uncond, cond = base.chunk(2)
                    if x.shape[0] % 2 == 0:
                        _, x_cond = x.chunk(2)
                        return torch.cat([uncond, cond + x_cond], dim=0)
                    if is_adapter:
                        return torch.cat([uncond, cond + x], dim=0)
                else:
                    cond, uncond = base.chunk(2)
                    if x.shape[0] % 2 == 0:
                        x_cond, _ = x.chunk(2)
                        return torch.cat([cond + x_cond, uncond], dim=0)
                    if is_adapter:
                        return torch.cat([cond + x, uncond], dim=0)
            
            # resize to sample resolution
            base_h, base_w = base.shape[-2:]
            xh, xw = x.shape[-2:]
            if base_h != xh or base_w != xw:
                x = th.nn.functional.interpolate(x, size=(base_h, base_w), mode="nearest")
            
            return base + x

        def forward(self, x, timesteps=None, context=None, **kwargs):
            total_extra_cond = outer.control_params.style_features
                
            # check if it's non-batch-cond mode (lowvram, edit model etc)
            if context.shape[0] % 2 != 0 and outer.batch_cond_available:
                outer.batch_cond_available = False
                if (total_extra_cond is not None) or outer.guess_mode or shared.opts.data.get("control_net_cfg_based_guidance", False):
                    print("Warning: StyleAdapter and cfg/guess mode may not works due to non-batch-cond inference")
                
            # concat styleadapter to cond, pad uncond to same length
            if (total_extra_cond is not None) and outer.batch_cond_available:
                total_extra_cond = torch.repeat_interleave(total_extra_cond, context.shape[0] // 2, dim=0)
                if outer.is_vanilla_samplers:  
                    uncond, cond = context.chunk(2)
                    cond = torch.cat([cond, total_extra_cond], dim=1)
                    uncond = torch.cat([uncond, uncond[:, -total_extra_cond.shape[1]:, :]], dim=1)
                    context = torch.cat([uncond, cond], dim=0)
                else:
                    cond, uncond = context.chunk(2)
                    cond = torch.cat([cond, total_extra_cond], dim=1)
                    uncond = torch.cat([uncond, uncond[:, -total_extra_cond.shape[1]:, :]], dim=1)
                    context = torch.cat([cond, uncond], dim=0)
                        
            assert timesteps is not None, ValueError(f"insufficient timestep: {timesteps}")
            hs = []
            with th.no_grad():
                if not outer.control_params.guidance_stopped:
                    outer.control_params.adapter_features = None
                t_emb = cond_cast_unet(timestep_embedding(timesteps, self.model_channels, repeat_only=False))
                emb = self.time_embed(t_emb) 
                h = x.type(self.dtype)
                adapter_idx = 0
                for id, module in enumerate(self.input_blocks):
                    h = module(h, emb, context)
                    if ((id+1)%3 == 0) and outer.control_params.adapter_features is not None:
                        h = h + outer.control_params.adapter_features[adapter_idx]
                        adapter_idx += 1
                    hs.append(h)
                if outer.control_params.adapter_features is not None:
                    assert len(outer.control_params.adapter_features)==adapter_idx, 'Wrong adapter_features'

                h = self.middle_block(h, emb, context)
                for module in self.output_blocks:
                    h = th.cat([h, hs.pop()], dim=1)
                    h = module(h, emb, context)
                h = h.type(x.dtype)
            
            return self.out(h)

        def forward2(*args, **kwargs):
            if shared.cmd_opts.lowvram:
                lowvram.send_everything_to_cpu()
                                        
            return forward(*args, **kwargs)
               
        model._original_forward = model.forward
        model.forward = forward2.__get__(model, UNetModel)
        scripts.script_callbacks.on_cfg_denoiser(guidance_schedule_handler)
    
    def notify(self, params, is_vanilla_samplers): 
        self.is_vanilla_samplers = is_vanilla_samplers
        self.control_params = params

    def restore(self, model):
        scripts.script_callbacks.remove_current_script_callbacks()
        if hasattr(self, "control_params"):
            del self.control_params
        
        if not hasattr(model, "_original_forward"):
            # no such handle, ignore
            return
        
        model.forward = model._original_forward
        del model._original_forward