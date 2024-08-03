from modules import sd_samplers_kdiffusion, sd_samplers_common
from backend.modules import k_diffusion_extra


class AlterSampler(sd_samplers_kdiffusion.KDiffusionSampler):
    def __init__(self, sd_model, sampler_name):
        self.sampler_name = sampler_name
        self.unet = sd_model.forge_objects.unet
        sampler_function = getattr(k_diffusion_extra, "sample_{}".format(sampler_name))
        super().__init__(sampler_function, sd_model, None)


def build_constructor(sampler_name):
    def constructor(m):
        return AlterSampler(m, sampler_name)
    
    return constructor


samplers_data_alter = [
    sd_samplers_common.SamplerData('Euler Comfy', build_constructor(sampler_name='euler'), ['euler'], {}),
    sd_samplers_common.SamplerData('Euler A Comfy', build_constructor(sampler_name='euler_ancestral'), ['euler_ancestral'], {}),
    sd_samplers_common.SamplerData('Heun Comfy', build_constructor(sampler_name='heun'), ['heun'], {}),
    sd_samplers_common.SamplerData('DPM++ 2S Ancestral Comfy', build_constructor(sampler_name='dpmpp_2s_ancestral'), ['dpmpp_2s_ancestral'], {}),
    sd_samplers_common.SamplerData('DPM++ SDE Comfy', build_constructor(sampler_name='dpmpp_sde'), ['dpmpp_sde'], {}),
    sd_samplers_common.SamplerData('DPM++ 2M Comfy', build_constructor(sampler_name='dpmpp_2m'), ['dpmpp_2m'], {}),
    sd_samplers_common.SamplerData('DPM++ 2M SDE Comfy', build_constructor(sampler_name='dpmpp_2m_sde'), ['dpmpp_2m_sde'], {}),
    sd_samplers_common.SamplerData('DPM++ 3M SDE Comfy', build_constructor(sampler_name='dpmpp_3m_sde'), ['dpmpp_3m_sde'], {}),
    # sd_samplers_common.SamplerData('Euler A Turbo', build_constructor(sampler_name='euler_ancestral_turbo'), ['euler_ancestral_turbo'], {}),
    # sd_samplers_common.SamplerData('DPM++ 2M Turbo', build_constructor(sampler_name='dpmpp_2m_turbo'), ['dpmpp_2m_turbo'], {}),
    # sd_samplers_common.SamplerData('DPM++ 2M SDE Turbo', build_constructor(sampler_name='dpmpp_2m_sde_turbo'), ['dpmpp_2m_sde_turbo'], {}),
    sd_samplers_common.SamplerData('DDPM', build_constructor(sampler_name='ddpm'), ['ddpm'], {}),
    sd_samplers_common.SamplerData('HeunPP2', build_constructor(sampler_name='heunpp2'), ['heunpp2'], {}),
    sd_samplers_common.SamplerData('IPNDM', build_constructor(sampler_name='ipndm'), ['ipndm'], {}),
    sd_samplers_common.SamplerData('IPNDM_V', build_constructor(sampler_name='ipndm_v'), ['ipndm_v'], {}),
    # sd_samplers_common.SamplerData('DEIS', build_constructor(sampler_name='deis'), ['deis'], {}),
    # sd_samplers_common.SamplerData('Euler CFG++', build_constructor(sampler_name='euler_cfg_pp'), ['euler_cfg_pp'], {}),
    # sd_samplers_common.SamplerData('Euler Ancestral CFG++', build_constructor(sampler_name='euler_ancestral_cfg_pp'), ['euler_ancestral_cfg_pp'], {}),
    # sd_samplers_common.SamplerData('DPM++ 2S Ancestral CFG++', build_constructor(sampler_name='dpmpp_2s_ancestral_cfg_pp'), ['dpmpp_2s_ancestral_cfg_pp'], {}),
    # sd_samplers_common.SamplerData('DPM++ SDE CFG++', build_constructor(sampler_name='dpmpp_sde_cfg_pp'), ['dpmpp_sde_cfg_pp'], {}),
    # sd_samplers_common.SamplerData('DPM++ 2M CFG++', build_constructor(sampler_name='dpmpp_2m_cfg_pp'), ['dpmpp_2m_cfg_pp'], {}),
    # sd_samplers_common.SamplerData('ODE (Bosh3)', build_constructor(sampler_name='ode_bosh3'), ['ode_bosh3'], {}),
    # sd_samplers_common.SamplerData('ODE (Fehlberg2)', build_constructor(sampler_name='ode_fehlberg2'), ['ode_fehlberg2'], {}),
    # sd_samplers_common.SamplerData('ODE (Adaptive Heun)', build_constructor(sampler_name='ode_adaptive_heun'), ['ode_adaptive_heun'], {}),
    # sd_samplers_common.SamplerData('ODE (Dopri5)', build_constructor(sampler_name='ode_dopri5'), ['ode_dopri5'], {}),
    # sd_samplers_common.SamplerData('ODE Custom', build_constructor(sampler_name='ode_custom'), ['ode_custom'], {}),
]
