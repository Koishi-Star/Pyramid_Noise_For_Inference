from . import pyramid_noise_sampling
from .pyramid_noise_sampling import sample_euler_pyramid, sample_heun_pyramid, sample_dpmpp_2s_pyramid

if pyramid_noise_sampling.BACKEND == "ComfyUI":
    if not pyramid_noise_sampling.INITIALIZED:
        from comfy.k_diffusion import sampling as k_diffusion_sampling
        from comfy.samplers import SAMPLER_NAMES

        setattr(k_diffusion_sampling, "sample_euler_pyramid", sample_euler_pyramid)
        setattr(k_diffusion_sampling, "sample_heun_pyramid", sample_heun_pyramid)
        setattr(k_diffusion_sampling, "sample_dpmpp_2s_pyramid", sample_dpmpp_2s_pyramid)

        SAMPLER_NAMES.append("euler_pyramid")
        SAMPLER_NAMES.append("heun_pyramid")
        SAMPLER_NAMES.append("dpmpp_2s_pyramid")

        pyramid_noise_sampling.INITIALIZED = True

NODE_CLASS_MAPPINGS = {}
