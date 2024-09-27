try:
    import pyramid_noise_sampling
    from pyramid_noise_sampling import sample_euler_pyramid, sample_heun_pyramid, sample_dpmpp_2s_pyramid

    if pyramid_noise_sampling.BACKEND == "WebUI":
        from modules import scripts, sd_samplers_common, sd_samplers
        from modules.sd_samplers_kdiffusion import sampler_extra_params, KDiffusionSampler

        class Pyramid(scripts.Script):
            def title(self):
                "Pyramid_Noise Samplers"

            def show(self, is_img2img):
                return False

            def __init__(self):
                if not pyramid_noise_sampling.INITIALIZED:
                    samplers_pyramid = [
                        ("Euler pyramid", sample_euler_pyramid, ["k_euler_pyramid"], {}),
                        ("Heun pyramid", sample_heun_pyramid, ["k_heun_pyramid"], {}),
                        ("DPM++ 2s pyramid", sample_dpmpp_2s_pyramid, ["k_dpmpp_2s_pyramid"], {}),
                    ]
                    samplers_data_pyramid = [
                        sd_samplers_common.SamplerData(label, lambda model, funcname=funcname: KDiffusionSampler(funcname, model), aliases, options)
                        for label, funcname, aliases, options in samplers_pyramid
                        if callable(funcname)
                    ]
                    sampler_extra_params["sample_euler_pyramid"] = ["s_churn", "s_tmin", "s_tmax", "s_noise"]
                    sampler_extra_params["sample_heun_pyramid"] = ["s_churn", "s_tmin", "s_tmax", "s_noise"]
                    sampler_extra_params["sample_dpmpp_2s_pyramid"] = ["s_churn", "s_tmin", "s_tmax", "s_noise"]
                    sd_samplers.all_samplers.extend(samplers_data_pyramid)
                    sd_samplers.all_samplers_map = {x.name: x for x in sd_samplers.all_samplers}
                    sd_samplers.set_samplers()
                    pyramid_noise_sampling.INITIALIZED = True

except ImportError as _:
    pass
