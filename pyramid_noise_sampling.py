from importlib import import_module
from tqdm.auto import trange
import torch
import random
import tqdm

sampling = None
BACKEND = None
INITIALIZED = False

if not BACKEND:
    try:
        _ = import_module("modules.sd_samplers_kdiffusion")
        sampling = import_module("k_diffusion.sampling")
        BACKEND = "WebUI"
    except ImportError as _:
        pass

if not BACKEND:
    try:
        sampling = import_module("comfy.k_diffusion.sampling")
        BACKEND = "ComfyUI"
    except ImportError as _:
        pass


def get_sigmas_karras(n, sigma_min, sigma_max, rho=7., device='cpu'):
    """Constructs the noise schedule of Karras et al. (2022)."""
    ramp = torch.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas).to(device)


def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])


def pyramid_noise_like2(noise, iterations=5, discount=0.4):
    # iterations * discount less than 2, for example, 4 * 0.3, 8 * 0.15,
    b, c, w, h = noise.shape
    u = torch.nn.Upsample(size=(w, h), mode="bilinear").cuda()
    for i in range(iterations):
        r = random.random() * 2 + 2
        wn, hn = max(1, int(w / (r ** i))), max(1, int(h / (r ** i)))
        temp_noise = torch.randn(b, c, wn, hn).cuda()
        noise += u(temp_noise) * discount ** i
        if wn == 1 or hn == 1:
            break
    return noise / noise.std()


def default_noise_sampler(x):
    return lambda sigma, sigma_next: torch.randn_like(x)


def get_ancestral_step(sigma_from, sigma_to, eta=1.):
    """Calculates the noise level (sigma_down) to step down to and the amount
    of noise to add (sigma_up) when doing an ancestral sampling step."""
    if not eta:
        return sigma_to, 0.
    sigma_up = min(sigma_to, eta * (sigma_to ** 2 * (sigma_from ** 2 - sigma_to ** 2) / sigma_from ** 2) ** 0.5)
    sigma_down = (sigma_to ** 2 - sigma_up ** 2) ** 0.5
    return sigma_down, sigma_up


@torch.no_grad()
def sample_euler_pyramid(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1.,
                         noise_sampler=None):
    """using pyramid noise for euler a"""
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    # addition noise to original noise
    addition_noise = torch.randn_like(x)
    x = x + pyramid_noise_like2(x)
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        d = sampling.to_d(x, sigmas[i], denoised)
        # Euler method
        dt = sigma_down - sigmas[i]
        x = x + d * dt
        if sigmas[i + 1] > 0:
            # get pyramid noise
            noise_up = pyramid_noise_like2(noise_sampler(sigmas[i], sigmas[i + 1]),
                                           iterations=8,
                                           discount=0.25)
            x = x + noise_up * s_noise * sigma_up
    return x


@torch.no_grad()
def sample_heun_pyramid(model, x, sigmas, extra_args=None, callback=None, disable=None, s_noise=1., restart_list=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    step_id = 0

    addition_noise = torch.randn_like(x)
    x = x + pyramid_noise_like2(x)

    def heun_step(x, old_sigma, new_sigma, second_order=True):
        nonlocal step_id
        denoised = model(x, old_sigma * s_in, **extra_args)
        d = sampling.to_d(x, old_sigma, denoised)
        if callback is not None:
            callback({'x': x, 'i': step_id, 'sigma': new_sigma, 'sigma_hat': old_sigma, 'denoised': denoised})
        dt = new_sigma - old_sigma
        if new_sigma == 0 or not second_order:
            # Euler method
            x = x + d * dt
        else:
            # Heun's method
            x_2 = x + d * dt
            denoised_2 = model(x_2, new_sigma * s_in, **extra_args)
            d_2 = sampling.to_d(x_2, new_sigma, denoised_2)
            d_prime = (d + d_2) / 2
            x = x + d_prime * dt
        step_id += 1
        return x

    steps = sigmas.shape[0] - 1
    if restart_list is None:
        if steps >= 10:
            restart_steps = 3
            restart_times = 1
            if steps >= 20:
                restart_steps = steps // 4
                restart_times = 2
            sigmas = get_sigmas_karras(steps - restart_steps * restart_times, sigmas[-2].item(), sigmas[0].item(),
                                       device=sigmas.device)
            restart_list = {0.1: [restart_steps + 1, restart_times, 2]}
        else:
            restart_list = {}

    restart_list = {int(torch.argmin(abs(sigmas - key), dim=0)): value for key, value in restart_list.items()}

    step_list = []
    for i in range(len(sigmas) - 1):
        step_list.append((sigmas[i], sigmas[i + 1]))
        if i + 1 in restart_list:
            restart_steps, restart_times, restart_max = restart_list[i + 1]
            min_idx = i + 1
            max_idx = int(torch.argmin(abs(sigmas - restart_max), dim=0))
            if max_idx < min_idx:
                sigma_restart = get_sigmas_karras(restart_steps, sigmas[min_idx].item(), sigmas[max_idx].item(),
                                                  device=sigmas.device)[:-1]
                while restart_times > 0:
                    restart_times -= 1
                    step_list.extend(zip(sigma_restart[:-1], sigma_restart[1:]))

    last_sigma = None
    for old_sigma, new_sigma in tqdm.tqdm(step_list, disable=disable):
        if last_sigma is None:
            last_sigma = old_sigma
        elif last_sigma < old_sigma:
            # print(f"add noise here,sigma is{sigmas}")
            noise_up = pyramid_noise_like2(torch.randn_like(x),
                                           iterations=4,
                                           discount=0.125)
            x = x + noise_up * s_noise * (old_sigma ** 2 - last_sigma ** 2) ** 0.5
        x = heun_step(x, old_sigma, new_sigma)
        # print(f"now old_sigma is {old_sigma},and new_sigma is {new_sigma}")
        last_sigma = new_sigma

    return x


@torch.no_grad()
def sample_dpmpp_2s_pyramid(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1.,
                            noise_sampler=None):
    """pyramid sampling with DPM-Solver++(2S) second-order steps."""
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    # addition noise to original noise
    addition_noise = torch.randn_like(x)
    x = x + pyramid_noise_like2(x)
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigma_down == 0:
            # Euler method
            d = sampling.to_d(x, sigmas[i], denoised)
            dt = sigma_down - sigmas[i]
            x = x + d * dt
        else:
            # DPM-Solver++(2S)
            t, t_next = t_fn(sigmas[i]), t_fn(sigma_down)
            r = 1 / 2
            h = t_next - t
            s = t + r * h
            x_2 = (sigma_fn(s) / sigma_fn(t)) * x - (-h * r).expm1() * denoised
            denoised_2 = model(x_2, sigma_fn(s) * s_in, **extra_args)
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_2
        # Noise addition
        if sigmas[i + 1] > 0:
            # get pyramid noise
            noise_up = pyramid_noise_like2(noise_sampler(sigmas[i], sigmas[i + 1]),
                                           iterations=8,
                                           discount=0.25)
            x = x + noise_up * s_noise * sigma_up
    return x
