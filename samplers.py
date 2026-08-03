import torch
import numpy as np


def expand_t_like_x(t, x_cur):
    """Function to reshape time t to broadcastable dimension of x
    Args:
      t: [batch_dim,], time vector
      x: [batch_dim,...], data point
    """
    dims = [1] * (len(x_cur.size()) - 1)
    t = t.view(t.size(0), *dims)
    return t


def get_score_from_velocity(vt, xt, t, path_type="linear"):
    """Wrapper function: transfrom velocity prediction model to score
    Args:
        velocity: [batch_dim, ...] shaped tensor; velocity model output
        x: [batch_dim, ...] shaped tensor; x_t data point
        t: [batch_dim,] time tensor
    """
    t = expand_t_like_x(t, xt)
    if path_type == "linear":
        alpha_t, d_alpha_t = 1 - t, torch.ones_like(xt, device=xt.device) * -1
        sigma_t, d_sigma_t = t, torch.ones_like(xt, device=xt.device)
    elif path_type == "cosine":
        alpha_t = torch.cos(t * np.pi / 2)
        sigma_t = torch.sin(t * np.pi / 2)
        d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
        d_sigma_t = np.pi / 2 * torch.cos(t * np.pi / 2)
    else:
        raise NotImplementedError

    mean = xt
    reverse_alpha_ratio = alpha_t / d_alpha_t
    var = sigma_t ** 2 - reverse_alpha_ratio * d_sigma_t * sigma_t
    score = (reverse_alpha_ratio * vt - mean) / var

    return score


def compute_diffusion(t_cur):
    return 2 * t_cur


def euler_sampler(
        model,
        latents,
        cond_kwargs,
        uncond_kwargs,
        num_steps=20,
        heun=False,
        cfg_scale=1.0,
        guidance_low=0.0,
        guidance_high=1.0,
        path_type="linear",
):
    _dtype = latents.dtype
    t_steps = torch.linspace(1, 0, num_steps + 1, dtype=torch.float64)
    x_next = latents.to(torch.float64)
    device = x_next.device

    def get_model_input(x, t, c_kwargs, u_kwargs):
        if cfg_scale > 1.0 and t <= guidance_high and t >= guidance_low:
            model_input = torch.cat([x] * 2, dim=0)
            t_input = torch.cat([t.expand(x.shape[0])] * 2, dim=0) if t.dim() == 0 else torch.cat([t] * 2, dim=0)

            combined_kwargs = {}
            for k in c_kwargs.keys():
                if k in ["prompt_embeds", "pooled_embeds"]:
                    combined_kwargs[k] = torch.cat([c_kwargs[k], u_kwargs[k]], dim=0)

                elif k in ["text_ids", "img_ids"]:
                    combined_kwargs[k] = c_kwargs[k]

                elif isinstance(c_kwargs[k], torch.Tensor):
                    val_u = u_kwargs.get(k, c_kwargs[k])
                    combined_kwargs[k] = torch.cat([c_kwargs[k], val_u], dim=0)
                else:
                    combined_kwargs[k] = c_kwargs[k]

            out = model(model_input.to(_dtype), t_input.to(_dtype), **combined_kwargs)
            if isinstance(out, tuple): out = out[0]
            out_cond, out_uncond = out.chunk(2)
            return out_uncond.to(torch.float64) + cfg_scale * (
                        out_cond.to(torch.float64) - out_uncond.to(torch.float64))
        else:
            t_input = t.expand(x.shape[0]) if t.dim() == 0 else t
            out = model(x.to(_dtype), t_input.to(_dtype), **c_kwargs)
            if isinstance(out, tuple): out = out[0]
            return out.to(torch.float64)

    with torch.no_grad():
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next
            d_cur = get_model_input(x_cur, t_cur.to(device), cond_kwargs, uncond_kwargs)
            x_next = x_cur + (t_next - t_cur) * d_cur

            if heun and (i < num_steps - 1):
                d_prime = get_model_input(x_next, t_next.to(device), cond_kwargs, uncond_kwargs)
                x_next = x_cur + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next


def euler_sampler_flux2(
        model,
        latents,
        cond_kwargs,  # x_ids, ctx, ctx_ids, y, res, lon, lat
        uncond_kwargs,  # null text / null metadata counterparts
        num_steps=20,
        heun=False,
        cfg_scale=1.0,
        guidance_low=0.0,
        guidance_high=1.0,
        path_type="linear",
):
    _dtype = latents.dtype
    # flow matching runs from t=1 (noise) to t=0 (data)
    t_steps = torch.linspace(1, 0, num_steps + 1, dtype=torch.float64)
    x_next = latents.to(torch.float64)
    device = x_next.device

    def get_model_input(x, t, c_kwargs, u_kwargs):
        if cfg_scale > 1.0 and guidance_low <= t <= guidance_high:
            # duplicate the batch: B -> 2B
            model_input = torch.cat([x] * 2, dim=0)
            t_input = torch.cat([t.expand(x.shape[0])] * 2, dim=0)

            combined_kwargs = {}
            for k in c_kwargs.keys():
                if k in ["ctx", "y"]:
                    # concatenate conditional and unconditional text embeddings
                    combined_kwargs[k] = torch.cat([c_kwargs[k], u_kwargs[k]], dim=0)

                elif k in ["res", "lon", "lat"]:
                    # geospatial metadata is usually shared; fall back to the conditional value
                    val_u = u_kwargs.get(k, c_kwargs[k])
                    combined_kwargs[k] = torch.cat([c_kwargs[k], val_u], dim=0)

                elif k in ["x_ids", "ctx_ids"]:
                    # RoPE ids are duplicated only when batched as [B, L, 3]
                    if c_kwargs[k].dim() == 3 and c_kwargs[k].shape[0] == x.shape[0]:
                        combined_kwargs[k] = torch.cat([c_kwargs[k]] * 2, dim=0)
                    else:
                        combined_kwargs[k] = c_kwargs[k]

                else:
                    combined_kwargs[k] = c_kwargs[k]

            # the GSA/REPA head output is unused at sampling time
            out = model(model_input.to(_dtype), combined_kwargs.pop("x_ids"),
                        t_input.to(_dtype), **combined_kwargs)

            if isinstance(out, tuple):
                out = out[0]  # keep img from (img, z_repa)

            # CFG: uncond + scale * (cond - uncond)
            out_cond, out_uncond = out.chunk(2)
            v_final = out_uncond.to(torch.float64) + cfg_scale * (
                    out_cond.to(torch.float64) - out_uncond.to(torch.float64))
            return v_final

        else:
            # no CFG: single forward pass
            t_input = t.expand(x.shape[0])
            k_copy = {**c_kwargs}
            x_ids = k_copy.pop("x_ids")
            out = model(x.to(_dtype), x_ids, t_input.to(_dtype), **k_copy)

            if isinstance(out, tuple):
                out = out[0]
            return out.to(torch.float64)

    # Euler / Heun Integration
    with torch.no_grad():
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next
            d_cur = get_model_input(x_cur, t_cur.to(device), cond_kwargs, uncond_kwargs)

            # x_{t+1} = x_t + dt * v
            x_next = x_cur + (t_next - t_cur) * d_cur

            if heun and (i < num_steps - 1):
                d_prime = get_model_input(x_next, t_next.to(device), cond_kwargs, uncond_kwargs)
                x_next = x_cur + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next.to(_dtype)


def euler_maruyama_sampler(
        model,
        latents,
        y,
        y_null,
        num_steps=20,
        heun=False,  # not used, just for compatability
        cfg_scale=1.0,
        guidance_low=0.0,
        guidance_high=1.0,
        path_type="linear",
):
    # setup conditioning
    _dtype = latents.dtype

    t_steps = torch.linspace(1., 0.04, num_steps, dtype=torch.float64)
    t_steps = torch.cat([t_steps, torch.tensor([0.], dtype=torch.float64)])
    x_next = latents.to(torch.float64)
    device = x_next.device

    with torch.no_grad():
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-2], t_steps[1:-1])):
            dt = t_next - t_cur
            x_cur = x_next
            if cfg_scale > 1.0 and t_cur <= guidance_high and t_cur >= guidance_low:
                model_input = torch.cat([x_cur] * 2, dim=0)
                y_cur = torch.cat([y, y_null], dim=0)
            else:
                model_input = x_cur
                y_cur = y
            kwargs = dict(context=y_cur)
            time_input = torch.ones(model_input.size(0)).to(device=device, dtype=torch.float64) * t_cur
            diffusion = compute_diffusion(t_cur)
            eps_i = torch.randn_like(x_cur).to(device)
            deps = eps_i * torch.sqrt(torch.abs(dt))

            # compute drift
            v_cur = model(
                model_input.to(dtype=_dtype), time_input.to(dtype=_dtype), **kwargs
            )[0].to(torch.float64)
            s_cur = get_score_from_velocity(v_cur, model_input, time_input, path_type=path_type)
            d_cur = v_cur - 0.5 * diffusion * s_cur
            if cfg_scale > 1. and t_cur <= guidance_high and t_cur >= guidance_low:
                d_cur_cond, d_cur_uncond = d_cur.chunk(2)
                d_cur = d_cur_uncond + cfg_scale * (d_cur_cond - d_cur_uncond)

            x_next = x_cur + d_cur * dt + torch.sqrt(diffusion) * deps

    # last step
    t_cur, t_next = t_steps[-2], t_steps[-1]
    dt = t_next - t_cur
    x_cur = x_next
    if cfg_scale > 1.0 and t_cur <= guidance_high and t_cur >= guidance_low:
        model_input = torch.cat([x_cur] * 2, dim=0)
        y_cur = torch.cat([y, y_null], dim=0)
    else:
        model_input = x_cur
        y_cur = y
    kwargs = dict(context=y_cur)
    time_input = torch.ones(model_input.size(0)).to(
        device=device, dtype=torch.float64
    ) * t_cur

    # compute drift
    v_cur = model(
        model_input.to(dtype=_dtype), time_input.to(dtype=_dtype), **kwargs
    )[0].to(torch.float64)
    s_cur = get_score_from_velocity(v_cur, model_input, time_input, path_type=path_type)
    diffusion = compute_diffusion(t_cur)
    d_cur = v_cur - 0.5 * diffusion * s_cur
    if cfg_scale > 1. and t_cur <= guidance_high and t_cur >= guidance_low:
        d_cur_cond, d_cur_uncond = d_cur.chunk(2)
        d_cur = d_cur_uncond + cfg_scale * (d_cur_cond - d_cur_uncond)

    mean_x = x_cur + dt * d_cur

    return mean_x