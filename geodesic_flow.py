import numpy as np
import torch

def slerp(start, end, t, dtype=torch.float64):

    if torch.is_tensor(t):
        t = t.to(dtype)
        if t.ndim < start.ndim:
            t = t.view(*t.shape, *((1,) * (start.ndim - t.ndim)))
    
    start, end = start.to(dtype), end.to(dtype)
    omega = get_cos_angle(start, end, keepdim=True, dtype=dtype)
    so = torch.sin(omega)

    return (torch.sin((1 - t) * omega) / so) * start + (torch.sin(t * omega) / so) * end

def get_cos_angle(start, end, keepdim=False, dtype=torch.float64):
    
    reduction_dims = tuple(range(1, start.ndim)) if start.ndim > 1 else (0,)

    start, end = start.to(dtype), end.to(dtype)
    start_len = torch.linalg.vector_norm(start, dim=reduction_dims, keepdim=True, dtype=dtype)
    end_len = torch.linalg.vector_norm(end, dim=reduction_dims, keepdim=True, dtype=dtype)

    return (start / start_len * end / end_len).sum(dim=reduction_dims, keepdim=keepdim).clamp(-1, 1).acos()

def normalize(x, zero_mean=False, dtype=torch.float64):

    reduction_dims = tuple(range(1, x.ndim)) if x.ndim > 1 else (0,)
    x = x.to(dtype)

    if zero_mean:
        x = x - x.mean(dim=reduction_dims, keepdim=True)

    return x / x.square().mean(dim=reduction_dims, keepdim=True).sqrt()

class GeodesicFlow:

    @torch.no_grad()
    def __init__(self, target_snr, schedule="linear", objective="scaled_v_pred"):
 
        if target_snr is None:
            target_snr = float("inf")
        else:
            target_snr = max(target_snr, 1e-10)

        self.target_snr = target_snr # 3.5177683092482117
        self.schedule = schedule
        self.objective = objective

        if schedule == "cos": # this means the _angle_ is a cosine function of the timestep (truncated to target_snr)
            time_scale = np.arccos(4*np.arctan(1/target_snr)/torch.pi - 1) / torch.pi # 0.72412583
            def theta_fn(timesteps):
                return (1 - ((1-timesteps) * torch.pi * time_scale).cos()) / 2 * (torch.pi/2)
        elif schedule == "linear": # this means _angular_ linear (similar to variance preserving cosine schedules, but truncated to target_snr)
            time_scale = np.arctan(target_snr) / (np.pi/2) # 0.8236786557085517
            def theta_fn(timesteps):
                return (1 - timesteps) * time_scale * (torch.pi/2)
        elif schedule == "acos":
            time_scale = (1 - np.cos(2*np.arctan(target_snr))) / 2 # ~0.925
            def theta_fn(timesteps):
                return (1 - 2*(1-timesteps) * time_scale).acos()/2
        elif schedule == "atan": # (linear snr)
            def theta_fn(timesteps):
                return ((1 - timesteps) * target_snr).atan()
        elif schedule == "edm2":
            time_scale = target_snr * (target_snr + (target_snr**2 + 1) ** 0.5)
            def theta_fn(timesteps):
                return (1 / (time_scale * (timesteps - 1) - 1) + 1).asin()
        else:
            raise ValueError(f"Invalid schedule: {schedule}")

        if objective == "epsilon": # classic noise prediction
            def objective_fn(sample, noise, timesteps):
                return -noise
        elif objective == "sample": # classic x0 prediction
            def objective_fn(sample, noise, timesteps): 
                return sample
        elif objective == "v_pred": # classic v-pred
            def objective_fn(sample, noise, timesteps): 
                return slerp(noise, sample, self.get_timestep_theta(timesteps) / (torch.pi/2) + 1/(get_cos_angle(noise, sample) / (torch.pi/2)))
        elif objective == "rectified_flow": # equivalent to variance preserving sample - noise
            bias = (np.arctan(target_snr) / (np.pi/2) - 1) / 2
            def objective_fn(sample, noise, timesteps):
                return slerp(noise, sample, 1.5 / (get_cos_angle(noise, sample) / (torch.pi/2)) + bias)
        elif objective == "scaled_v_pred": # v-pred with objective scaled to a specific target_snr
            scale = np.arctan(target_snr) / (np.pi/2) # 0.8236786557085517
            def objective_fn(sample, noise, timesteps):
                return slerp(noise, sample, self.get_timestep_theta(timesteps) / (torch.pi/2) + scale/(get_cos_angle(noise, sample) / (torch.pi/2)))
        else:
            raise ValueError(f"Invalid objective: {objective}")
            
        self.theta_fn = theta_fn
        self.objective_fn = objective_fn
    
    @torch.no_grad()
    def get_timestep_theta(self, timesteps): # theta is the timestep as an angle between 0 (no signal) and pi/2 (no noise)
        original_dtype = timesteps.dtype
        return self.theta_fn(timesteps.to(torch.float64)).to(original_dtype)

    @torch.no_grad()
    def get_timestep_snr(self, timesteps): # snr for a timestep angle is just tan(theta)
        original_dtype = timesteps.dtype
        return self.get_timestep_theta(timesteps.to(torch.float64)).tan().to(original_dtype)

    @torch.no_grad()
    def get_timestep_ln_snr(self, timesteps, eps=1e-4):
        original_dtype = timesteps.dtype
        timestep_snr = self.get_timestep_theta(timesteps.to(torch.float64)).tan()
        return timestep_snr.clip(min=eps).log().to(original_dtype)
    
    @torch.no_grad()
    def get_timestep_sigma(self, timesteps, eps=1e-3): # non-relative noise std if signal std == 1
        original_dtype = timesteps.dtype
        timestep_theta = self.get_timestep_theta(timesteps.to(torch.float64)).clip(min=eps)
        return timestep_theta.cos() / timestep_theta.sin().to(original_dtype)
    
    @torch.no_grad()
    def get_timestep_noise_label(self, timesteps):
        return timesteps
    
    @torch.no_grad()
    def add_noise(self, sample, noise, timesteps):
        original_dtype = sample.dtype
        theta = self.get_timestep_theta(timesteps) / get_cos_angle(sample, noise)
        noised_sample = slerp(noise, sample, theta)
        return normalize(noised_sample).to(original_dtype)
    
    @torch.no_grad()
    def get_objective(self, sample, noise, timesteps):
        original_dtype = sample.dtype
        objective = self.objective_fn(sample, noise, timesteps)
        return normalize(objective).to(original_dtype)

    @torch.no_grad()
    def reverse_step(self, sample, model_output, v_scale, p_scale, t, next_t, generator=None):
        
        original_dtype = sample.dtype

        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=torch.float64, device=sample.device)
        if not torch.is_tensor(next_t):
            next_t = torch.tensor(next_t, dtype=torch.float64, device=sample.device)

        if p_scale > 0: # interpret the model output as a gaussian distribution with an implicit variance given by 1 - |output|^2
            model_output = model_output.to(torch.float64)
            output_len = model_output.square().mean(dim=(1,2,3), keepdim=True)
            perturbation = torch.empty_like(model_output).normal_(generator=generator)
            model_output = model_output + (1 - output_len.clip(max=1)).sqrt() * perturbation * p_scale
        
        if self.objective == "v_pred":
            v_scale = v_scale * (self.get_timestep_theta(next_t) - self.get_timestep_theta(t)) / (torch.pi/2)
        else:
            v_scale = v_scale * (self.get_timestep_theta(next_t) - self.get_timestep_theta(t)) / get_cos_angle(sample, model_output)
            
        denoised_sample = slerp(sample, model_output, v_scale)
        return normalize(denoised_sample).to(original_dtype)


if __name__ == "__main__": # small test with some debug output for target_snr / schedule / objective

    from dual_diffusion_utils import save_raw, get_fractal_noise2d
    from dotenv import load_dotenv
    import os

    load_dotenv(override=True)

    target_snr = 16
    schedule = "edm2"
    objective = "v_pred"
    num_steps = 200
    noise_degree = 0
    
    print("target_snr:", target_snr, "schedule:", schedule, "objective:", objective, "num_steps:", num_steps)

    flow = GeodesicFlow(target_snr, schedule=schedule, objective=objective)
    timesteps = torch.linspace(1, 0, 200+1, dtype=torch.float64)[:-1]
    if noise_degree == 0:
        noise_fn = torch.randn
    else:
        noise_fn = lambda shape, **kwargs: get_fractal_noise2d(shape, **kwargs)

    min_timestep_normalized_theta = flow.get_timestep_theta(timesteps.amin()) / (torch.pi/2)
    min_timestep_snr = flow.get_timestep_snr(timesteps.amin())
    max_timestep_normalized_theta = flow.get_timestep_theta(timesteps.amax()) / (torch.pi/2)
    max_timestep_snr = flow.get_timestep_snr(timesteps.amax())

    timestep_normalized_theta = flow.get_timestep_theta(timesteps) / (torch.pi/2)
    timestep_snr = flow.get_timestep_snr(timesteps)
    timestep_noise_std = timestep_snr.atan().cos()
    timestep_normalized_velocity = timestep_normalized_theta[1:] - timestep_normalized_theta[:-1]

    sample = normalize(noise_fn(1, 4, 256, 256))
    noise = normalize(noise_fn(sample.shape))
    noised_sample = flow.add_noise(sample, noise, timesteps)
    objective = flow.get_objective(sample, noise, timesteps)
    print("input * objective normalized cos angles:")
    print(get_cos_angle(noised_sample, objective) / (torch.pi/2))
    print("")

    print("min_timestep_normalized_theta:", min_timestep_normalized_theta)
    print("min_timestep_snr:", min_timestep_snr)
    print("max_timestep_normalized_theta:", max_timestep_normalized_theta)
    print("max_timestep_snr:", max_timestep_snr)

    debug_path = os.environ.get("DEBUG_PATH", None)
    if debug_path is not None:
        debug_path = os.path.join(debug_path, "geodesic_flow")

        save_raw(timestep_normalized_theta, os.path.join(debug_path, "timestep_normalized_theta.raw"))
        save_raw(timestep_snr, os.path.join(debug_path, "timestep_snr.raw"))
        save_raw(timestep_noise_std, os.path.join(debug_path, "timestep_noise_std.raw"))
        save_raw(timestep_snr.clip(min=timestep_snr[1:].amin()).log(), os.path.join(debug_path, "timestep_ln_snr.raw"))
        save_raw(timestep_normalized_velocity, os.path.join(debug_path, "timestep_normalized_velocity.raw"))