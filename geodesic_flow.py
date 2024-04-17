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

    def __init__(self, target_snr, schedule="linear", objective="v_pred"):
 
        if target_snr is None:
            target_snr = float("inf")
        else:
            target_snr = max(target_snr, 1e-10)

        self.target_snr = target_snr # 3.5177683092482117
        self.schedule = schedule
        self.objective = objective

        if schedule == "cos":
            time_scale = np.arccos(4*np.arctan(1/target_snr)/torch.pi - 1) / torch.pi # 0.72412583
            def theta_fn(timesteps):
                return (1 - ((1-timesteps) * torch.pi * time_scale).cos()) / 2 * (torch.pi/2)
        elif schedule == "linear":
            time_scale = np.arctan(target_snr) / (np.pi/2) # 0.8236786557085517
            def theta_fn(timesteps):
                return (1 - timesteps) * time_scale * (torch.pi/2)
        elif schedule == "acos":
            time_scale = (1 - np.cos(2*np.arctan(target_snr))) / 2 # ~0.925
            def theta_fn(timesteps):
                return np.arccos(1 - 2*(1-timesteps) * time_scale)/2
        else:
            raise ValueError(f"Invalid schedule: {schedule}")
        
        if objective == "v_pred":
            def objective_fn(sample, noise, timesteps):
                return slerp(noise, sample, self.get_timestep_theta(timesteps) / (torch.pi/2) + 1)
        elif objective == "rectified_flow":
            def objective_fn(sample, noise, timesteps):
                return slerp(noise, sample, 1.5)
        else:
            raise ValueError(f"Invalid objective: {objective}")
            
        self.theta_fn = theta_fn
        self.objective_fn = objective_fn
    
    def get_timestep_theta(self, timesteps):
        original_dtype = timesteps.dtype
        return self.theta_fn(timesteps).to(original_dtype)

    def get_timestep_snr(self, timesteps):
        original_dtype = timesteps.dtype
        return self.get_timestep_theta(timesteps.to(torch.float64)).tan().to(original_dtype)

    def add_noise(self, sample, noise, timesteps):

        original_dtype = sample.dtype
        sample = normalize(sample, zero_mean=True)
        noise = normalize(noise, zero_mean=True)

        noised_sample = slerp(noise, sample, self.get_timestep_theta(timesteps) / (torch.pi/2))
        return normalize(noised_sample, zero_mean=True).to(original_dtype)
    
    def get_objective(self, sample, noise, timesteps):
        
        original_dtype = sample.dtype
        sample = normalize(sample, zero_mean=True)
        noise = normalize(noise, zero_mean=True)

        objective = self.objective_fn(sample, noise, timesteps)
        return normalize(objective, zero_mean=True).to(original_dtype)
    
    def reverse_step(self, sample, model_output, v_scale): # if sampling over n steps, v_scale = 1/n
        
        original_dtype = sample.dtype
        output_v_scale = model_output.std(dim=(1,2,3), keepdim=True) * (torch.pi/2) * v_scale

        sample = normalize(sample, zero_mean=True)
        model_output = normalize(model_output, zero_mean=True)
        
        denoised_sample = slerp(sample, model_output, output_v_scale)
        return normalize(denoised_sample, zero_mean=True).to(original_dtype)

    def reverse_step_fixed(self, sample, model_output, v_scale, t, next_t):
        
        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=torch.float64, device=sample.device)
        if not torch.is_tensor(next_t):
            next_t = torch.tensor(next_t, dtype=torch.float64, device=sample.device)

        original_dtype = sample.dtype
        output_v_scale = (self.get_timestep_theta(next_t) - self.get_timestep_theta(t)) / (torch.pi/2)

        sample = normalize(sample, zero_mean=True)
        model_output = normalize(model_output, zero_mean=True)
        
        denoised_sample = slerp(sample, model_output, v_scale * output_v_scale)
        return normalize(denoised_sample, zero_mean=True).to(original_dtype)


if __name__ == "__main__":

    from dual_diffusion_utils import save_raw
    from dotenv import load_dotenv
    import os

    load_dotenv(override=True)

    target_snr = 3.5177683092482117
    flow = GeodesicFlow(target_snr, schedule="acos")

    timesteps = torch.linspace(1, 0, 120+1, dtype=torch.float64)[:-1]

    min_timestep_normalized_theta = flow.get_timestep_theta(timesteps.amin()) / (torch.pi/2)
    min_timestep_snr = flow.get_timestep_snr(timesteps.amin())
    max_timestep_normalized_theta = flow.get_timestep_theta(timesteps.amax()) / (torch.pi/2)
    max_timestep_snr = flow.get_timestep_snr(timesteps.amax())

    timestep_normalized_theta = flow.get_timestep_theta(timesteps) / (torch.pi/2)
    timestep_snr = flow.get_timestep_snr(timesteps)
    timestep_noise_std = timestep_snr.atan().cos()
    timestep_normalized_velocity = timestep_normalized_theta[1:] - timestep_normalized_theta[:-1]

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
        save_raw(timestep_snr.clip(min=1e-10).log(), os.path.join(debug_path, "timestep_ln_snr.raw"))
        save_raw(timestep_normalized_velocity, os.path.join(debug_path, "timestep_normalized_velocity.raw"))