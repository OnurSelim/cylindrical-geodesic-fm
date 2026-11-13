import torch
from tqdm import tqdm

from .path import ConditionalProbabilityPath


class UNetVectorFieldODE:
    def __init__(self, net: torch.nn.Module):
        self.net = net

    def drift_coefficient(self, x, t, cond=None, x0=None, **kwargs) -> torch.Tensor:
        return self.net(x, t, cond, x0)


class ProjectedHeunSimulator:
    def __init__(self, ode: UNetVectorFieldODE | None = None, path: ConditionalProbabilityPath = None):
        self.ode = ode
        self.path = path

    def _euler_update(self, xt: torch.Tensor, drift: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        while dt.ndim < drift.ndim:
            dt = dt.unsqueeze(-1)
        if self.path is None:
            return xt + drift * dt
        xt_sim_next = self.path.to_simulation_coords(xt) + drift * dt
        xt_sim_next = self.path.project_simulation_coords(xt_sim_next)
        return self.path.from_simulation_coords(xt_sim_next)

    def step(self, xt, t, dt, cond: torch.Tensor | None = None, x0: torch.Tensor | None = None, **kwargs):
        drift = kwargs.pop("drift", None)
        drift_next = kwargs.pop("drift_next", None)

        if drift is None:
            if self.ode is None:
                raise ValueError("ProjectedHeunSimulator.step requires either an ODE or a `drift` tensor.")
            drift = self.ode.drift_coefficient(xt, t, cond=cond, x0=x0, **kwargs)

        xt_pred = self._euler_update(xt, drift, dt)

        if drift_next is None:
            if self.ode is None:
                return xt_pred
            drift_next = self.ode.drift_coefficient(xt_pred, t + dt, cond=cond, x0=x0, **kwargs)

        return self._euler_update(xt, 0.5 * (drift + drift_next), dt)

    @torch.no_grad()
    def simulate(self, x, ts, cond=None, x0=None, **kwargs) -> torch.Tensor:
        for i in tqdm(range(ts.shape[1] - 1)):
            x = self.step(x, ts[:, i], ts[:, i + 1] - ts[:, i], cond=cond, x0=x0, **kwargs)
        return x

    @torch.no_grad()
    def simulate_with_trajectory(
        self,
        x: torch.Tensor,
        ts: torch.Tensor,
        num_intermediate: int | None = None,
        cond: torch.Tensor | None = None,
        x0: torch.Tensor | None = None,
        **kwargs
    ) -> torch.Tensor:
        trajectory = [x.clone()]
        nts = ts.shape[1]
        if num_intermediate is None:
            save_set = None
        else:
            num_intermediate = max(0, min(num_intermediate, max(nts - 2, 0)))
            if num_intermediate == 0:
                save_set = set()
            else:
                indices = torch.linspace(1, nts - 2, steps=num_intermediate)
                save_set = set(indices.round().to(torch.int64).tolist())

        for i in tqdm(range(nts - 1)):
            x = self.step(x, ts[:, i], ts[:, i + 1] - ts[:, i], cond=cond, x0=x0, **kwargs)
            step_idx = i + 1
            if save_set is None or step_idx in save_set or step_idx == nts - 1:
                trajectory.append(x.clone())

        return torch.stack(trajectory, dim=1)
