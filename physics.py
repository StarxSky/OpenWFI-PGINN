import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import pi


def ricker_wavelet(f, nt, dt, t0=None):
    f = torch.as_tensor(f, dtype=torch.float32)
    if t0 is None:
        t0 = 1.5 / f
    t = torch.arange(nt, dtype=torch.float32) * dt
    tau = t - t0
    y = (1.0 - 2.0 * (pi * f * tau) ** 2) * torch.exp(-(pi * f * tau) ** 2)
    return y.view(1, -1)


def denormalize_velocity(v_norm, vmin=1500.0, vmax=4500.0):
    return v_norm * 0.5 * (vmax - vmin) + 0.5 * (vmax + vmin)


def normalize_velocity(v_phys, vmin=1500.0, vmax=4500.0):
    return 2.0 * (v_phys - vmin) / (vmax - vmin) - 1.0


def normalize_data(data_phys, vmin, vmax, k=1.0):
    log_data = torch.sign(data_phys) * torch.log1p(torch.abs(k * data_phys))
    norm_data = 2.0 * (log_data - vmin) / (vmax - vmin) - 1.0
    return norm_data


def denormalize_data(data_norm, vmin, vmax, k=1.0):
    log_data = (data_norm + 1.0) / 2.0 * (vmax - vmin) + vmin
    phys_data = torch.sign(log_data) * (torch.expm1(torch.abs(log_data))) / k
    return phys_data


class DampingBoundary(nn.Module):
    def __init__(self, nz, nx, n_boundary, gamma=0.015):
        super().__init__()
        nzp, nxp = nz + 2 * n_boundary, nx + 2 * n_boundary
        damp = torch.ones(1, 1, nzp, nxp)
        g = torch.as_tensor(gamma)
        for i in range(n_boundary):
            coeff = torch.exp(-(g * (n_boundary - i)) ** 2)
            damp[:, :, i, :] *= coeff
            damp[:, :, -(i + 1), :] *= coeff
            damp[:, :, :, i] *= coeff
            damp[:, :, :, -(i + 1)] *= coeff
        self.register_buffer('damping', damp)

    def forward(self, u):
        return u * self.damping


class AcousticWaveSolver2D(nn.Module):
    """
    2D Acoustic Wave Equation Solver (O(2,2) Finite Differences).

    Solves: d^2 u / dt^2 = v^2 * (d^2 u / dx^2 + d^2 u / dz^2) + s(t)

    Physics parameters are read from dataset_config.json for consistency.
    """

    def __init__(self, dx=10.0, dt=0.001, nz=70, nx=70, n_boundary=30,
                 n_sources=5, n_receivers=70, source_depth=1, receiver_depth=1,
                 source_freq=15.0, save_every=1):
        super().__init__()
        self.dx = dx
        self.dt = dt
        self.nz = nz
        self.nx = nx
        self.n_boundary = n_boundary
        self.n_sources = n_sources
        self.n_receivers = n_receivers
        self.source_freq = source_freq
        self.save_every = save_every

        self.nz_pad = nz + 2 * n_boundary
        self.nx_pad = nx + 2 * n_boundary

        source_x = torch.linspace(n_boundary + 1, n_boundary + nx - 2, n_sources).long()
        source_z = torch.full_like(source_x, n_boundary + source_depth)
        self.register_buffer('source_indices', torch.stack([source_z, source_x], dim=1))

        rx = torch.arange(n_boundary, n_boundary + nx, dtype=torch.long)
        rz = torch.full_like(rx, n_boundary + receiver_depth)
        self.register_buffer('receiver_indices', torch.stack([rz, rx], dim=1))

        wavelet = ricker_wavelet(source_freq, 1000, dt)
        self.register_buffer('source_wavelet', wavelet)
        self.damping = DampingBoundary(nz, nx, n_boundary)

    @staticmethod
    def _fd_step(u_prev, u_cur, c_sq, damping, source_term):
        lap = (u_cur[:, :, 2:, 1:-1] + u_cur[:, :, :-2, 1:-1] +
               u_cur[:, :, 1:-1, 2:] + u_cur[:, :, 1:-1, :-2] -
               4 * u_cur[:, :, 1:-1, 1:-1])
        u_next = torch.zeros_like(u_cur)
        u_next[:, :, 1:-1, 1:-1] = (
            2 * u_cur[:, :, 1:-1, 1:-1] - u_prev[:, :, 1:-1, 1:-1] +
            c_sq[:, :, 1:-1, 1:-1] * lap
        )
        u_next = (u_next + source_term) * damping
        return u_next

    def forward(self, velocity):
        B = velocity.shape[0]
        device = velocity.device
        nb = self.n_boundary
        n_steps = self.source_wavelet.shape[1]

        c_sq = F.pad(velocity, [nb] * 4, mode='replicate') * (self.dt / self.dx) ** 2
        damping = self.damping.damping

        u_prev = torch.zeros(B, 1, self.nz_pad, self.nx_pad, device=device)
        u_cur = torch.zeros(B, 1, self.nz_pad, self.nx_pad, device=device)
        rz, rx = self.receiver_indices[:, 0], self.receiver_indices[:, 1]

        all_shots = []
        for s in range(self.n_sources):
            sz, sx = self.source_indices[s]
            u_prev.zero_()
            u_cur.zero_()
            shot_records = []

            for t in range(n_steps):
                src = torch.zeros(B, 1, self.nz_pad, self.nx_pad, device=device)
                src[:, :, sz, sx] = self.source_wavelet[:, t] * (self.dt ** 2)
                u_next = self._fd_step(u_prev, u_cur, c_sq, damping, src)
                u_prev, u_cur = u_cur, u_next
                if t % self.save_every == 0:
                    shot_records.append(u_cur[:, 0, rz, rx])

            all_shots.append(torch.stack(shot_records, dim=1))

        return torch.stack(all_shots, dim=1)


class PhysicsGuidedLoss(nn.Module):
    """
    Physics-Guided Loss for Full Waveform Inversion.

    L_total = L_data(v_pred, v_true) + lambda_phys * L_phys(F(v_pred), d_obs)

    The physics loss enforces consistency between the predicted velocity model
    and the observed seismic data via the acoustic wave equation.
    """

    def __init__(self, solver, lambda_phys=0.1, data_min=-24.86, data_max=50.28,
                 label_min=1500.0, label_max=4500.0, k=1.0):
        super().__init__()
        self.solver = solver
        self.lambda_phys = lambda_phys
        self.label_min = label_min
        self.label_max = label_max
        self.k = k
        self.register_buffer('data_min', torch.as_tensor(data_min))
        self.register_buffer('data_max', torch.as_tensor(data_max))
        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()

    def forward(self, pred_vel, obs_data, true_vel=None,
                lambda_g1v=1.0, lambda_g2v=1.0):
        loss_g1v = self.l1(pred_vel, true_vel) if true_vel is not None else 0.0
        loss_g2v = self.l2(pred_vel, true_vel) if true_vel is not None else 0.0
        loss_data = lambda_g1v * loss_g1v + lambda_g2v * loss_g2v

        if self.lambda_phys > 0 and self.training:
            v_phys = denormalize_velocity(pred_vel, self.label_min, self.label_max)
            vmin_d, vmax_d = self.data_min.item(), self.data_max.item()

            pred_data = self.solver(v_phys)
            pred_data_norm = normalize_data(pred_data, vmin_d, vmax_d, self.k)

            T = min(obs_data.shape[2], pred_data_norm.shape[2])
            loss_phys = self.l2(pred_data_norm[:, :, :T, :], obs_data[:, :, :T, :])
        else:
            loss_phys = torch.zeros(1, device=pred_vel.device)

        loss = loss_data + self.lambda_phys * loss_phys
        return loss, loss_g1v, loss_g2v, loss_phys
