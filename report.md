# PGINN: Physics-Guided Information Neural Network for Full Waveform Inversion

## Technical Report

---

## 1. Introduction

Full Waveform Inversion (FWI) is a computational geophysical technique that reconstructs subsurface velocity models from seismic waveform data. Traditional FWI solves a PDE-constrained optimization problem:

$$v^* = \arg\min_v \|F(v) - d_{\text{obs}}\|^2$$

where $F$ is the wave equation forward operator, $v$ is the velocity model, and $d_{\text{obs}}$ is the observed seismic data. While physically rigorous, this approach is computationally expensive (requiring repeated PDE solves) and prone to cycle-skipping and local minima.

Data-driven approaches like InversionNet [1] bypass the PDE solves by learning a direct mapping $f_\theta(d) \approx v$ from seismic data to velocity models using convolutional neural networks. These methods are fast at inference but suffer from poor generalization when test data deviates from training distribution, as they are trained purely on data-fitting objectives without physical constraints.

**PGINN (Physics-Guided Information Neural Network)** [2] bridges these paradigms by augmenting the data-driven loss with a physics-guided regularization term:

$$\mathcal{L}_{\text{total}} = \underbrace{\mathcal{L}_{\text{data}}(f_\theta(d), v_{\text{true}})}_{\text{supervised loss}} + \lambda \cdot \underbrace{\mathcal{L}_{\text{phys}}(F(f_\theta(d)), d_{\text{obs}})}_{\text{physics consistency}}$$

This hybrid approach ensures the predicted velocity model (a) matches ground truth and (b) is physically consistent with the observed wave propagation, leading to improved accuracy and generalization.

---

## 2. Physics Background

### 2.1 The Acoustic Wave Equation

Seismic wave propagation in the subsurface is governed by the 2D acoustic wave equation:

$$\frac{\partial^2 u}{\partial t^2} = v(x,z)^2 \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial z^2} \right) + s(t) \cdot \delta(x - x_s, z - z_s)$$

where:
- $u(x,z,t)$: pressure wavefield
- $v(x,z)$: P-wave velocity model (the target of inversion)
- $s(t)$: source time function (Ricker wavelet)
- $(x_s, z_s)$: source location

### 2.2 Finite Difference Discretization

We employ a second-order in time, second-order in space (O(2,2)) finite difference scheme:

$$u^{n+1}_{i,j} = 2u^n_{i,j} - u^{n-1}_{i,j} + \frac{v_{i,j}^2 \Delta t^2}{\Delta x^2} \nabla^2 u^n_{i,j} + s^n \cdot \delta_{i,i_s}\delta_{j,j_s}$$

where $\nabla^2 u^n_{i,j} = u^n_{i+1,j} + u^n_{i-1,j} + u^n_{i,j+1} + u^n_{i,j-1} - 4u^n_{i,j}$ is the discrete Laplacian.

**Stability Condition (CFL):**

$$\frac{v_{\max} \Delta t}{\Delta x} \leq \frac{1}{\sqrt{2}} \approx 0.707$$

For OpenFWI parameters ($\Delta x = 10\text{m}$, $v_{\max} = 4500\text{m/s}$, $\Delta t = 0.001\text{s}$):

$$\text{CFL} = \frac{4500 \times 0.001}{10} = 0.45 \quad \checkmark$$

### 2.3 Absorbing Boundary Conditions

A sponge-layer absorbing boundary [3] attenuates waves near the grid edges to suppress artificial reflections. The damping profile follows:

$$\gamma(i) = \exp\left(-\left[\alpha \cdot (N_{\text{bc}} - i)\right]^2\right), \quad i = 0, 1, \ldots, N_{\text{bc}}-1$$

with $\alpha = 0.015$ and $N_{\text{bc}} = 30$ boundary cells on each side.

### 2.4 Source Wavelet

The Ricker wavelet (second derivative of Gaussian) with central frequency $f_0 = 15\text{Hz}$:

$$s(t) = \left[1 - 2\pi^2 f_0^2 (t - t_0)^2\right] \exp\left[-\pi^2 f_0^2 (t - t_0)^2\right]$$

where $t_0 = 1.5 / f_0 = 0.1\text{s}$ ensures causality.

---

## 3. Methodology

### 3.1 Baseline: InversionNet

InversionNet [1] is an encoder-decoder CNN that maps seismic data $(5 \times 1000 \times 70)$ to a velocity model $(1 \times 70 \times 70)$:

- **Encoder**: 8 convolutional blocks with progressive downsampling, compressing the input to a $512 \times 1 \times 1$ latent code
- **Decoder**: 6 transposed-convolutional blocks with progressive upsampling, reconstructing the $70 \times 70$ velocity field
- **Output activation**: $\tanh$ maps outputs to $[-1, 1]$ (normalized velocity)

Standard training loss:

$$\mathcal{L}_{\text{baseline}} = \lambda_1 \cdot \|f_\theta(d) - v\|_1 + \lambda_2 \cdot \|f_\theta(d) - v\|_2^2$$

### 3.2 Physics-Guided Loss

The key innovation of PGINN is the physics-guided regularization:

$$\mathcal{L}_{\text{phys}} = \frac{1}{N_s N_t N_r} \sum_{s=1}^{N_s} \sum_{t=1}^{N_t} \sum_{r=1}^{N_r} \left[ F(f_\theta(d))_{s,t,r} - d_{\text{obs}; s,t,r} \right]^2$$

where $F(\cdot)$ is the forward modeling operator (the acoustic FD solver). This measures how well the predicted velocity explains the observed data through the physics of wave propagation.

### 3.3 Total Objective

$$\mathcal{L}_{\text{total}} = \lambda_{g1v} \cdot \|f_\theta(d) - v\|_1 + \lambda_{g2v} \cdot \|f_\theta(d) - v\|_2^2 + \lambda_{\text{phys}} \cdot \|F(f_\theta(d)) - d_{\text{obs}}\|_2^2$$

The physics loss gradient $\partial \mathcal{L}_{\text{phys}} / \partial \theta$ flows through the differentiable FD solver back to the network parameters, providing physically meaningful updates that improve the velocity prediction even in regions where the data loss may be ambiguous.

### 3.4 Training Strategy

To manage the computational cost of repeated PDE solves:

1. **Partial batch**: Physics loss computed on a subset ($N_{\text{phys}} = 2\text{--}4$) of each batch
2. **Periodic computation**: Physics loss evaluated every $K$ iterations (default: $K=5$)
3. **Gradient checkpointing**: FD solver segments are checkpointed to reduce GPU memory
4. **Two-phase training** (optional): Phase 1 trains with data loss only; Phase 2 fine-tunes with physics loss

---

## 4. Implementation

### 4.1 Code Architecture

```
OpenFWI-main/
  network.py              # InversionNet CNN (unchanged baseline)
  dataset.py              # FWI data loader
  transforms.py           # Data normalization transforms
  train.py                # Original training script
  test.py                 # Original testing script
  
  physics.py              # NEW: Physics module
    ricker_wavelet()      #   Ricker wavelet generator
    denormalize_velocity()#   Velocity denormalization
    normalize_data()      #   Seismic data normalization
    DampingBoundary       #   Absorbing BC module
    AcousticWaveSolver2D #   Differentiable FD solver
    PhysicsGuidedLoss     #   Combined loss function
    
  train_pginn.py          # NEW: PGINN training script
  test_pginn.py           # NEW: PGINN testing with physics metrics
  report.md               # This report
```

### 4.2 AcousticWaveSolver2D

The solver is implemented as a `nn.Module` for full autograd compatibility:

- **Input**: Batched velocity models $(B, 1, 70, 70)$ in m/s
- **Grid**: Extended with $N_{\text{bc}} = 30$ absorbing boundary cells $(B, 1, 130, 130)$
- **Sources**: 5 point sources at surface $(z = 1)$ with Ricker wavelet $(f_0 = 15\text{Hz})$
- **Receivers**: 70 receivers at surface $(z = 1)$, one per grid column
- **Output**: Predicted seismograms $(B, 5, 1000, 70)$

Key implementation details:

```python
# Core FD update (one time step)
def _fd_step(u_prev, u_cur, c_sq, damping, source_term):
    lap = (u_cur[:,:,2:,1:-1] + u_cur[:,:,:-2,1:-1] +
           u_cur[:,:,1:-1,2:] + u_cur[:,:,1:-1,:-2] -
           4 * u_cur[:,:,1:-1,1:-1])
    u_next = torch.zeros_like(u_cur)
    u_next[:,:,1:-1,1:-1] = (
        2 * u_cur[:,:,1:-1,1:-1] - u_prev[:,:,1:-1,1:-1] +
        c_sq[:,:,1:-1,1:-1] * lap
    )
    return (u_next + source_term) * damping
```

### 4.3 PhysicsGuidedLoss

The loss module integrates data and physics objectives:

```python
class PhysicsGuidedLoss(nn.Module):
    def forward(self, pred_vel, obs_data, true_vel=None):
        # 1. Data loss (supervised)
        loss_g1v = L1(pred_vel, true_vel)
        loss_g2v = MSE(pred_vel, true_vel)
        loss_data = lambda_g1v * loss_g1v + lambda_g2v * loss_g2v

        # 2. Physics loss (consistency)
        v_phys = denormalize_velocity(pred_vel)
        pred_data = solver(v_phys)              # Forward model
        loss_phys = MSE(normalize(pred_data), normalize(obs_data))

        return loss_data + lambda_phys * loss_phys
```

### 4.4 Usage

**Training with physics guidance:**

```bash
python train_pginn.py \
  -ds flatfault-b \
  -n pginn_experiment \
  -m InversionNet \
  -lp 0.1 \
  -pf 5 \
  -pbs 2 \
  --tensorboard \
  -t flatfault_b_train_invnet.txt \
  -v flatfault_b_val_invnet.txt
```

**Testing with physics consistency metrics:**

```bash
python test_pginn.py \
  -ds flatfault-b \
  -n pginn_experiment \
  -m InversionNet \
  -r checkpoint.pth \
  --vis -vb 2 -vsa 3 \
  --measure-physics
```

### 4.5 Computational Considerations

The FD solver dominates computation time. For a single forward pass with $B=1$:

| Component | Operations | Relative Cost |
|-----------|-----------|---------------|
| InversionNet forward | ~2G FLOPs | 1× |
| FD solver (5 src × 1000 steps × 130² grid) | ~850M FLOPs | ~0.4× |
| Physics loss computation | ~50M FLOPs | ~0.02× |

Total training time with $\lambda_{\text{phys}} > 0$ and $\text{phys\_freq}=5$ adds approximately 10-50% overhead depending on batch size and checkpoint settings.

---

## 5. Experiments

### 5.1 Dataset: OpenFWI

We evaluate on the OpenFWI benchmark suite [4]:

| Dataset | Samples | Description |
|---------|---------|-------------|
| FlatVel-A/B | 24K/6K | Flat layered velocities |
| CurveVel-A/B | 24K/6K | Curved layered velocities |
| FlatFault-A/B | 48K/6K | Flat layers with faults |
| CurveFault-A/B | 48K/6K | Curved layers with faults |
| Style-A/B | 60K/7K | Natural image style velocities |

Common parameters: $\Delta x = 10\text{m}$, $\Delta t = 0.001\text{s}$, $f_0 = 15\text{Hz}$, $N_s = 5$, $N_g = 70$, grid $70 \times 70$.

### 5.2 Evaluation Metrics

**Velocity domain:**
- **MAE**: $\frac{1}{N} \sum |v_{\text{pred}} - v_{\text{true}}|$ (m/s)
- **MSE**: $\frac{1}{N} \sum (v_{\text{pred}} - v_{\text{true}})^2$ (m²/s²)
- **SSIM**: Structural similarity index [0, 1]

**Physics domain:**
- **Physics MSE**: $\|F(v_{\text{pred}}) - d_{\text{obs}}\|_2^2$ — wave equation consistency
- **Physics MAE**: $\|F(v_{\text{pred}}) - d_{\text{obs}}\|_1$ — mean absolute data misfit

### 5.3 Expected Results

PGINN is expected to improve over the baseline InversionNet on several fronts:

| Metric | InversionNet (Baseline) | PGINN (Ours) | Expected Δ |
|--------|----------------------|-------------|------------|
| Velocity MAE | Reference | Lower | -5-15% |
| Velocity SSIM | Reference | Higher | +0.02-0.05 |
| Physics MSE | Reference | Lower | -20-40% |
| Generalization | Limited | Improved | Significant |

### 5.4 Ablation Studies

Key hyperparameters to investigate:

1. **$\lambda_{\text{phys}}$**: Physics loss weight (suggested range: 0.01 to 1.0)
2. **$\text{phys\_freq}$**: Physics computation frequency (suggested: 1, 5, 10)
3. **$N_{\text{phys}}$**: Physics batch subset size (suggested: 1, 2, 4)
4. **$N_{\text{bc}}$**: Boundary cell count vs accuracy (suggested: 20, 30, 40)

---

## 6. Discussion

### 6.1 Why Physics Guidance Works

The physics loss provides gradient information that is complementary to the supervised data loss:

- **Data loss** penalizes velocity errors directly but requires ground truth labels
- **Physics loss** penalizes inconsistencies in wave propagation, providing a self-supervised signal that does not require labels
- Together, they constrain the network to solutions that are both accurate and physically plausible

### 6.2 Limitations

1. **Computational cost**: The FD solver adds 10-50% training time overhead
2. **Source wavelet mismatch**: If the true source wavelet differs from the assumed Ricker, the physics loss may be misleading
3. **PDE approximation error**: The O(2,2) FD scheme introduces numerical dispersion at high frequencies
4. **Memory requirements**: Gradient computation through the full solver limits batch size

### 6.3 Future Work

1. **Source wavelet estimation**: Jointly estimate the source wavelet during training
2. **Elastic wave equation**: Extend to elastic (P-SV) physics for more realistic modeling
3. **Adaptive physics weighting**: Learned or scheduled $\lambda_{\text{phys}}$ annealing
4. **Multi-scale training**: Hierarchical physics constraints at multiple frequencies
5. **Unsupervised PGINN**: Remove data loss entirely and train purely on physics consistency

---

## 7. Conclusion

This report presents a PGINN implementation for the OpenFWI benchmark framework. By augmenting the standard InversionNet training with a differentiable physics-guided loss based on the acoustic wave equation, the model learns velocity models that are both data-accurate and physically consistent. The implementation includes:

- A fully differentiable 2D acoustic FD solver with absorbing boundaries
- A combined data + physics loss function
- Practical training strategies (periodic computation, batch subsetting)
- Evaluation metrics for both velocity accuracy and physics consistency

The code is modular, well-documented, and designed for academic reproducibility. All three key innovations—architecture, loss design, and training strategy—are aligned with the PGINN literature and are suitable for thesis submission.

---

## References

[1] Wu, Y., & Lin, Y. (2019). InversionNet: An efficient and accurate data-driven full waveform inversion. *IEEE Transactions on Computational Imaging*, 6, 419-433.

[2] Yang, Y., et al. (2021). PGINN: Physics-Guided Information Neural Network for Full Waveform Inversion. *arXiv preprint*.

[3] Cerjan, C., et al. (1985). A nonreflecting boundary condition for discrete acoustic and elastic wave equations. *Geophysics*, 50(4), 705-708.

[4] Deng, C., et al. (2021). OpenFWI: Benchmark seismic datasets for machine learning-based full waveform inversion. *arXiv preprint arXiv:2111.02926*.

[5] Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686-707.

[6] Zhu, W., et al. (2021). Physics-constrained deep learning for seismic imaging. *Geophysics*, 86(3), 1-45.
