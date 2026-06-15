# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import warp as wp

import newton.ik as ik
from newton._src.sim.ik.ik_common import IKJacobianType


def _normalize_active_dof_indices(active_dof_indices, n_model_dofs):
    if active_dof_indices is None:
        return np.arange(n_model_dofs, dtype=np.int32)

    active_dof_indices = np.asarray(active_dof_indices, dtype=np.int32)
    if active_dof_indices.ndim != 1:
        raise ValueError("active_dof_indices must be a 1D array")
    if len(active_dof_indices) == 0:
        raise ValueError("active_dof_indices cannot be empty")
    if np.any(active_dof_indices < 0) or np.any(active_dof_indices >= n_model_dofs):
        raise ValueError(
            f"active_dof_indices must be in [0, {n_model_dofs}); got "
            f"[{active_dof_indices.min()}, {active_dof_indices.max()}]")
    if len(np.unique(active_dof_indices)) != len(active_dof_indices):
        raise ValueError("active_dof_indices cannot contain duplicates")

    return active_dof_indices


@wp.func
def _wp_smooth_joint_filter_func(
    x            : wp.float32,
    lower_limit  : wp.float32,
    upper_limit  : wp.float32,
    padding_limit: wp.float32,
    m            : wp.float32,
    p            : wp.float32
):
    c = (lower_limit + upper_limit) * 0.5
    lower_limit += (padding_limit - c)
    upper_limit -= (padding_limit + c)
    if lower_limit < x and x <= upper_limit:
        return 0.0

    diff = wp.where(x <= lower_limit, lower_limit-x, x-upper_limit) * m
    return 1.0 - wp.exp(-wp.pow(diff, p))


@wp.kernel
def _smooth_joint_filter_residuals(
    joint_q: wp.array2d(dtype=wp.float32),           # (n_batch, n_coords)
    dof_to_coord: wp.array1d(dtype=wp.int32),        # (n_dofs)
    joint_limit_lower: wp.array1d(dtype=wp.float32), # (n_dofs)
    joint_limit_upper: wp.array1d(dtype=wp.float32), # (n_dofs)
    coord_masks: wp.array1d(dtype=wp.float32),       # (n_coords)
    weight: wp.array1d(dtype=wp.float32),            # (1)
    start_idx: int,
    # outputs
    residuals: wp.array2d(dtype=wp.float32),     # (n_batch, n_residuals)
):
    problem, dof_idx = wp.tid()
    coord_idx = dof_to_coord[dof_idx]
    mask = coord_masks[coord_idx]

    if coord_idx < 0:
        return

    if mask > 0.0:
        lower = joint_limit_lower[dof_idx]
        upper = joint_limit_upper[dof_idx]
        c = (lower + upper) * 0.5

        q = joint_q[problem, coord_idx]
        error = (q - c)

        smoother = _wp_smooth_joint_filter_func(error, lower, upper, 1.02, 1.0, 6.5)
        residuals[problem, start_idx + dof_idx] = error * smoother * weight[0] * mask
    else:
        residuals[problem, start_idx + dof_idx] = 0.0


@wp.kernel
def _limit_residuals_indexed(
    joint_q: wp.array2d(dtype=wp.float32),           # (n_batch, n_coords)
    joint_limit_lower: wp.array1d(dtype=wp.float32), # (n_model_dofs)
    joint_limit_upper: wp.array1d(dtype=wp.float32), # (n_model_dofs)
    active_dof_indices: wp.array1d(dtype=wp.int32),  # (n_active_dofs)
    active_dof_to_coord: wp.array1d(dtype=wp.int32), # (n_active_dofs)
    weight: float,
    start_idx: int,
    # outputs
    residuals: wp.array2d(dtype=wp.float32),         # (n_batch, n_residuals)
):
    problem, active_idx = wp.tid()
    dof_idx = active_dof_indices[active_idx]
    coord_idx = active_dof_to_coord[active_idx]

    if coord_idx < 0:
        return

    q = joint_q[problem, coord_idx]
    lower = joint_limit_lower[dof_idx]
    upper = joint_limit_upper[dof_idx]

    # Newton treats huge ranges as no limit.
    if upper - lower > 9.9e5:
        return

    viol = wp.max(0.0, q - upper) + wp.max(0.0, lower - q)
    residuals[problem, start_idx + active_idx] = weight * viol


@wp.kernel
def _limit_jac_analytic_indexed(
    joint_q: wp.array2d(dtype=wp.float32),           # (n_batch, n_coords)
    joint_limit_lower: wp.array1d(dtype=wp.float32), # (n_model_dofs)
    joint_limit_upper: wp.array1d(dtype=wp.float32), # (n_model_dofs)
    active_dof_indices: wp.array1d(dtype=wp.int32),  # (n_active_dofs)
    active_dof_to_coord: wp.array1d(dtype=wp.int32), # (n_active_dofs)
    start_idx: int,
    weight: float,
    # outputs
    jacobian: wp.array3d(dtype=wp.float32),          # (n_batch, n_residuals, n_model_dofs)
):
    problem, active_idx = wp.tid()
    dof_idx = active_dof_indices[active_idx]
    coord_idx = active_dof_to_coord[active_idx]

    if coord_idx < 0:
        return

    q = joint_q[problem, coord_idx]
    lower = joint_limit_lower[dof_idx]
    upper = joint_limit_upper[dof_idx]

    if upper - lower > 9.9e5:
        return

    grad = float(0.0)
    if q >= upper:
        grad = weight
    elif q <= lower:
        grad = -weight

    jacobian[problem, start_idx + active_idx, dof_idx] = grad


@wp.kernel
def _jac_fill_indexed(
    q_grad: wp.array2d(dtype=wp.float32),            # (n_batch, n_model_dofs)
    active_dof_indices: wp.array1d(dtype=wp.int32),  # (n_active_dofs)
    start_idx: int,
    # outputs
    jacobian: wp.array3d(dtype=wp.float32),
):
    problem, active_idx = wp.tid()
    dof_idx = active_dof_indices[active_idx]
    jacobian[problem, start_idx + active_idx, dof_idx] = q_grad[problem, dof_idx]


@wp.kernel
def _smooth_joint_filter_residuals_indexed(
    joint_q: wp.array2d(dtype=wp.float32),           # (n_batch, n_coords)
    active_dof_indices: wp.array1d(dtype=wp.int32),  # (n_active_dofs)
    active_dof_to_coord: wp.array1d(dtype=wp.int32), # (n_active_dofs)
    joint_limit_lower: wp.array1d(dtype=wp.float32), # (n_model_dofs)
    joint_limit_upper: wp.array1d(dtype=wp.float32), # (n_model_dofs)
    coord_masks: wp.array1d(dtype=wp.float32),       # (n_coords)
    weight: wp.array1d(dtype=wp.float32),            # (1)
    start_idx: int,
    # outputs
    residuals: wp.array2d(dtype=wp.float32),         # (n_batch, n_residuals)
):
    problem, active_idx = wp.tid()
    dof_idx = active_dof_indices[active_idx]
    coord_idx = active_dof_to_coord[active_idx]

    if coord_idx < 0:
        return

    mask = coord_masks[coord_idx]
    if mask <= 0.0:
        residuals[problem, start_idx + active_idx] = 0.0
        return

    lower = joint_limit_lower[dof_idx]
    upper = joint_limit_upper[dof_idx]
    c = (lower + upper) * 0.5

    q = joint_q[problem, coord_idx]
    error = (q - c)

    smoother = _wp_smooth_joint_filter_func(error, lower, upper, 1.02, 1.0, 6.5)
    residuals[problem, start_idx + active_idx] = error * smoother * weight[0] * mask


@wp.kernel
def _update_weight(
    in_value: wp.float32,
    out_weight: wp.array1d(dtype=wp.float32),  # (1)
):
    out_weight[0] = in_value


@wp.kernel
def _smooth_joint_filter_jac_analytic(
    dof_to_coord: wp.array1d(dtype=wp.int32),    # (n_dofs)
    coord_masks: wp.array1d(dtype=wp.float32),   # (n_coords)
    n_dofs: int,
    start_idx: int,
    weight: wp.array1d(dtype=wp.float32), # (1)
    # outputs
    jacobian: wp.array3d(dtype=wp.float32),      # (n_batch, n_residuals, n_dofs)
):
    problem, dof_idx = wp.tid()
    coord_idx = dof_to_coord[dof_idx]
    mask = coord_masks[coord_idx]

    if coord_idx < 0:
        return

    # Jacobian is diagonal: dr[dof]/dq[dof] = weight
    jacobian[problem, start_idx + dof_idx, dof_idx] = weight[0] * mask


@wp.kernel
def _smooth_joint_filter_jac_analytic_indexed(
    active_dof_indices: wp.array1d(dtype=wp.int32),  # (n_active_dofs)
    active_dof_to_coord: wp.array1d(dtype=wp.int32), # (n_active_dofs)
    coord_masks: wp.array1d(dtype=wp.float32),       # (n_coords)
    start_idx: int,
    weight: wp.array1d(dtype=wp.float32),            # (1)
    # outputs
    jacobian: wp.array3d(dtype=wp.float32),          # (n_batch, n_residuals, n_model_dofs)
):
    problem, active_idx = wp.tid()
    dof_idx = active_dof_indices[active_idx]
    coord_idx = active_dof_to_coord[active_idx]

    if coord_idx < 0:
        return

    mask = coord_masks[coord_idx]
    jacobian[problem, start_idx + active_idx, dof_idx] = weight[0] * mask


class IKJointLimit(ik.IKObjective):
    """Joint-limit objective whose residual rows can be restricted to active DOFs."""

    def __init__(self, joint_limit_lower, joint_limit_upper, weight=0.1, active_dof_indices=None):
        super().__init__()
        self.joint_limit_lower = joint_limit_lower
        self.joint_limit_upper = joint_limit_upper
        self.weight = weight
        self.n_model_dofs = len(joint_limit_lower)
        self.active_dof_indices_np = _normalize_active_dof_indices(active_dof_indices, self.n_model_dofs)
        self.n_dofs = len(self.active_dof_indices_np)
        self.active_dof_indices = None
        self.active_dof_to_coord = None
        self.e_array = None

    def init_buffers(self, model, jacobian_mode):
        self._require_batch_layout()

        dof_to_coord_np = _build_dof_to_coord(model, self.n_model_dofs)
        self.active_dof_indices = wp.array(self.active_dof_indices_np, dtype=wp.int32, device=self.device)
        self.active_dof_to_coord = wp.array(
            dof_to_coord_np[self.active_dof_indices_np],
            dtype=wp.int32,
            device=self.device)

        if jacobian_mode == IKJacobianType.AUTODIFF:
            e = np.zeros((self.n_batch, self.total_residuals), dtype=np.float32)
            for prob_idx in range(self.n_batch):
                for active_idx in range(self.n_dofs):
                    e[prob_idx, self.residual_offset + active_idx] = 1.0
            self.e_array = wp.array(e.flatten(), dtype=wp.float32, device=self.device)

    def supports_analytic(self):
        return True

    def residual_dim(self):
        return self.n_dofs

    def compute_residuals(self, body_q, joint_q, model, residuals, start_idx, problem_idx):
        count = joint_q.shape[0]
        wp.launch(
            _limit_residuals_indexed,
            dim=[count, self.n_dofs],
            inputs=[
                joint_q,
                self.joint_limit_lower,
                self.joint_limit_upper,
                self.active_dof_indices,
                self.active_dof_to_coord,
                self.weight,
                start_idx,
            ],
            outputs=[residuals],
            device=self.device,
        )

    def compute_jacobian_autodiff(self, tape, model, jacobian, start_idx, dq_dof):
        self._require_batch_layout()
        tape.backward(grads={tape.outputs[0]: self.e_array})
        q_grad = tape.gradients[dq_dof]

        wp.launch(
            _jac_fill_indexed,
            dim=[self.n_batch, self.n_dofs],
            inputs=[
                q_grad,
                self.active_dof_indices,
                start_idx,
            ],
            outputs=[jacobian],
            device=self.device,
        )

    def compute_jacobian_analytic(self, body_q, joint_q, model, jacobian, joint_S_s, start_idx):
        count = joint_q.shape[0]
        wp.launch(
            _limit_jac_analytic_indexed,
            dim=[count, self.n_dofs],
            inputs=[
                joint_q,
                self.joint_limit_lower,
                self.joint_limit_upper,
                self.active_dof_indices,
                self.active_dof_to_coord,
                start_idx,
                self.weight,
            ],
            outputs=[jacobian],
            device=self.device,
        )


def _build_dof_to_coord(model, n_model_dofs):
    dof_to_coord_np = np.full(n_model_dofs, -1, dtype=np.int32)
    q_start_np = model.joint_q_start.numpy()
    qd_start_np = model.joint_qd_start.numpy()
    joint_dof_dim_np = model.joint_dof_dim.numpy()

    for j in range(model.joint_count):
        dof0 = int(qd_start_np[j])
        coord0 = int(q_start_np[j])
        lin, ang = joint_dof_dim_np[j]
        for k in range(int(lin + ang)):
            if dof0 + k < n_model_dofs:
                dof_to_coord_np[dof0 + k] = coord0 + k

    return dof_to_coord_np


class IKSmoothJointFilter(ik.IKObjective):
    """
    An IK objective that applies a smooth penalty to joint coordinates that approach or exceed specified limits
    using an inverse gaussian filter.

    Args:
        joint_limit_lower (wp.array1d): An array of shape (n_dofs,) containing the lower limits for each joint degree of freedom.
        joint_limit_upper (wp.array1d): An array of shape (n_dofs,) containing the upper limits for each joint degree of freedom.
        weight (float, optional): A scalar weight that controls the strength of the joint limit penalty. Defaults to 0.01.
        coord_masks (wp.array1d, optional): An array of shape (n_coords,) containing mask values for each joint coordinate.
            Mask values should be in the range [0, 1], where 0 means the coordinate is ignored by this objective and 1 means it is fully considered.
            All coords are used by default if no masks are specified.
    """
    def __init__(self, joint_limit_lower, joint_limit_upper, weight=0.01, coord_masks=None, active_dof_indices=None):
        super().__init__()
        self.joint_limit_lower = joint_limit_lower
        self.joint_limit_upper = joint_limit_upper
        self.n_model_dofs = len(joint_limit_lower)
        self.active_dof_indices_np = _normalize_active_dof_indices(active_dof_indices, self.n_model_dofs)
        self.n_dofs = len(self.active_dof_indices_np)
        self.active_dof_indices = None
        self.active_dof_to_coord = None
        self.e_array = None
        self._weight = wp.array([weight], dtype=wp.float32)

        self.coord_masks = None
        self.coord_masks_np = None
        if coord_masks is not None:
            if isinstance(coord_masks, np.ndarray):
                self.coord_masks_np = coord_masks.astype(np.float32)
                self.coord_masks = None
            elif isinstance(coord_masks, wp.array):
                self.coord_masks = coord_masks
                self.coord_masks_np = None

    def bind_device(self, device):
        super().bind_device(device)

    def init_buffers(self, model, jacobian_mode):
        self._require_batch_layout()

        if self.coord_masks_np is not None and len(self.coord_masks_np) == model.joint_coord_count:
            self.coord_masks = wp.array(self.coord_masks_np, dtype=wp.float32, device=self.device)

        # All coords are considered if no coord masks have been declared
        if self.coord_masks is None:
            self.coord_masks = wp.ones(shape=model.joint_coord_count, dtype=wp.float32, device=self.device)

        dof_to_coord_np = _build_dof_to_coord(model, self.n_model_dofs)
        self.active_dof_indices = wp.array(self.active_dof_indices_np, dtype=wp.int32, device=self.device)
        self.active_dof_to_coord = wp.array(
            dof_to_coord_np[self.active_dof_indices_np],
            dtype=wp.int32,
            device=self.device)

        # For autodiff mode
        if jacobian_mode == IKJacobianType.AUTODIFF:
            e = np.zeros((self.n_batch, self.total_residuals), dtype=np.float32)
            for prob_idx in range(self.n_batch):
                for active_idx in range(self.n_dofs):
                    e[prob_idx, self.residual_offset + active_idx] = 1.0
            self.e_array = wp.array(e.flatten(), dtype=wp.float32, device=self.device)

    def supports_analytic(self):
        return True

    def residual_dim(self):
        return self.n_dofs

    def set_weight(self, value):
        if self.coord_masks is None:
            return

        wp.launch(
            _update_weight,
            dim=1,
            inputs=[value],
            outputs=[self._weight],
            device=self.device)

    def compute_residuals(self, body_q, joint_q, model, residuals, start_idx, problem_idx):
        count = joint_q.shape[0]
        wp.launch(
            _smooth_joint_filter_residuals_indexed,
            dim=[count, self.n_dofs],
            inputs=[
                joint_q,
                self.active_dof_indices,
                self.active_dof_to_coord,
                self.joint_limit_lower,
                self.joint_limit_upper,
                self.coord_masks,
                self._weight,
                start_idx,
            ],
            outputs=[residuals],
            device=self.device,
        )

    def compute_jacobian_autodiff(self, tape, model, jacobian, start_idx, dq_dof):
        self._require_batch_layout()
        tape.backward(grads={tape.outputs[0]: self.e_array})

        q_grad = tape.gradients[dq_dof]

        # Use the analytic Jacobian fill since it's simple
        wp.launch(
            _smooth_joint_filter_jac_analytic_indexed,
            dim=[self.n_batch, self.n_dofs],
            inputs=[
                self.active_dof_indices,
                self.active_dof_to_coord,
                self.coord_masks,
                start_idx,
                self._weight,
            ],
            outputs=[jacobian],
            device=self.device,
        )

    def compute_jacobian_analytic(self, body_q, joint_q, model, jacobian, joint_S_s, start_idx):
        count = joint_q.shape[0]
        wp.launch(
            _smooth_joint_filter_jac_analytic_indexed,
            dim=[count, self.n_dofs],
            inputs=[
                self.active_dof_indices,
                self.active_dof_to_coord,
                self.coord_masks,
                start_idx,
                self._weight,
            ],
            outputs=[jacobian],
            device=self.device,
        )


@wp.kernel
def _joint_motion_smooth_residuals(
    joint_q: wp.array2d(dtype=wp.float32),           # (n_batch, n_coords)
    target_joint_q: wp.array2d(dtype=wp.float32),    # (n_batch, n_coords)
    dof_to_coord: wp.array1d(dtype=wp.int32),        # (n_dofs)
    coord_masks: wp.array1d(dtype=wp.float32),       # (n_coords)
    weight: wp.array1d(dtype=wp.float32),            # (1)
    start_idx: int,
    # outputs
    residuals: wp.array2d(dtype=wp.float32),         # (n_batch, n_residuals)
):
    problem, dof_idx = wp.tid()
    coord_idx = dof_to_coord[dof_idx]
    if coord_idx < 0:
        return

    mask = coord_masks[coord_idx]
    residuals[problem, start_idx + dof_idx] = (
        joint_q[problem, coord_idx] - target_joint_q[problem, coord_idx]
    ) * weight[0] * mask


@wp.kernel
def _joint_motion_smooth_residuals_indexed(
    joint_q: wp.array2d(dtype=wp.float32),           # (n_batch, n_coords)
    target_joint_q: wp.array2d(dtype=wp.float32),    # (n_batch, n_coords)
    active_dof_indices: wp.array1d(dtype=wp.int32),  # (n_active_dofs)
    active_dof_to_coord: wp.array1d(dtype=wp.int32), # (n_active_dofs)
    coord_masks: wp.array1d(dtype=wp.float32),       # (n_coords)
    weight: wp.array1d(dtype=wp.float32),            # (1)
    start_idx: int,
    # outputs
    residuals: wp.array2d(dtype=wp.float32),         # (n_batch, n_residuals)
):
    problem, active_idx = wp.tid()
    dof_idx = active_dof_indices[active_idx]
    coord_idx = active_dof_to_coord[active_idx]
    if coord_idx < 0:
        return

    mask = coord_masks[coord_idx]
    residuals[problem, start_idx + active_idx] = (
        joint_q[problem, coord_idx] - target_joint_q[problem, coord_idx]
    ) * weight[0] * mask


class IKJointMotionSmooth(ik.IKObjective):
    """Temporal joint smoothing objective.

    This objective penalizes deviation from a per-frame target joint
    configuration, typically the previous frame's solved ``joint_q``. Unlike
    ``IKSmoothJointFilter``, this is a true temporal smoothing term.
    """

    def __init__(self, joint_limit_lower, weight=0.0, coord_masks=None, active_dof_indices=None):
        super().__init__()
        self.n_model_dofs = len(joint_limit_lower)
        self.active_dof_indices_np = _normalize_active_dof_indices(active_dof_indices, self.n_model_dofs)
        self.n_dofs = len(self.active_dof_indices_np)
        self.active_dof_indices = None
        self.active_dof_to_coord = None
        self.e_array = None
        self.target_joint_q = None
        self._weight = wp.array([weight], dtype=wp.float32)

        self.coord_masks = None
        self.coord_masks_np = None
        if coord_masks is not None:
            if isinstance(coord_masks, np.ndarray):
                self.coord_masks_np = coord_masks.astype(np.float32)
            elif isinstance(coord_masks, wp.array):
                self.coord_masks = coord_masks

    def bind_device(self, device):
        super().bind_device(device)

    def init_buffers(self, model, jacobian_mode):
        self._require_batch_layout()

        if self.coord_masks_np is not None and len(self.coord_masks_np) == model.joint_coord_count:
            self.coord_masks = wp.array(self.coord_masks_np, dtype=wp.float32, device=self.device)
        if self.coord_masks is None:
            self.coord_masks = wp.ones(shape=model.joint_coord_count, dtype=wp.float32, device=self.device)

        dof_to_coord_np = _build_dof_to_coord(model, self.n_model_dofs)
        self.active_dof_indices = wp.array(self.active_dof_indices_np, dtype=wp.int32, device=self.device)
        self.active_dof_to_coord = wp.array(
            dof_to_coord_np[self.active_dof_indices_np],
            dtype=wp.int32,
            device=self.device)
        self.target_joint_q = wp.zeros(
            shape=(self.n_batch, model.joint_coord_count),
            dtype=wp.float32,
            device=self.device,
        )

        if jacobian_mode == IKJacobianType.AUTODIFF:
            e = np.zeros((self.n_batch, self.total_residuals), dtype=np.float32)
            for prob_idx in range(self.n_batch):
                for active_idx in range(self.n_dofs):
                    e[prob_idx, self.residual_offset + active_idx] = 1.0
            self.e_array = wp.array(e.flatten(), dtype=wp.float32, device=self.device)

    def supports_analytic(self):
        return True

    def residual_dim(self):
        return self.n_dofs

    def set_weight(self, value):
        if self.coord_masks is None:
            return
        wp.launch(
            _update_weight,
            dim=1,
            inputs=[value],
            outputs=[self._weight],
            device=self.device,
        )

    def set_target_joint_q(self, joint_q):
        if self.target_joint_q is None:
            return
        wp.copy(self.target_joint_q, joint_q)

    def compute_residuals(self, body_q, joint_q, model, residuals, start_idx, problem_idx):
        count = joint_q.shape[0]
        wp.launch(
            _joint_motion_smooth_residuals_indexed,
            dim=[count, self.n_dofs],
            inputs=[
                joint_q,
                self.target_joint_q,
                self.active_dof_indices,
                self.active_dof_to_coord,
                self.coord_masks,
                self._weight,
                start_idx,
            ],
            outputs=[residuals],
            device=self.device,
        )

    def compute_jacobian_autodiff(self, tape, model, jacobian, start_idx, dq_dof):
        self._require_batch_layout()
        tape.backward(grads={tape.outputs[0]: self.e_array})
        self.compute_jacobian_analytic(None, None, model, jacobian, None, start_idx)

    def compute_jacobian_analytic(self, body_q, joint_q, model, jacobian, joint_S_s, start_idx):
        count = joint_q.shape[0]
        wp.launch(
            _smooth_joint_filter_jac_analytic_indexed,
            dim=[count, self.n_dofs],
            inputs=[
                self.active_dof_indices,
                self.active_dof_to_coord,
                self.coord_masks,
                start_idx,
                self._weight,
            ],
            outputs=[jacobian],
            device=self.device,
        )
