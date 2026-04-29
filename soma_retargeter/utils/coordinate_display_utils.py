# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class CoordinateRow:
    name: str
    position: tuple[float, float, float]
    is_global: bool


def _to_tuple3(position: Sequence[float]) -> tuple[float, float, float]:
    return tuple(round(float(position[i]), 6) for i in range(3))


def build_coordinate_rows(
    names: Sequence[str],
    positions: np.ndarray | Iterable[Sequence[float]],
    root_name: str,
) -> list[CoordinateRow]:
    """Build display rows where the root is global and all other rows are root-relative."""
    positions_np = np.asarray(positions, dtype=np.float32)
    if len(names) == 0:
        return []
    if positions_np.shape[0] != len(names):
        raise ValueError(
            f"Position count [{positions_np.shape[0]}] does not match name count [{len(names)}]."
        )

    try:
        root_idx = names.index(root_name)
    except ValueError:
        root_idx = 0

    root_position = positions_np[root_idx]
    rows = []
    for idx, name in enumerate(names):
        is_global = idx == root_idx
        position = positions_np[idx] if is_global else positions_np[idx] - root_position
        rows.append(CoordinateRow(name, _to_tuple3(position), is_global))

    return rows


def format_coordinate_row(row: CoordinateRow) -> str:
    scope = "gbl" if row.is_global else "rel"
    x, y, z = row.position
    return f"{row.name:<12s} {scope}  ({x: .3f}, {y: .3f}, {z: .3f})"
