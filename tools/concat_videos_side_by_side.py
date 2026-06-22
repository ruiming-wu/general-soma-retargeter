#!/usr/bin/env python3
"""Concatenate two videos horizontally for before/after comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--left-label", default="before")
    parser.add_argument("--right-label", default="after")
    return parser.parse_args()


def _read_frame(cap: cv2.VideoCapture) -> np.ndarray | None:
    ok, frame = cap.read()
    if not ok:
        return None
    return frame


def _resize_to_height(frame: np.ndarray, height: int) -> np.ndarray:
    if frame.shape[0] == height:
        return frame
    scale = height / float(frame.shape[0])
    width = max(1, int(round(frame.shape[1] * scale)))
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def _put_label(frame: np.ndarray, label: str) -> np.ndarray:
    out = frame.copy()
    cv2.rectangle(out, (12, 12), (12 + 16 * len(label), 48), (0, 0, 0), -1)
    cv2.putText(out, label, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def main() -> None:
    args = parse_args()
    left = cv2.VideoCapture(str(args.left.expanduser()))
    right = cv2.VideoCapture(str(args.right.expanduser()))
    if not left.isOpened():
        raise FileNotFoundError(args.left)
    if not right.isOpened():
        raise FileNotFoundError(args.right)

    fps = left.get(cv2.CAP_PROP_FPS) or right.get(cv2.CAP_PROP_FPS) or 30.0
    first_left = _read_frame(left)
    first_right = _read_frame(right)
    if first_left is None or first_right is None:
        raise ValueError("Input videos must contain at least one frame")
    height = max(first_left.shape[0], first_right.shape[0])
    first_left = _put_label(_resize_to_height(first_left, height), args.left_label)
    first_right = _put_label(_resize_to_height(first_right, height), args.right_label)
    first = np.concatenate([first_left, first_right], axis=1)

    args.output.expanduser().parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(args.output.expanduser()), fourcc, fps, (first.shape[1], first.shape[0]))
    writer.write(first)

    while True:
        lf = _read_frame(left)
        rf = _read_frame(right)
        if lf is None or rf is None:
            break
        lf = _put_label(_resize_to_height(lf, height), args.left_label)
        rf = _put_label(_resize_to_height(rf, height), args.right_label)
        frame = np.concatenate([lf, rf], axis=1)
        if frame.shape[1] != first.shape[1] or frame.shape[0] != first.shape[0]:
            frame = cv2.resize(frame, (first.shape[1], first.shape[0]), interpolation=cv2.INTER_AREA)
        writer.write(frame)

    writer.release()
    left.release()
    right.release()
    print(args.output.expanduser())


if __name__ == "__main__":
    main()
