"""OWLv2 と深度画像によるブロックの位置・向き推定。"""

from __future__ import annotations

import math

import numpy as np
import torch
from PIL import Image, ImageDraw

from isaaclab.utils.math import quat_apply, quat_conjugate

DEFAULT_MODEL = "google/owlv2-base-patch16-ensemble"


class BlockDetector:
    def __init__(self, device: str = "cuda:0", model_name: str = DEFAULT_MODEL, threshold: float = 0.1):
        # transformers を遅延読み込み
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        self.device = device
        self.threshold = threshold
        self.processor = Owlv2Processor.from_pretrained(model_name)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_name).to(device).eval()

    @torch.no_grad()
    def detect(self, rgb: np.ndarray, query: str) -> tuple[list[float], float] | None:
        # query に最も合う枠を検出
        # 余白を足して正方形に変換
        h, w = rgb.shape[:2]
        size = max(h, w)
        square = np.full((size, size, 3), 128, dtype=np.uint8)
        square[:h, :w] = rgb
        image = Image.fromarray(square)
        inputs = self.processor(text=[[query]], images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        target = torch.tensor([[size, size]], device=self.device)
        result = self.processor.post_process_object_detection(
            outputs=outputs, threshold=self.threshold, target_sizes=target
        )[0]
        if len(result["scores"]) == 0:
            return None
        i = int(result["scores"].argmax())
        return result["boxes"][i].tolist(), result["scores"][i].item()


def pixels_to_world(
    us: torch.Tensor, vs: torch.Tensor, depth: torch.Tensor,
    intrinsic: torch.Tensor, cam_pos: torch.Tensor, cam_quat_ros: torch.Tensor,
) -> torch.Tensor:
    # 画素と深度をワールド座標に変換（ROS 規約）
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    x = (us - cx) * depth / fx
    y = (vs - cy) * depth / fy
    pts_cam = torch.stack([x, y, depth], dim=-1)
    quat = cam_quat_ros.unsqueeze(0).expand(pts_cam.shape[0], 4)
    return cam_pos.unsqueeze(0) + quat_apply(quat, pts_cam)


def world_to_pixel(
    point: torch.Tensor, intrinsic: torch.Tensor, cam_pos: torch.Tensor, cam_quat_ros: torch.Tensor
) -> tuple[float, float]:
    # ワールド座標の点を画素に変換
    p = quat_apply(quat_conjugate(cam_quat_ros).unsqueeze(0), (point - cam_pos).unsqueeze(0))[0]
    u = intrinsic[0, 0] * p[0] / p[2] + intrinsic[0, 2]
    v = intrinsic[1, 1] * p[1] / p[2] + intrinsic[1, 2]
    return u.item(), v.item()


def box_points(
    box: list[float], depth: torch.Tensor, intrinsic: torch.Tensor,
    cam_pos: torch.Tensor, cam_quat_ros: torch.Tensor,
) -> torch.Tensor:
    # 枠内の画素を点群に変換
    h, w = depth.shape
    x0, y0, x1, y1 = box
    x0, x1 = max(0, int(math.floor(x0))), min(w, int(math.ceil(x1)))
    y0, y1 = max(0, int(math.floor(y0))), min(h, int(math.ceil(y1)))
    if x1 <= x0 or y1 <= y0:
        return torch.zeros(0, 3, device=depth.device)
    vs, us = torch.meshgrid(
        torch.arange(y0, y1, device=depth.device, dtype=torch.float32),
        torch.arange(x0, x1, device=depth.device, dtype=torch.float32),
        indexing="ij",
    )
    z = depth[y0:y1, x0:x1]
    valid = torch.isfinite(z) & (z > 0)
    return pixels_to_world(us[valid], vs[valid], z[valid], intrinsic, cam_pos, cam_quat_ros)


def estimate_block_pose(
    box: list[float], depth: torch.Tensor, intrinsic: torch.Tensor,
    cam_pos: torch.Tensor, cam_quat_ros: torch.Tensor,
    top_z_range: tuple[float, float], half_height: float, length: float, occlusion_tol: float,
    max_radius: float = 0.045, min_points: int = 30,
) -> tuple[torch.Tensor, float, int, bool] | None:
    # 上面の点群から中心と yaw を推定
    all_pts = box_points(box, depth, intrinsic, cam_pos, cam_quat_ros)
    # 上面の高さの点を抽出
    keep = (all_pts[:, 2] > top_z_range[0]) & (all_pts[:, 2] < top_z_range[1])
    pts = all_pts[keep]
    if pts.shape[0] < min_points:
        return None
    # 中央値から遠い点を除去
    med = pts[:, :2].median(dim=0).values
    pts = pts[(pts[:, :2] - med).norm(dim=-1) < max_radius]
    if pts.shape[0] < min_points:
        return None

    # 主成分分析で長手方向を計算
    mean = pts.mean(dim=0)
    xy = pts[:, :2] - mean[:2]
    cov = xy.T @ xy / xy.shape[0]
    _, vecs = torch.linalg.eigh(cov)
    axis = vecs[:, -1]
    perp = torch.stack([-axis[1], axis[0]])
    yaw = math.atan2(axis[1].item(), axis[0].item())

    # 点の範囲の中央を計算
    q = torch.tensor([0.01, 0.99], device=pts.device)
    s_lo, s_hi = torch.quantile(xy @ axis, q).tolist()
    t_lo, t_hi = torch.quantile(xy @ perp, q).tolist()
    s_mid, t_mid = (s_lo + s_hi) / 2, (t_lo + t_hi) / 2

    # 片側が隠れているときは見えている端から中心を補正
    occluded = s_hi - s_lo < length - occlusion_tol
    if occluded:
        above = all_pts[all_pts[:, 2] >= top_z_range[1]]
        s_above = (above[:, :2] - mean[:2]) @ axis
        if (s_above > s_mid).sum() >= (s_above < s_mid).sum():
            s_mid = s_lo + length / 2
        else:
            s_mid = s_hi - length / 2

    center = mean.clone()
    center[:2] = mean[:2] + axis * s_mid + perp * t_mid
    center[2] = center[2] - half_height
    return center, yaw, pts.shape[0], occluded


def yaw_to_quat(yaw: float, device) -> torch.Tensor:
    # yaw をクォータニオン (w, x, y, z) に変換
    return torch.tensor([[math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]], device=device)


def save_detection_image(
    path: str, rgb: np.ndarray, box: list[float] | None, score: float, label: str,
    center: torch.Tensor | None, yaw: float, intrinsic: torch.Tensor,
    cam_pos: torch.Tensor, cam_quat_ros: torch.Tensor, half_length: float,
):
    # 検出結果の画像を保存
    image = Image.fromarray(rgb).convert("RGB")
    draw = ImageDraw.Draw(image)
    if box is not None:
        draw.rectangle(box, outline=(255, 255, 0), width=2)
        draw.text((box[0] + 2, box[1] - 12), f"{label} {score:.2f}", fill=(255, 255, 0))
    if center is not None:
        axis = torch.tensor([math.cos(yaw), math.sin(yaw), 0.0], device=center.device)
        top = center.clone()
        p0 = world_to_pixel(top - axis * half_length, intrinsic, cam_pos, cam_quat_ros)
        p1 = world_to_pixel(top + axis * half_length, intrinsic, cam_pos, cam_quat_ros)
        pc = world_to_pixel(top, intrinsic, cam_pos, cam_quat_ros)
        draw.line([p0, p1], fill=(0, 255, 0), width=3)
        draw.ellipse([pc[0] - 4, pc[1] - 4, pc[0] + 4, pc[1] + 4], outline=(0, 255, 0), width=2)
    image.save(path)
