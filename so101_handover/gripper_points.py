import math
import os
import struct
import xml.etree.ElementTree as ET

import numpy as np


def load_stl(path: str) -> np.ndarray:
    # STL の三角形を読み込み (n, 3, 3)
    with open(path, "rb") as f:
        data = f.read()
    if data[:5] == b"solid" and b"facet" in data[:512]:
        verts = [
            [float(v) for v in line.split()[1:4]]
            for line in data.decode(errors="ignore").splitlines()
            if line.strip().startswith("vertex")
        ]
        return np.array(verts).reshape(-1, 3, 3)
    count = struct.unpack("<I", data[80:84])[0]
    dtype = np.dtype([("normal", "<3f4"), ("v", "<9f4"), ("attr", "<u2")])
    tri = np.frombuffer(data[84:84 + count * 50], dtype=dtype)
    return tri["v"].reshape(-1, 3, 3).astype(np.float64)


def _rpy_matrix(r: float, p: float, y: float) -> np.ndarray:
    # rpy を回転行列に変換
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def sample_surface(tri: np.ndarray, count: int, seed: int = 0) -> np.ndarray:
    # 面積に比例して表面の点を抽出
    rng = np.random.default_rng(seed)
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=-1)
    idx = rng.choice(len(tri), size=count, p=area / area.sum())
    u, v = rng.random(count), rng.random(count)
    flip = u + v > 1.0
    u[flip], v[flip] = 1.0 - u[flip], 1.0 - v[flip]
    t = tri[idx]
    return t[:, 0] + (t[:, 1] - t[:, 0]) * u[:, None] + (t[:, 2] - t[:, 0]) * v[:, None]


def voxel_down(points: np.ndarray, size: float) -> np.ndarray:
    # 格子ごとに 1 点へ間引き
    keys = np.floor(points / size).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(idx)]


def link_points(urdf_path: str, link: str, voxel: float = 0.005, samples: int = 20000) -> np.ndarray:
    # リンクの見た目の表面点をリンク座標系で取得
    root = ET.parse(urdf_path).getroot()
    base = os.path.dirname(urdf_path)
    points = []
    for node in root.findall("link"):
        if node.get("name") != link:
            continue
        for visual in node.findall("visual"):
            mesh = visual.find("geometry/mesh")
            if mesh is None:
                continue
            origin = visual.find("origin")
            xyz = np.array([float(v) for v in origin.get("xyz", "0 0 0").split()]) if origin is not None else np.zeros(3)
            rpy = [float(v) for v in origin.get("rpy", "0 0 0").split()] if origin is not None else [0.0, 0.0, 0.0]
            tri = load_stl(os.path.join(base, mesh.get("filename")))
            pts = sample_surface(tri, samples)
            points.append(pts @ _rpy_matrix(*rpy).T + xyz)
    return voxel_down(np.concatenate(points, axis=0), voxel)


def _axis_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    # 軸と角度を回転行列に変換
    x, y, z = axis / np.linalg.norm(axis)
    c, s, t = math.cos(angle), math.sin(angle), 1.0 - math.cos(angle)
    return np.array([
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
    ])


def child_points(urdf_path: str, joint_name: str, angle: float, voxel: float = 0.005) -> np.ndarray:
    # 関節角を指定して子リンクの表面点を親リンク座標系で取得
    root = ET.parse(urdf_path).getroot()
    for joint in root.findall("joint"):
        if joint.get("name") != joint_name:
            continue
        origin = joint.find("origin")
        xyz = np.array([float(v) for v in origin.get("xyz", "0 0 0").split()])
        rpy = [float(v) for v in origin.get("rpy", "0 0 0").split()]
        axis = np.array([float(v) for v in joint.find("axis").get("xyz").split()])
        rot = _rpy_matrix(*rpy) @ _axis_matrix(axis, angle)
        pts = link_points(urdf_path, joint.find("child").get("link"), voxel)
        return pts @ rot.T + xyz
    raise ValueError(joint_name)
