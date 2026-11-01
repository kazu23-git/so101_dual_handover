from __future__ import annotations

import torch

from .ik import KinematicChain


class BatchArm:
    # URDF の運動学を全環境まとめて計算

    def __init__(self, urdf_path: str, end_link: str, joint_names: list[str],
                 tcp_offset: tuple[float, float, float], device: str):
        chain = KinematicChain(urdf_path, end_link, joint_names)
        self.num_joints = len(joint_names)
        self.device = device
        self.joints = []
        for joint, idx in zip(chain.chain, chain.q_index):
            ax, ay, az = (float(v) for v in joint["axis"])
            skew = torch.tensor([[0.0, -az, ay], [az, 0.0, -ax], [-ay, ax, 0.0]], device=device)
            self.joints.append({
                "xyz": torch.tensor(joint["xyz"], dtype=torch.float32, device=device),
                "rot": torch.tensor(joint["rot"], dtype=torch.float32, device=device),
                "skew": skew,
                "skew2": skew @ skew,
                "index": idx,
            })
        self.tcp_offset = torch.tensor(tcp_offset, dtype=torch.float32, device=device)
        self.eye3 = torch.eye(3, device=device)

    def fk(self, q: torch.Tensor):
        # TCP の位置と姿勢を計算（土台座標系）
        n = q.shape[0]
        rot = self.eye3.expand(n, 3, 3)
        pos = torch.zeros(n, 3, device=self.device)
        for joint in self.joints:
            pos = pos + rot @ joint["xyz"]
            rot = rot @ joint["rot"]
            if joint["index"] >= 0:
                th = q[:, joint["index"]].view(n, 1, 1)
                rot = rot @ (self.eye3 + torch.sin(th) * joint["skew"] + (1.0 - torch.cos(th)) * joint["skew2"])
        return pos + rot @ self.tcp_offset, rot

    def step(self, q: torch.Tensor, target_pos: torch.Tensor, target_y: torch.Tensor | None = None,
             target_z: torch.Tensor | None = None,
             rot_weight: float = 0.2, damping: float = 0.02, eps: float = 1e-3) -> torch.Tensor:
        # 減衰最小二乗法で 1 回更新（位置と、必要なら爪の幅の向きと手先の向き）
        pos, rot = self.fk(q)
        y = rot[:, :, 1]
        z = rot[:, :, 2]
        cols = []
        for i in range(self.num_joints):
            dq = torch.zeros_like(q)
            dq[:, i] = eps
            pos2, rot2 = self.fk(q + dq)
            col = [(pos2 - pos) / eps]
            if target_y is not None:
                col.append(rot_weight * torch.cross(y, (rot2[:, :, 1] - y) / eps, dim=-1))
            if target_z is not None:
                col.append(rot_weight * torch.cross(z, (rot2[:, :, 2] - z) / eps, dim=-1))
            cols.append(torch.cat(col, dim=-1))
        jac = torch.stack(cols, dim=-1)
        err = target_pos - pos
        if target_y is not None:
            sign = torch.where(torch.sum(y * target_y, dim=-1, keepdim=True) >= 0, 1.0, -1.0)
            err = torch.cat([err, rot_weight * torch.cross(y, sign * target_y, dim=-1)], dim=-1)
        if target_z is not None:
            err = torch.cat([err, rot_weight * torch.cross(z, target_z, dim=-1)], dim=-1)
        rows = jac.shape[1]
        a = jac @ jac.transpose(1, 2) + (damping**2) * torch.eye(rows, device=self.device)
        sol = torch.linalg.solve(a, err.unsqueeze(-1))
        return q + (jac.transpose(1, 2) @ sol).squeeze(-1)
