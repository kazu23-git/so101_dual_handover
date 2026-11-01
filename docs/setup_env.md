# conda 環境 `env_pai_handover` の作り方

## 環境

| 項目 | 版 |
|---|---|
| 環境名 | `env_pai_handover` |
| Python | 3.11 |
| PyTorch | 2.7.0 + cu128（torchvision 0.22.0） |
| Isaac Sim | 5.1.0（pip 版） |
| Isaac Lab | v2.3.2 タグ ＋ RSL-RL |
| 固定するもの | tensordict 0.14.1、psutil 5.9.8、setuptools < 81、flatdict 4.0.1 |

配置

| パス | 中身 |
|---|---|
| `~/physical_ai/so101_dual_handover` | このリポジトリ |
| `~/physical_ai/ext/IsaacLab` | Isaac Lab（このプロジェクト専用に clone） |
| `~/physical_ai/ext/SO-ARM100` | SO-101 の URDF とメッシュの取得元 |

Isaac Lab の main / develop は Isaac Sim 6.0 が前提なので、必ず v2.3.2 のタグを使う。

所要時間は 1〜2 時間。ディスクは 40 GB 以上を空けておく。

---

## 0. リポジトリを置く

```bash
mkdir -p ~/physical_ai
cd ~/physical_ai
git clone <このリポジトリの URL> so101_dual_handover
```

zip で受け取った場合は `unzip so101_dual_handover.zip -d ~/physical_ai` で展開する。

---

## 1. 前提の確認

```bash
lsb_release -ds                  # Ubuntu 22.04
nvidia-smi                       # ドライバ 550 系以上
df -h ~                          # 40 GB 以上
echo "ROS_DISTRO=[$ROS_DISTRO]"  # humble と出たら 2-3 も行う
conda --version
```

`~/.bashrc` は変更しない。環境変数はすべて conda の activate 時に設定する。

---

## 2. conda 環境と activate 時の設定

### 2-1. 環境を作成

```bash
conda create -n env_pai_handover python=3.11 -y
conda activate env_pai_handover
conda install -c conda-forge libstdcxx-ng -y
ls $CONDA_PREFIX/lib/libstdc++.so.6
python -m pip install --upgrade pip
mkdir -p $CONDA_PREFIX/etc/conda/activate.d $CONDA_PREFIX/etc/conda/deactivate.d
```

### 2-2. libstdc++ の設定

Isaac Sim の起動時の `CXXABI_1.3.15' not found` と `circular import` を防ぐ。

```bash
cat > $CONDA_PREFIX/etc/conda/activate.d/10_libstdcxx.sh << 'EOF'
export _PAI_OLD_LD_PRELOAD="${LD_PRELOAD-}"
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6
EOF

cat > $CONDA_PREFIX/etc/conda/deactivate.d/10_libstdcxx.sh << 'EOF'
if [ -n "${_PAI_OLD_LD_PRELOAD-}" ]; then export LD_PRELOAD="$_PAI_OLD_LD_PRELOAD"; else unset LD_PRELOAD; fi
unset _PAI_OLD_LD_PRELOAD
EOF
```

### 2-3. ROS 2 を隠す（`~/.bashrc` で ROS 2 を読み込んでいる PC だけ）

```bash
cat > $CONDA_PREFIX/etc/conda/activate.d/20_unros.sh << 'EOF'
export _UNROS_ACTIVE=1
export _UNROS_PYTHONPATH="${PYTHONPATH-}"
export _UNROS_LD_LIBRARY_PATH="${LD_LIBRARY_PATH-}"
export _UNROS_PATH="$PATH"
export _UNROS_CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH-}"
export _UNROS_AMENT_PREFIX_PATH="${AMENT_PREFIX_PATH-}"
export _UNROS_COLCON_PREFIX_PATH="${COLCON_PREFIX_PATH-}"
export _UNROS_ROS_DISTRO="${ROS_DISTRO-}"
export _UNROS_ROS_VERSION="${ROS_VERSION-}"
export _UNROS_ROS_PYTHON_VERSION="${ROS_PYTHON_VERSION-}"
_strip_ros() {
  local -a parts; local p out=""
  IFS=':' read -ra parts <<< "$1"
  for p in "${parts[@]}"; do
    [ -z "$p" ] && continue
    case "$p" in /opt/ros/*|*/ros2_ws/*) continue;; esac
    out="${out:+$out:}$p"
  done
  printf '%s' "$out"
}
export PATH="$(_strip_ros "$PATH")"
PYTHONPATH="$(_strip_ros "${PYTHONPATH-}")";               if [ -n "$PYTHONPATH" ]; then export PYTHONPATH; else unset PYTHONPATH; fi
LD_LIBRARY_PATH="$(_strip_ros "${LD_LIBRARY_PATH-}")";     if [ -n "$LD_LIBRARY_PATH" ]; then export LD_LIBRARY_PATH; else unset LD_LIBRARY_PATH; fi
CMAKE_PREFIX_PATH="$(_strip_ros "${CMAKE_PREFIX_PATH-}")"; if [ -n "$CMAKE_PREFIX_PATH" ]; then export CMAKE_PREFIX_PATH; else unset CMAKE_PREFIX_PATH; fi
unset AMENT_PREFIX_PATH COLCON_PREFIX_PATH ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION
unset -f _strip_ros
EOF

cat > $CONDA_PREFIX/etc/conda/deactivate.d/20_unros.sh << 'EOF'
if [ -n "${_UNROS_ACTIVE-}" ]; then
  export PATH="$_UNROS_PATH"
  _r() { if [ -n "$2" ]; then export "$1=$2"; else unset "$1"; fi; }
  _r PYTHONPATH "$_UNROS_PYTHONPATH"
  _r LD_LIBRARY_PATH "$_UNROS_LD_LIBRARY_PATH"
  _r CMAKE_PREFIX_PATH "$_UNROS_CMAKE_PREFIX_PATH"
  _r AMENT_PREFIX_PATH "$_UNROS_AMENT_PREFIX_PATH"
  _r COLCON_PREFIX_PATH "$_UNROS_COLCON_PREFIX_PATH"
  _r ROS_DISTRO "$_UNROS_ROS_DISTRO"
  _r ROS_VERSION "$_UNROS_ROS_VERSION"
  _r ROS_PYTHON_VERSION "$_UNROS_ROS_PYTHON_VERSION"
  unset -f _r
  unset _UNROS_ACTIVE _UNROS_PYTHONPATH _UNROS_LD_LIBRARY_PATH _UNROS_PATH _UNROS_CMAKE_PREFIX_PATH _UNROS_AMENT_PREFIX_PATH _UNROS_COLCON_PREFIX_PATH _UNROS_ROS_DISTRO _UNROS_ROS_VERSION _UNROS_ROS_PYTHON_VERSION
fi
EOF
```

### 2-4. プロジェクトの設定

置き場、EULA の承諾、このリポジトリを `PYTHONPATH` に追加する。リポジトリは pip でインストールしない。

```bash
cat > $CONDA_PREFIX/etc/conda/activate.d/30_project.sh << 'EOF'
export PAI_ROOT=$HOME/physical_ai
export PAI_EXT=$PAI_ROOT/ext
export PAI_REPO=$PAI_ROOT/so101_dual_handover
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH="$PAI_REPO${PYTHONPATH:+:$PYTHONPATH}"
EOF

cat > $CONDA_PREFIX/etc/conda/deactivate.d/30_project.sh << 'EOF'
if [ -n "${PAI_REPO-}" ]; then
  PYTHONPATH=$(printf '%s' "${PYTHONPATH-}" | tr ':' '\n' | grep -vx "$PAI_REPO" | paste -sd: -)
  if [ -n "$PYTHONPATH" ]; then export PYTHONPATH; else unset PYTHONPATH; fi
fi
unset PAI_ROOT PAI_EXT PAI_REPO OMNI_KIT_ACCEPT_EULA
EOF
```

### 2-5. 設定を確認

```bash
conda deactivate && conda activate env_pai_handover
mkdir -p $PAI_EXT
echo "PAI_REPO=[$PAI_REPO] ROS_DISTRO=[$ROS_DISTRO] LD_PRELOAD=[$LD_PRELOAD]"
echo "PYTHONPATH=[$PYTHONPATH]"
python -c "import sys; print([p for p in sys.path if '/opt/ros' in p])"
```

期待値：`PAI_REPO` はリポジトリのパス、`ROS_DISTRO` は空、`LD_PRELOAD` は環境の中の `lib/libstdc++.so.6`、`PYTHONPATH` にリポジトリのパス、最後の行は `[]`。

---

## 3. PyTorch 2.7.0 + cu128

```bash
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 期待：2.7.0+cu128 True
```

---

## 4. Isaac Sim 5.1.0（pip 版）

```bash
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
pip show isaacsim | grep Version          # Version: 5.1.0.0
isaacsim                                  # 初回はシェーダのコンパイルで数分。GUI が出たら閉じる
```

起動ログに `CXXABI_1.3.15' not found`、`circular import`、`rclpy` が出ないこと。

---

## 5. Isaac Lab v2.3.2

```bash
cd $PAI_EXT
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
git fetch --tags
git checkout v2.3.2
git describe --tags                       # v2.3.2

# flatdict を先に入れる（新しい setuptools ではビルドできない）
pip install "setuptools<81"
pip install flatdict==4.0.1 --no-build-isolation

# Isaac Lab 本体と RSL-RL を入れる
./isaaclab.sh --install rsl_rl

# 版を固定する
pip install psutil==5.9.8
pip install "tensordict==0.14.1" --no-deps
```

確認

```bash
python -c "import torch, isaaclab, tensordict; print(torch.__version__, isaaclab.__file__, tensordict.__version__)"
# 期待：2.7.0+cu128 /home/<ユーザー>/physical_ai/ext/IsaacLab/source/isaaclab/isaaclab/__init__.py 0.14.1
python -c "import rsl_rl.env; print('rsl_rl OK')"
pip check
```

`pip check` で残ってよいのは次の 3 件（Isaac Sim と Isaac Lab の固定値の衝突で既知のもの）。

- fastapi ↔ starlette
- ipython ↔ psutil
- isaacsim-kernel ↔ typing_extensions

---

## 6. SO-101 のモデルを取得

```bash
cd $PAI_EXT
git clone --depth 1 https://github.com/TheRobotStudio/SO-ARM100.git
mkdir -p $PAI_REPO/assets/so101
cp -r $PAI_EXT/SO-ARM100/Simulation/SO101/* $PAI_REPO/assets/so101/
ls $PAI_REPO/assets/so101/so101_new_calib.urdf
```

---

## 6-2. 物体検出モデルの実行環境

カメラによる認識（`build_t_ik.py --perception`）で使う。モデルの重みは初回実行時に Hugging Face から取得される（約 600 MB）。

```bash
pip install "transformers==4.46.3"
python -c "import torch, transformers; print(torch.__version__, transformers.__version__)"
# 期待：2.7.0+cu128 4.46.3（torch が置き換わっていないこと）
```

---

## 7. 動作確認

```bash
conda activate env_pai_handover
cd $PAI_REPO

# (1) パッケージが読めるか
python -c "import so101_handover; print(so101_handover.TASK_NAME)"   # SO101-Dual-Handover-v0

# (2) シーンが表示されるか
python scripts/check_scene.py --num_envs 4

# (3) 学習が始まるか（20 回だけ）
python scripts/train.py --headless --num_envs 1024 --max_iterations 20
```

`Learning iteration 1/20 ...` が流れ、`logs/rsl_rl/so101_handover/<日時>/` に `model_*.pt` ができれば完成。
