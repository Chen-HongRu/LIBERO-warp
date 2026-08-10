# LIBERO-warp 多代理路线图与续接交接

更新日期：2026-08-10（Asia/Shanghai）

这份文件是新 Codex 对话的续接入口。它同时记录原始执行路线图、已经完成的工作、当前本地与 AutoDL 状态、下一步命令和验收门槛。

## 新对话如何开始

在新对话中请先要求主代理完整阅读本文件，然后继续执行，不要重新规划或重做已完成工作。建议首条消息：

> 请完整阅读 `/Users/chenhongru/workspace/LIBERO-warp/docs/CODEX_LIBERO_WARP_HANDOFF.md`，按其中的多代理组织、当前状态和下一步命令继续。使用 Terra High 子代理；你只负责指挥、设计、审查和 Git。先完成 AutoDL M1 GPU 正确性与性能门槛，不要重做 M0，也不要提前进入 M2。

新对话应重新创建三个 Terra High 子代理：

| 代理 | 职责 | 主要所有权 |
|---|---|---|
| `foundation_runtime` | 依赖、公共接口、TaskCompiler、状态/reset | runtime 与 packaging |
| `warp_systems` | MJWarp、renderer、Torch bridge、GPU OSC、predicates | GPU backend |
| `verification_review` | official/Warp 测试、ctrl trace、benchmark、独立审查 | tests 与 benchmarks |

创建参数：

```text
model: gpt-5.6-terra
reasoning_effort: high
fork_turns: "3"
```

协作规则：

- 最多三个子代理并行，加主代理共四个并发槽位。
- 子代理共享工作区，必须划分互不重叠的文件所有权。
- 子代理禁止切分支、commit、push、rebase；主代理是唯一执行 Git 操作的人。
- 不 push、不创建 PR，除非用户另行授权。
- Python 环境用 `uv`；任何 Python 修改后运行 `uvx ruff format` 和 `uvx ruff check`。
- 每轮由主代理审查完整 diff、运行验收并决定是否进入下一阶段。

## 仓库与 Git 状态

本地仓库：

```text
/Users/chenhongru/workspace/LIBERO-warp
```

分支：

```text
codex/libero-warp-roadmap
```

已完成的 M0 本地提交：

```text
f3d5d1850bc371c7a2920afaff9e1318fe1cb337
M0: establish MuJoCo 3 official baseline
```

M1 本轮以 correctness pass / performance no-go 收尾；提交范围主要包括：

```text
M  libero/libero/envs/env_wrapper.py
M  pyproject.toml
M  pytest.ini
M  tests/conftest.py
M  tests/test_packaging_contract.py
?? benchmarks/
?? libero/libero/runtime/
?? libero/libero/warp/
?? tests/test_mjwarp_gpu_parity.py
?? tests/test_runtime_api.py
?? tests/test_task_compiler.py
?? tests/test_warp_spike_contract.py
?? docs/CODEX_LIBERO_WARP_HANDOFF.md
```

本地测试曾生成 `build/` 和 `libero_warp.egg-info/`。它们不是源码；在 M1 提交前应再次移到 Trash 或清理，并确认 `git status` 只包含有意修改。

## 里程碑状态

| 里程碑 | 状态 | 说明 |
|---|---|---|
| M0 MuJoCo 3 official baseline | 已完成并提交 | 依赖、packaging、MuJoCo 3 兼容、official 测试基线 |
| M1 MJWarp physics/render spike | GPU 正确性通过；性能 hard gate 未通过，no-go 证据已记录 | 本轮收尾 |
| M2 GPU OSC / action-exact boundary | no-go，暂停 | M1 aggregate speedup 仅 2.475x，低于 5x 停止线 |
| M3 130-task Warp | 暂停 | 等待 M1 性能问题解决并重新获得 M2 go |
| M4 发行与最终报告 | 暂停 | 等待前序里程碑恢复 |

## M0 已完成内容

- 使用 `.python-version`、`pyproject.toml`、`uv.lock`；项目发布环境固定 Python 3.11。
- 固定 robosuite 1.5.2、MuJoCo/MJWarp 3.11.0、Warp 1.16.0、Torch 2.13.0、NumPy 1.26.4。
- 删除旧 `libero/lifelong`、训练 Hydra 配置和算法 notebook；保留任务、资产、init states、benchmark、demo 和数据下载工具。
- 修复 package discovery、资源路径、CLI、HF downloader、teleop extra 和 wheel 内容。
- 导入并审查 MuJoCo 3 migration；`mujoco3_compat.py` 对版本 fail-closed。
- 修复 segmentation 显式 robot IDs、reset 仅重试 `RandomizationError`、collector BDDL 文本、random colors、seed、CLI 和残余 `init_path` 问题。
- M0 official 基线测试和近期定向 official 回归通过。

## M1 已完成的本地实现

### 公共 runtime API

文件：

```text
libero/libero/runtime/__init__.py
libero/libero/runtime/types.py
libero/libero/runtime/official.py
libero/libero/runtime/compiler.py
libero/libero/runtime/warp.py
```

已冻结合同：

- `CameraConfig`、`EnvConfig`、`ObservationBatch`、`StepBatch`、`RenderExactState`、`RuntimeEnv`。
- 相机仅由调用方按有序列表声明；无固定 view role/slot；允许不同分辨率。
- RGB：top-left、NHWC、`uint8`。
- depth：metric camera-Z meters、`float32`。
- segmentation：`int32`，末通道为 1。
- official backend 只允许 `num_worlds=1`，返回 CPU tensor。
- action 必须 `[1,7]`、real numeric、finite，并按 official 语义裁剪到 `[-1,1]`。
- success -> `terminated`；horizon -> `truncated`；默认不 auto-reset。
- public snapshot 仅 render-exact；没有 public restore 或 BoundaryState。
- `make_env(backend="warp")` 仍明确 `NotImplemented`，避免在 M2 OSC 前暴露错误 action 语义。
- seed 在模型构造前限制到 NumPy legacy RandomState 可接受的 `[0, 2**32-1]`。

### TaskCompiler 与 init bank

- 通过 official BDDL backend 构造 exact `mujoco.MjModel`。
- 编译协议是显式 official hard reset 一次，随后 `hard_reset=False`，防止 model identity 漂移。
- metadata 保存 task/version/timestep/decimation、nq/nv/nu/na 和所有 body/site/geom/joint/camera/actuator ID。
- init states 通过 official set-state 路径和 MuJoCo state API 解码，不猜 flattened offset。
- canonical bank 为 CPU float64；`.to(device,dtype)` 显式一次上传；gather/scatter/reset 要求 device-local，禁止隐式 CPU->CUDA。
- seed=0 的 MJB、portable arrays 和全局 NumPy RNG 隔离已验证。

### MJWarp spike

文件：

```text
libero/libero/warp/__init__.py
libero/libero/warp/spike.py
```

已实现：

- `mjw.put_model`、`mjw.make_data(nworld=N)`。
- FULLPHYSICS 全量/partial reset 和 exact trace reset。
- actuator `ctrl` replay，每个 control boundary 25 个 physics substeps。
- CUDA zero-copy readout：qpos/qvel/act/time/body_xpos/body_xquat/site_xpos/site_xmat。
- caller-ordered batched renderer，任意 `CameraConfig`，BVH refit。
- renderer 显式只启用 geom group 1、2，与 official offscreen `vopt.geomgroup=[0,1,1,0,0,0]` 对齐，避免 group 0 半透明 collision proxy 遮挡 visual mesh。
- 预分配 Warp kernel 将 packed framebuffer 转为 NHWC `uint8`，不做每步 CPU RGB/state 回传。
- metric planar camera-Z depth。
- MJWarp segmentation `(geom_id, ObjType)` 正确解码；background/non-geom 为 -1；instance/class mapping 对齐 robosuite。
- unknown camera、NaN ctrl、overflow 和 stale/closed model 显式失败。
- `close()` 幂等并释放 CUDA/model/render/official references；所有 public GPU 操作 closed 后 fail-closed。
- init bank 在 GPU 侧只保留 M1 所需 FULLPHYSICS，避免重复常驻。

最新版本门禁：`runtime/warp.py::_validate_trace_metadata` 已要求 trace 的 MuJoCo/robosuite 版本等于 compiled metadata；对应 static test 已修正并通过。

### Ctrl trace

真实成功 demo：

```text
/tmp/libero-m1-demo/libero_spatial/
pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5
```

本地 v2 trace：

```text
/tmp/libero-m1-pilot-ctrl-trace-v2.npz
SHA256 7b60b3f7f01c05f6afe2330dc2dc7c35bed947ef12ee6fc8e88266b36df7b040
```

内容：

- 98 actions，25 physics substeps/action，FULLPHYSICS state size 92。
- CPU raw ctrl replay max abs error 0.0。
- reset protocol：`official-env-explicit-reset-once-v1`。
- exact env：libero_spatial task0、agentview 64x64、horizon1000、control20、seed0。
- MJB SHA：`08c0c0fb589c8ffb6309d41714cd8d820d234ad6ca1acfbb6041da0463bb8876`，268620526 bytes。
- portable arrays v2 SHA：`5ead00a6f942c73b12b36d3ee663348ec05376b426de586a1127be476e103a1c`，471 fields，234683998 bytes。
- MuJoCo 3.11.0、robosuite 1.5.2、timestep 0.002。

`benchmarks/ctrl_trace.py` 是 pickle-free、versioned NPZ 格式；当前 `TRACE_FORMAT_VERSION=2`。portable fingerprint 排除 host path offsets、BVH 和 compiler-derived mesh normals，并对跨 Darwin/Linux 的浮点编译舍入做 canonicalization；Mac 与 AutoDL 的 v2 SHA 已精确一致。loader 拒绝旧 v1、string/complex/object/NaN/Inf，并按 model metadata 严格校验 ctrl/qpos/qvel/body/site shape。

### 测试与 benchmark

新增：

```text
tests/test_runtime_api.py
tests/test_task_compiler.py
tests/test_warp_spike_contract.py
tests/test_mjwarp_gpu_parity.py
benchmarks/ctrl_trace.py
benchmarks/benchmark_official_spike.py
benchmarks/benchmark_warp_spike.py
benchmarks/compare_m1_reports.py
```

Benchmark 固定：

- worlds：1、16、64、128；256 单独 record-only。
- profile：agentview、robot0_eye_in_hand、sideview，三路 128x128 RGB。
- warm-up 100 operations；measure 1000 operations。
- modes：physics-only、render-only、step+render、partial-reset。
- 每个 mode 前恢复 exact trace initial state/cursor。
- CUDA timing 边界同步；health check 在计时外。
- comparator fail-closed 校验同 hostname/platform、同 trace/env/fingerprint、MuJoCo/robosuite 版本、100/1000、25 substeps、3 cameras、elapsed 与 throughput 等式。

M1 性能门槛：

- Warp N=128 `batch_hz`（也即 per-world control Hz）>= 20。
- `warp_128_aggregate_world_hz / official_single_world_hz >= 10x`。
- 若 aggregate speedup < 5x，M2 OSC 为 no-go，暂停后续实现并报告。
- N=256 只记录，不阻塞。

## 最新本地验证证据

最近一次主代理验证：

```text
uvx ruff format --check <全部本轮修改 Python>
uvx ruff check <全部本轮修改 Python>
# 通过

pytest -m static
# 197 passed, 1 skipped, 161 deselected
# 唯一 skip：本地无 CUDA；CPU/meta device mismatch 已覆盖

tests/test_warp_spike_contract.py -m static
# 25 passed, 1 deselected
```

AutoDL GPU parity 最终结果：

```text
tests/test_mjwarp_gpu_parity.py
# 14 passed in 69.90s
```

三相机 RGB/segmentation/keypoint/IoU 继续以 official offscreen 输出为准；depth 使用同一 official visible geom groups 的 MuJoCo `mj_ray` 全 target-pixel oracle。复杂 mesh 自遮挡处 CPU `mj_ray` 与 MJWarp ray depth 在约 `5e-7 m` 内一致，而 OpenGL raster depth 会相差厘米，因此 GL depth 只报告、不作为 ray renderer 的错误 oracle。

近期定向 official/runtime/compiler/packaging integration 由主代理和 verifier 多次通过；真实 official trace replay 和完整 official `step+render` benchmark 也通过。

一次最新的本地全 `official_integration` 重跑在 9 个测试后卡进 macOS GLFW/Cocoa event loop；已精确终止 PID。sample 显示阻塞在 `glfwInit -> NSApplication run`，属于 macOS headless backend 限制，不是断言失败。不要把它误报为代码回归；AutoDL 使用 EGL。

注意：无路径的全仓 Ruff 仍会命中大量 upstream legacy 文件（最近清理生成物后约 23 个需 format、813 lint errors）。本轮约定是所有修改文件 Ruff clean；不要在 M1 GPU 验收期间大规模格式化 legacy 源码。M4 再决定全仓基线策略。

## AutoDL 当前状态

Host：

```text
autodl-multi-gpu
```

GPU：RTX 4090 D，约 24 GB；driver 580.76.05；EGL 可用。

远端仓库：

```text
/root/autodl-tmp/LIBERO-warp-validation
branch: codex/libero-warp-roadmap
```

M1 scoped source/tests 已同步到远端 dirty worktree，但没有任何 Git 操作。

### 复用环境

为遵循“尽可能复用 AutoDL 环境”，不再全量安装锁文件。当前隔离 overlay：

```text
/root/autodl-tmp/libero-m1-overlay-py310
```

它复用 `/root/LeWM/.venv` 的 Python 3.10 / Torch CUDA，同时在 overlay 中覆盖 exact physics/model packages：

```text
Python 3.10.12
Torch 2.8.0+cu128
torch.cuda.is_available() == True
MuJoCo 3.11.0
mujoco-warp 3.11.0
warp-lang 1.16.0
robosuite 1.5.2
```

这是一项明确记录的 GPU-host variance；项目发布锁仍是 Python 3.11 / Torch 2.13。MuJoCo、MJWarp、Warp、robosuite 保持 exact，因为它们影响 trace/model/physics。Warp 已在 4090 初始化为 `cuda:0`；真实 EGL 64x64 RGB 渲染成功。

overlay 必须使用以下 PYTHONPATH 才能复用 Torch 和 LeWM 依赖：

```bash
export REPO=/root/autodl-tmp/LIBERO-warp-validation
export OVERLAY=/root/autodl-tmp/libero-m1-overlay-py310
export LEWM_SITE=/root/LeWM/.venv/lib/python3.10/site-packages
export PYTHONPATH="$REPO:$OVERLAY/lib/python3.10/site-packages:$LEWM_SITE"
export MUJOCO_GL=egl
```

### 远端 trace

正确、可使用的 v2 trace：

```text
/root/autodl-tmp/LIBERO-warp-validation/libero-m1-pilot-ctrl-trace-v2.npz
SHA256 7b60b3f7f01c05f6afe2330dc2dc7c35bed947ef12ee6fc8e88266b36df7b040
```

不要使用旧 v1 trace 或下面的旧 transfer 副本；v2 loader 会 fail-closed 拒绝格式版本 1，且 transfer 副本 SHA 不匹配：

```text
/root/autodl-tmp/libero-warp-transfer/libero-m1-pilot-ctrl-trace.npz
SHA256 6638fc07d37a64d9ae85a0b2121a6f9bc636e026151a0146585d6f2eac24ec5e
```

### 远端待清理项

macOS 传输产生了未跟踪 AppleDouble 文件：`._*`。GPU 测试前先只读列出，再精确删除这些传输元数据；不要删除正常源码或正确 trace。

示例：

```bash
find /root/autodl-tmp/LIBERO-warp-validation -name '._*' -type f -print
find /root/autodl-tmp/LIBERO-warp-validation -name '._*' -type f -delete
```

AppleDouble 文件已精确清理。GPU parity、N<=128 正式 benchmark 和 N=256 record-only 均已结束；当前没有 GPU pytest 或 benchmark 进程在运行。

## 本轮 AutoDL 执行结果与复现顺序

### 1. 清理并核对远端同步

```bash
ssh -o ClearAllForwardings=yes -S none autodl-multi-gpu
cd /root/autodl-tmp/LIBERO-warp-validation
find . -name '._*' -type f -print
# 核对后删除上述 AppleDouble 文件
sha256sum libero-m1-pilot-ctrl-trace.npz
git status --short --branch
```

期望 v2 trace SHA 为 `7b60b3f7...df7b040`。

### 2. 环境与 import smoke

```bash
cd /root/autodl-tmp/LIBERO-warp-validation
export OVERLAY=/root/autodl-tmp/libero-m1-overlay-py310
export LEWM_SITE=/root/LeWM/.venv/lib/python3.10/site-packages
export PYTHONPATH="$PWD:$OVERLAY/lib/python3.10/site-packages:$LEWM_SITE"
export MUJOCO_GL=egl

$OVERLAY/bin/python -c 'import torch, mujoco, mujoco_warp, warp, robosuite; print(torch.__version__, torch.cuda.is_available(), mujoco.__version__, warp.__version__, robosuite.__version__)'
nvidia-smi
```

### 3. 远端窄 static

至少运行版本/trace/Warp contract 的 static tests。若 Python 3.10 因某个纯 static packaging test 缺 `tomllib`，不要因此更换 GPU 环境；只运行 GPU 相关 contract，或补最小 `tomli` compatibility，但不要升级 Torch。

```bash
$OVERLAY/bin/python -m pytest tests/test_warp_spike_contract.py -m static -q
```

### 4. GPU 正确性，按节点逐步跑

共同环境：

```bash
export LIBERO_RUN_WARP_GPU=1
export LIBERO_M1_CTRL_TRACE="$PWD/libero-m1-pilot-ctrl-trace-v2.npz"
export MUJOCO_GL=egl
```

顺序：

1. `test_mjwarp_trace_initial_state_reset_matches_official_at_one_e6`
2. `test_mjwarp_first_control_and_contact_drift_match_the_official_trace`
3. `test_mjwarp_three_camera_geometry_and_element_silhouettes_match_official`
4. `test_mjwarp_readout_and_render_stay_on_cuda_without_host_outputs`
5. `test_mjwarp_rejects_unknown_camera_nonfinite_ctrl_and_overflow`
6. `test_mjwarp_close_is_idempotent_and_invalidates_public_cuda_operations`
7. `test_mjwarp_batch_reset_and_one_control_step_smoke`（N=1、16）

最后跑完整文件：

```bash
$OVERLAY/bin/python -m pytest tests/test_mjwarp_gpu_parity.py -q
```

正确性硬门槛已全部通过：

- reset qpos/qvel/act max error <= 1e-6。
- 一个 control step：robot qpos <= 1e-3 rad；body/site position <= 2 mm；orientation <= 0.5 degree。
- target bowl keypoint <= 1 pixel；silhouette IoU >= 0.95。
- metric target depth max abs error <= 5 mm。
- 10/50-step contact drift 只报告。
- 无 NaN、无静默 overflow、无 CPU output/隐式替代 backend。

最终完整 GPU 文件为 `14 passed in 69.90s`。JUnit 证据保存在 `/root/autodl-tmp/libero-m1-gpu-parity.xml`。

任何失败都由 `warp_systems` 代理在 owned source 中窄修；tests 由 `verification_review` 独立审查。每次 Python 修改后 Ruff。

### 5. 同机 official/Warp benchmark

只有 GPU 正确性全部通过后才运行。

先生成 AutoDL 同机 official baseline：

```bash
$OVERLAY/bin/python -m benchmarks.benchmark_official_spike \
  "$LIBERO_M1_CTRL_TRACE" \
  --modes step+render \
  --output /root/autodl-tmp/libero-m1-official-autodl.json
```

Warp N<=128 正式报告：

```bash
$OVERLAY/bin/python -m benchmarks.benchmark_warp_spike \
  --worlds 1 16 64 128 \
  --modes physics-only render-only step+render partial-reset \
  --output /root/autodl-tmp/libero-m1-warp-autodl.json
```

比较：

```bash
$OVERLAY/bin/python -m benchmarks.compare_m1_reports \
  /root/autodl-tmp/libero-m1-official-autodl.json \
  /root/autodl-tmp/libero-m1-warp-autodl.json \
  --output /root/autodl-tmp/libero-m1-verdict-autodl.json
```

N=256 单独运行并单独保存，失败或 OOM 不得覆盖 N=128 的门槛报告：

```bash
$OVERLAY/bin/python -m benchmarks.benchmark_warp_spike \
  --worlds 256 \
  --modes physics-only render-only step+render partial-reset \
  --output /root/autodl-tmp/libero-m1-warp-256-record-only.json
```

N<=128 正式报告已完成（100 warm-up / 1000 measured，三相机 128x128 RGB，25 physics substeps）：

| worlds | physics-only Hz | render-only Hz | step+render Hz | aggregate step+render world-Hz |
|---:|---:|---:|---:|---:|
| 1 | 2.923 | 25.733 | 2.482 | 2.482 |
| 16 | 2.678 | 23.763 | 2.349 | 37.583 |
| 64 | 2.518 | 22.887 | 2.326 | 148.838 |
| 128 | 2.595 | 24.705 | 2.323 | 297.310 |

N=256 record-only 也完成且未 OOM：physics-only `2.593 Hz`、render-only `26.430 Hz`、step+render `2.198 Hz`（aggregate `562.750 world-Hz`）、partial-reset `38.259 Hz`。它不参与 hard gate。

同机 official N=1 `step+render` 为 `120.139 Hz`。Comparator 结论：

```text
warp_128_per_world_hz = 2.322733
warp_128_aggregate_world_hz = 297.309828
aggregate_speedup = 2.474721x
per_world_20hz_passed = false
aggregate_10x_passed = false
hard_gate_passed = false
go_no_go_5x = no-go
```

因此不要进入 M2。AutoDL JSON/JUnit 证据已回传到本地 `/tmp/libero-m1-autodl-evidence-20260810/`；后续若要恢复路线，必须先单独设计并验证 M1 physics/kernel-launch 性能优化。

### 6. M1 审查与提交

M1 go/no-go 生成后：

- 把远端 JSON/pytest 证据复制回本地非源码临时目录。
- 邀请 `verification_review` 做最终只读审查。
- 主代理逐文件审查全部 diff。
- 清理本地 `build/`、`*.egg-info/`、`__pycache__` 等生成物。
- 重跑所有本轮修改文件 Ruff、完整 static、定向 official/runtime/compiler/packaging integration。
- `git diff --check`。
- 仅主代理 stage/commit：

```text
M1: validate batched MJWarp physics and rendering spike
```

不 push、不发 PR。

## M2：GPU OSC 与 action-exact boundary（M1 go 后）

只读设计已完成，但没有实现。关键官方语义：

- Panda OSC_POSE action 为 `[delta position 3, delta axis-angle 3, gripper 1]`。
- public runtime 先 finite 检查和 clip `[-1,1]`。
- position scale +/-0.05 m；rotation scale +/-0.5 rad；base frame delta。
- kp=150、damping=1、uncoupled position/orientation。
- 20 Hz control、0.002 s physics，共 25 substeps。
- 仅第一个 substep set_goal；每个 substep 重算 controller torque。
- gripper accumulator 每 policy boundary 按 sign 更新 0.2；零 action 保持。
- 比较权威值是写入 `data.ctrl` 前经过 actuator ctrlrange clip 的 final ctrl。

`warp_systems`：

- 实现 batched GPU OSC：Jacobian、operational-space inertia、gains、bias/gravity、nullspace、actuator limits、gripper accumulator、25-substep timing。

`foundation_runtime`：

- 接入 `WarpBatchEnv.step(actions[N,7])`。
- partial reset、reward、success、horizon。
- 版本化 `BoundaryStateBatch`。

Boundary state 必须用 `mjSTATE_INTEGRATION`，不是 M1 FULLPHYSICS；还需 OSC goal/origin/initial_joint/kp/kd、gripper accumulator/goal、episode timestep/horizon、RNG、solver/controller transient 和必要 observation buffers。只允许 control boundary capture；public restore 仅在 next-action continuation parity 通过后开放。

`verification_review`：

- official final ctrl corpus：至少 10 init states，加 zero/axis/orientation/gripper/out-of-range/random actions。
- 每样本比较 25 个 substeps final arm ctrl：relative L2 <= 1%，max abs <= 0.5 Nm。
- gripper/clipping/timing exact；原始 7D demo action closed-loop replay。
- pilot Warp success 不比 official 低超过 5 个百分点。
- BoundaryState restore 后 next action parity。

M2 本地提交名：

```text
M2: add GPU OSC and verified action boundary state
```

## M3：扩展到 130 tasks

按 suite 顺序逐步推进：

1. libero_spatial、libero_object
2. libero_goal
3. libero_10
4. libero_90

`warp_systems`：

- batched `In/On/Stack/Open/Close/TurnOn/TurnOff/contact`。
- drawer、microwave、stove 等关节物体。
- per-world visual state、全部 arena/camera、capacity 自动估计。

`foundation_runtime`：

- BDDL goal -> static predicate graph。
- TaskCompiler、init-state bank、partial reset、success、metadata 扩展到所有任务。

`verification_review`：

- 每种 predicate official corpus，CPU/Warp 100% 一致。
- 每 task 至少 10 init states、100 random actions、reset/step/render/success/overflow。
- nightly 全 init states 和 demo replay。
- 同 GPU 性能下降 >15% 为回归。

M3 提交名：

```text
M3: support all 130 LIBERO tasks on Warp
```

## M4：发行与最终审查

- 安装/API/camera/backend 文档。
- 明确旧 lifelong training 移除和迁移方式。
- 不引入 LIBERO 专用数据存储 schema；数据存储仍是下游职责。
- compatibility report：130-task official、demo、predicate、camera geometry、Warp support matrix、4090/5090 benchmark。
- GPU profiler、显存、kernel launch、隐式 CPU sync/device copy 审计。
- 主代理运行最终 uv/Ruff/CPU/NVIDIA/performance suite，逐文件审查。
- 最终提交：

```text
M4: prepare LIBERO-warp research distribution
```

最终论文表述边界：

- Warp：训练与高速交互。
- Official：论文 success-rate evaluation。
- 不声称 CPU/Warp 长期接触物理轨迹完全一致。

## 禁止事项与风险边界

- M1 <5x 时不要继续实现 GPU OSC；先提交 go/no-go 报告。
- 不因 256 OOM 阻塞 N=128 门槛。
- 不使用错误 SHA 的 transfer trace。
- 不把 MJB cross-host 差异当硬失败；portable arrays + exact versions 是硬门槛。
- 不把 `RenderExactState` 当 action continuation checkpoint。
- 不在 M2 前公开 Warp 7D action runtime。
- 不静默 fallback 到 official CPU backend。
- 不由子代理执行 Git。
- 不 push、不创建 PR，除非用户明确授权。
