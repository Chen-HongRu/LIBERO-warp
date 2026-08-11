# LIBERO-Warp 通用兼容发行版实施计划

状态：`approved direction / implementation pending`

更新日期：2026-08-11（Asia/Shanghai）

目标仓库：`/Users/chenhongru/workspace/LIBERO-warp`

## 0. 新 thread 的唯一入口

新 thread 开始后，先完整阅读：

1. 本文件；
2. `docs/CODEX_LIBERO_WARP_HANDOFF.md`；
3. `docs/M1_5_DATA_FACTORY_EVIDENCE_INDEX.md`（如果该文件及其对应工作仍存在）；
4. 当前 `git status --short --branch` 和最近五个提交。

建议给新 thread 的首条指令：

> 请完整阅读
> `/Users/chenhongru/workspace/LIBERO-warp/docs/LIBERO_WARP_GENERAL_COMPATIBILITY_PLAN.md`
> 和 `docs/CODEX_LIBERO_WARP_HANDOFF.md`，按前者定义的新产品边界继续。
> 目标是建设一个保留 LIBERO Python API、可独立安装、可选择 official MuJoCo
> 或 MJWarp 后端的通用 LIBERO-Warp 发行版，而不是为 MVBeliefWM、某个 collector
> 或某个固定 task/camera profile 做特异适配。先执行 G0 API 审计和 G1 official
> compatibility 基线；在兼容矩阵、测试清单和 scoped diff 经审查前，不进入 Warp
> OSC、130-task 或多卡实现。保留当前工作树中的其他修改，不 commit、不 push，除非我明确授权。

当前仓库在撰写本文件时已有未提交的 M1.5 修改。新 thread 必须把它们视为用户/其他
工作的既有内容，不得 reset、checkout、覆盖或混入兼容计划的首个改动。应先确认这些修改
是否仍在进行，再决定在当前分支上窄改，还是经用户授权创建独立 worktree/分支。

---

## 1. 产品定义

### 1.1 一句话定义

`LIBERO-Warp` 是 LIBERO 的兼容发行版：保留 LIBERO 的任务、BDDL、资产、benchmark、
环境类、动作、观测、reward、success 和常用状态 API；同一发行版内提供 official MuJoCo
和 MJWarp 两个 simulator backend，并把 GPU batching、CUDA 输出、直接分支和多 GPU
数据生产作为显式、可查询的增强能力。

### 1.2 谁是公共 API 的所有者

公共 API 的所有者是 LIBERO，而不是 MVBeliefWM、LeWM、当前 M1 `RuntimeEnv`，也不是某个
离线增广脚本。未来研究代码应继续面向如下入口编程：

```python
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

env = OffScreenRenderEnv(...)
obs = env.reset()
obs, reward, done, info = env.step(action)
env.close()
```

下游可以是：

- 原始 LIBERO evaluation / demonstration 工具；
- MVBeliefWM；
- LeWM online learning；
- 离线 Gaussian branching / data augmentation；
- 尚未出现的 policy、world model、planner、dataset 或可视化代码。

这些下游都不是 LIBERO-Warp 内部架构的依赖，也不能反过来定义其公共类型。

### 1.3 发行与导入身份

推荐采用一个统一发行版：

- distribution name：`libero-warp`；
- Python import namespace：继续为 `libero`；
- 包内同时包含 official 和 Warp backend；
- 不要求与另一个安装的 official `libero` wheel 共存；
- 所有 backend 选择都由这一发行版处理，不运行时 monkey-patch 第三方包。

安装 `libero-warp` 后，原有 `from libero...` 导入路径必须保持成立。发行 metadata 必须同时
记录：

- `libero_warp_version`；
- `libero_compat_target`（上游 commit/tag）；
- `robosuite_version`；
- `mujoco_version`；
- `mujoco_warp_version`；
- `warp_version`；
- build/platform/CUDA provenance。

### 1.4 backend 选择

默认保持 official 行为：

```python
env = OffScreenRenderEnv(...)
# 等价于 backend="official"
```

新代码可显式选择：

```python
env = OffScreenRenderEnv(..., backend="warp")
```

为了允许不改源码的旧下游做实验，可支持进程级配置：

```bash
LIBERO_SIM_BACKEND=warp python downstream_script.py
```

选择优先级固定为：

1. 构造器显式 `backend=`；
2. `LIBERO_SIM_BACKEND`；
3. 默认 `official`。

任何环境都必须暴露只读 `backend_info`，并把实际选择写入 episode/report provenance。
禁止 silent fallback：调用方要求 Warp 而该配置不支持时必须抛出明确异常，不能偷偷运行
official 后端。

---

## 2. 范围与非目标

### 2.1 本计划范围

- 保留 LIBERO 既有 Python import 和环境调用方式；
- 保留 BDDL、task discovery、init states、资产和 task predicates；
- 支持原有单环境 API；
- 支持原有 vector-env 调用语义；
- 提供 Warp-native batched 扩展；
- 支持任意调用方声明的合法相机列表、分辨率和 modality；
- 支持 LIBERO 默认 7-D `OSC_POSE` action 语义；
- 明确 render-exact state 与 action-exact continuation 的区别；
- 将底层 Warp model/physics/render/controller/state 组件设计为可复用，不硬编码 pilot task；
- correctness 优先，性能和多卡在语义门通过后推进；
- 最终覆盖所有 130 个 LIBERO tasks。

### 2.2 明确非目标

- 不把 MVBeliefWM 的 `MVObservation`、固定四视角或 HDF5 schema 放入 LIBERO 公共 API；
- 不把当前 M1 的 `ObservationBatch`/`StepBatch` 强制给现有 LIBERO 调用者；
- 不承诺所有 robosuite/MuJoCo 私有可变对象都能透明模拟；
- 不通过 CPU mirror 假装 GPU 写操作已经生效；
- 不因为吞吐高就降低 action、camera、reward、predicate 或 branching 正确性门槛；
- 不以一个 task、一个机器人、固定相机或固定分辨率宣称通用完成；
- 不在首个 vertical slice 中重写所有 130 tasks、训练栈和多 GPU 调度；
- 不把 LIBERO-Warp 扩张成未经验证的通用机器人模拟框架。

### 2.3 “通用”的精确定义

本计划中的“通用”首先表示“对不同 LIBERO 下游调用方式通用”，其次才表示底层组件可被
未来其他 MuJoCo 场景复用。发布承诺仍以 LIBERO compatibility 为边界。

底层实现不得硬编码：

- suite 或 task index；
- BDDL 文件；
- 三相机或四相机 profile；
- 128×128 分辨率；
- 某一条 demo/ctrl trace；
- 某个 collector/writer；
- 某个下游的 tensor/schema 类型。

---

## 3. 当前基线与目标差距

### 3.1 已验证资产

原 handoff 中的 M0/M1 证据继续有效：

- MuJoCo 3 official baseline；
- exact official model compilation；
- portable fingerprint v2；
- MJWarp batched FULLPHYSICS reset/step；
- caller-ordered RGB/depth/segmentation renderer；
- CUDA zero-copy readout；
- GPU correctness suite；
- M1 正式 benchmark 和 no-go 证据。

这些是底层内核证据，不等于 LIBERO API 兼容完成。

### 3.2 当前不兼容点

当前仓库仍有以下产品边界差距：

1. `libero.libero.runtime` 暴露了新的 batch 类型，但 Warp 尚未接回原 `ControlEnv`、
   `OffScreenRenderEnv` 和 `SegmentationRenderEnv`。
2. `make_env(backend="warp")` 尚未提供完整 7-D action 语义。
3. M1 使用 raw actuator ctrl replay，不等同于 LIBERO 默认 `OSC_POSE`。
4. M1 benchmark 使用固定三相机 profile，不覆盖任意相机和原环境观测字典。
5. 原 `libero/lifelong`、训练 configs 和部分 scripts 已在 M0 删除；这与“任意下游 import
   兼容”存在冲突，必须做显式决策。
6. 当前 Warp spike 没有通用 `ControlEnv` facade、完整 task predicate 生命周期、robosuite
   observable scheduling 和 action-exact branch contract。
7. 当前 `sim`/controller/private state 访问与 official 对象不是兼容层。
8. 130-task、不同 camera modality、vector-env 和真实 end-to-end collector 尚未验证。

### 3.3 既有性能 no-go 的新解释

M1 N=128 的 `20 Hz` 和 `10× aggregate` hard gate 未通过，仍是有效事实。它说明当前 spike
不能进入原路线定义的 M2 性能扩张；它不应被改写或删除。

但新产品目标加入了“LIBERO API 兼容发行版”。因此：

- G0/G1/G2 的 API 清点、official compatibility 和 backend seam 不受 M1 性能 no-go 阻塞；
- Warp OSC、batch 与多卡仍必须有新的 correctness/performance gate；
- 离线增广的主指标改为端到端有效数据吞吐，而非单 world 是否实时；
- 当前 spike 性能不能外推为四相机 RGBD、OSC、predicate、branch 和 writer 的最终性能。

---

## 4. 兼容性合同

### 4.1 Level A：发布必须保证的 LIBERO 公共 API

至少包含：

```text
libero.libero
libero.libero.benchmark
libero.libero.envs
libero.libero.envs.env_wrapper
libero.libero.envs.venv
libero.libero.utils
benchmark_scripts
scripts（发布清单中保留的 CLI）
```

以下导入必须由 static contract test 覆盖：

```python
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import (
    DummyVectorEnv,
    OffScreenRenderEnv,
    SegmentationRenderEnv,
    SubprocVectorEnv,
)
from libero.libero.envs.env_wrapper import ControlEnv, DemoRenderEnv
```

G0 必须从 compatibility target commit 自动导出完整 public module/symbol manifest，不能只以本
文件列出的少量示例为最终清单。

### 4.2 Level B：环境构造与方法合同

`ControlEnv`/`OffScreenRenderEnv` 构造参数必须保持 backward compatible，至少包括：

```text
bddl_file_name
robots
controller
gripper_types
initialization_noise
use_camera_obs
has_renderer
has_offscreen_renderer
render_camera
render_collision_mesh
render_visual_mesh
render_gpu_device_id
control_freq
horizon
ignore_done
hard_reset
camera_names
camera_heights
camera_widths
camera_depths
camera_segmentations
renderer
renderer_config
**kwargs
```

新增 `backend` 必须是可选 keyword，默认不改变旧代码行为，也不能未经处理继续传给
robosuite task constructor。

必须覆盖的方法/属性：

```text
reset
step
seed
close
check_success
set_init_state
get_sim_state
set_state
regenerate_obs_from_state
reset_from_xml_string
obj_of_interest
robots
sim（见 Level C 限制）
language_instruction
problem_name
domain_name
```

### 4.3 Level C：黑盒行为合同

对于 `num_worlds=1` 的兼容路径：

- `reset()` 返回原式 observation dict；
- `step(action)` 返回 `(obs, reward, done, info)` 四元组；
- action shape、clipping、controller update 频率与 official 一致；
- observation keys、shape、dtype、图像方向和缺失 modality 行为一致；
- reward、success、horizon、`ignore_done` 与 official 一致；
- seed/model placement/init-state 行为一致；
- camera 名称、顺序、内外参和 metric depth 定义一致；
- segmentation class/instance/element 映射一致；
- hard reset、XML reset 和 close 生命周期一致；
- 不引入自动 reset，除非原调用路径本来如此。

浮点动力学不要求跨 CPU/GPU bitwise 相同，但必须在预注册 tolerance 和 task-level 行为门内。
所有 tolerance 都要有物理含义和反例测试，不能为通过测试临时放宽。

### 4.4 Level D：事实上的私有访问

研究代码常见但不属于稳定公共合同的访问包括：

```python
env.env
env.sim
env.sim.model
env.sim.data
env.env.robots[0].controller
env.env._observables
env.env._obs_cache
```

处理策略：

1. G0 通过仓库和已知下游代码搜索，建立真实 usage corpus；
2. 高频只读访问可提供显式同步、只读 proxy；
3. proxy 必须暴露 freshness/device/provenance，不能看似 live 实则 stale；
4. 私有写操作只有在能可靠写入 GPU authoritative state 时才支持；
5. 不支持的访问 fail loudly，并给出正式替代 API；
6. official backend 保留原对象访问，用作兼容 fallback；
7. 发行说明单独列出 Level D 的 supported/unsupported matrix。

不能承诺“任何 Python 私有实现细节都兼容”。可以承诺的是：公开 API 全覆盖、高频事实 API
有清晰兼容层、其余不支持行为明确失败。

### 4.5 training/lifelong API 决策

G0 必须为 M0 删除的模块做逐项决策：

| 模块 | 推荐策略 |
|---|---|
| `libero.lifelong` 源码 | 恢复 import surface，放入 `legacy` optional extra |
| Hydra configs | 恢复作为 package data 或提供明确 migration mapping |
| 训练依赖 | 不进入核心 runtime dependencies；放入 optional extra |
| 旧 notebooks | 不作为 wheel compatibility gate，可归档 |
| scripts/CLI | 对公开、文档化入口恢复并加 smoke test |

“可 import”与“在 MuJoCo 3/新依赖下完整训练复现”是两个等级，必须分别声明。未经审计不能把
删除当成永久决定，也不能一次性把过时训练依赖重新塞回核心环境。

---

## 5. 内部架构

### 5.1 分层原则

```text
LIBERO public wrappers
  ControlEnv / OffScreenRenderEnv / SegmentationRenderEnv / VectorEnv
                          |
                  LiberoBackendSession
                 /                    \
        OfficialLiberoSession     WarpLiberoSession
                 \                    /
                  shared LIBERO task contract
       BDDL / exact model / controller / observables / predicates
                                      |
                         reusable Warp primitives
        model compiler / physics batch / renderer / state / device bridge
```

公共 wrapper 保持原名称和返回类型。backend session 是内部 seam，不要求下游 import。

### 5.2 内部 backend protocol

内部 protocol 只表达实现所需能力，不取代 LIBERO API：

```python
class LiberoBackendSession(Protocol):
    backend_info: BackendInfo

    def reset(self) -> dict[str, np.ndarray]: ...
    def step(self, action: np.ndarray) -> tuple[dict, float, bool, dict]: ...
    def seed(self, seed: int | None) -> None: ...
    def get_sim_state(self) -> np.ndarray: ...
    def set_init_state(self, state: np.ndarray) -> dict[str, np.ndarray]: ...
    def regenerate_obs_from_state(self, state: np.ndarray) -> dict[str, np.ndarray]: ...
    def reset_from_xml_string(self, xml: str) -> None: ...
    def check_success(self) -> bool: ...
    def close(self) -> None: ...
```

实际实现可以拆细，但不能让新的 batch tensor 类型泄漏到 legacy `OffScreenRenderEnv`。

### 5.3 official backend

official backend 继续由现有 robosuite/LIBERO task object 执行，作为：

- 默认 backend；
- 行为 oracle；
- 不支持的 private access fallback；
- API compatibility golden producer；
- trace/model/predicate 对照来源。

第一阶段只做结构性透传，不改变 official 数值、RNG、返回值或异常行为。

### 5.4 Warp backend

Warp backend 使用相同 BDDL/任务构造路径生成 exact model，而不是维护平行 task 定义。建议内部
组件为：

```text
WarpModelCompiler
WarpPhysicsBatch
WarpRenderer
WarpOSCController
WarpObservationEngine
WarpPredicateEngine
WarpStateCodec
WarpHealthMonitor
```

要求：

- compiler 输入以 `bddl_file_name`/exact XML/config 为主，不以 suite/task index 为唯一入口；
- camera 完全由调用方参数决定；
- controller 接受原 7-D action；
- predicate/observation 不依赖 CPU `mjData` 被错误假定为 authoritative；
- N=1 compatibility 与 N>1 native batch 共享同一个物理、控制器和渲染核心；
- model/data/device ownership 和 close 生命周期明确；
- no implicit CPU↔CUDA transfer in timed native path。

### 5.5 task predicates 和 observables

允许分两步实现：

1. correctness-first：从 Warp authoritative state 做明确 batched readout，在 CPU 运行原 predicate
   oracle；
2. performance：把已审计 predicate/observable 逐类向量化到 device。

第一步的同步成本必须记录，但比重新实现错 task semantics 更安全。第二步要逐 predicate 做
official differential tests；不能一次性将所有 task logic 翻译后只看 success rate。

### 5.6 `sim` compatibility proxy

N=1 Warp 环境可提供受限 proxy：

```text
env.sim.model  -> immutable model metadata / read-only arrays
env.sim.data   -> explicit synchronized read-only view
env.sim.forward/reset/get_state/set_state -> only if semantics are implemented
```

规则：

- proxy 类名和文档必须表明它不是原 `MjSim`；
- 每次可能同步的操作都可观测，不在 native batch timed path 隐式发生；
- mutable ndarray view 禁止返回，除非写回机制真实存在；
- private code compatibility 不得牺牲 authoritative-state 单一来源。

---

## 6. 单环境兼容路径与批量扩展

### 6.1 单环境兼容路径

```python
env = OffScreenRenderEnv(..., backend="warp")
```

要求：

- 完全保持原 NumPy/dict/tuple API；
- `action.shape == (7,)`；
- 不返回 `torch.Tensor`；
- 不新增 batch 维；
- 不要求调用方理解 CUDA stream；
- 用于旧下游、debug、online closed-loop、差分验证。

该路径允许合理的同步成本，主要目标是可替换性和语义正确。

### 6.2 原有 vector-env 兼容

`DummyVectorEnv`、`SubprocVectorEnv` 原语义继续工作。它们可以包装 official 或 N=1 Warp
环境，但这不等于 GPU-native batching。

需要测试：

- reset/step subset ids；
- async/wait 行为；
- seed 分配；
- success/state RPC；
- close/worker failure；
- heterogeneous task restriction。

### 6.3 Warp-native vector extension

在不改变旧 vector-env 返回合同的前提下新增显式入口，例如：

```python
from libero.libero.warp import WarpVectorEnv

env = WarpVectorEnv(
    bddl_file_name=...,
    num_envs=256,
    device="cuda:0",
    output="torch",       # explicit
    output_device="cuda:0",
    ...,
)
```

native API：

```text
actions          [N, 7]
reward           [N]
terminated       [N]
truncated        [N]
per-camera RGB   [N, H, W, 3]
per-camera depth [N, H, W, 1]
state/proprio    [N, ...]
```

它是增强 API，不冒充原 `OffScreenRenderEnv.step`。输出类型、device、同步边界和 ownership
必须显式。

### 6.4 多 GPU

首版多卡使用数据并行：

```text
rank 0 / GPU 0 -> independent WarpVectorEnv -> shard 0
rank 1 / GPU 1 -> independent WarpVectorEnv -> shard 1
rank 2 / GPU 2 -> independent WarpVectorEnv -> shard 2
rank 3 / GPU 3 -> independent WarpVectorEnv -> shard 3
                                      -> deterministic manifest/merge
```

规则：

- 一进程一 GPU；
- 不共享 EGL、Warp、MuJoCo context；
- seed/episode/branch ID 由全局 deterministic partition 分配；
- 不用 DDP 同步 simulator；
- 每 rank 独立、可恢复地写 shard；
- merge 验证唯一 ID、schema、provenance、row count 和 hash；
- 先过 1-card correctness，再测 2-card，最后 4-card。

---

## 7. 状态、rerender 与直接分支

### 7.1 三种能力必须分开

```text
reset/init-state exact
render-exact restore
action-exact continuation
```

`get_sim_state()/set_init_state()` 兼容不自动意味着 action continuation exact。LIBERO 默认
OSC、robot buffers、observables、task bookkeeping、RNG 和 solver transient state 可能不在 flattened
physics state 中。

### 7.2 capabilities

每个 backend 暴露只读 capability：

```python
env.backend_info.capabilities
```

至少包含：

```text
single_env_numpy_api
native_batch
cuda_observations
render_exact_restore
action_exact_continuation
partial_reset
segmentation
metric_depth
private_sim_read_proxy
```

能力必须来自真实验证结果，不能由 backend 名称推断。

### 7.3 continuation 扩展

直接分支使用版本化正式 API，例如：

```python
snapshot = env.capture_continuation_state()
env.restore_continuation_state(snapshot)
```

发布 `action_exact_continuation=True` 前必须：

- inventory 所有 state family；
- 保存 controller/gripper/robot/observable/cache/RNG/solver state；
- 恢复后执行相同下一 action；
- state、reward、success、obs 在冻结 tolerance 内；
- 验证多步 suffix，不只一帧；
- snapshot 带 schema/runtime/model/controller identity；
- 跨不匹配 runtime/model fail closed。

---

## 8. 分阶段路线图

所有阶段都以一个可审查的 scoped diff 完成。上一阶段 hard gate 未通过，不进入下一阶段。

### G0：兼容目标与 API inventory 冻结

目标：回答“到底要兼容什么”，不写 Warp 功能。

任务：

1. 冻结 `libero_compat_target` 的准确 commit/tag；
2. 从 target 自动生成 module/symbol/signature manifest；
3. 清点环境、benchmark、venv、dataset、scripts、lifelong/configs；
4. 搜索已知下游仓库对 `libero`、`.env`、`.sim`、controller/private state 的调用；
5. 建立 Level A/B/C/D compatibility matrix；
6. 为删除模块写 restore/optional/deprecated/unsupported 决策；
7. 冻结 backend selection 和 `BackendInfo` schema；
8. 给每个 unsupported private call 指定失败方式或正式替代 API。

交付物：

- `docs/LIBERO_API_COMPATIBILITY_MATRIX.md`；
- machine-readable API manifest；
- import/signature static tests；
- downstream usage report；
- 不超过一个小型 provenance/capability schema proposal。

Hard gate：

- compatibility target 可复现；
- manifest 不是人工挑选子集；
- 所有删除的 public module 有显式决策；
- 用户审查并冻结兼容边界。

停止条件：若“任意私有可变 MjSim 访问也必须完全兼容”被设为硬要求，则先报告不可行边界，
不得用假的 proxy 继续。

### G1：完整发行面和 official golden baseline

目标：统一发行版在 `backend="official"` 下成为兼容基线。

任务：

1. 恢复/重构必要 public import surface；
2. 将旧训练、Torch tensor runtime 和 Warp GPU 依赖移入各自 optional
   profiles，保持 core runtime 精简；以 Python 3.12 为主要开发版本，并只发布
   通过 3.10/3.11/3.12 matrix 的解释器范围；
3. 保持原 import paths、constructors、returns 和 exceptions；
4. 加 `backend` 参数但默认 official；
5. 加 `backend_info` 和 provenance，不改变数值行为；
6. 跑官方 walkthrough、task suite、collector 和 vector-env smoke；
7. 生成 official golden fixtures。

Hard gate：

- upstream API manifest 的 Level A/B 项全部通过；
- `backend` 未指定时 golden fixtures 不变；
- official integration/static/packaging tests 通过；
- wheel 安装后 import/data/CLI smoke 通过；
- no Warp code path required for official use。

### G2：内部 backend seam（仍只运行 official）

目标：把 `ControlEnv` 实现拆出可替换 session，但对外行为零变化。

任务：

1. 定义最小内部 `LiberoBackendSession`；
2. 实现 `OfficialLiberoSession` 透传；
3. `ControlEnv` 只负责兼容 wrapper、metadata 和 backend selection；
4. 把 camera/state/predicate/controller 的 authoritative ownership 写清；
5. 防止内部 `RuntimeEnv` 新类型泄漏到 legacy API；
6. 加 lifecycle、exception、RNG、hard-reset differential tests。

Hard gate：

- official golden 全绿；
- 无 NumPy/RNG/model identity 漂移；
- `OffScreenRenderEnv` 和 vector-env 用户代码无需修改；
- seam 不包含 MVBeliefWM/collector-specific 类型。

### G3：Warp N=1 reset/render/state 兼容

目标：不含 action step 的 N=1 Warp compatibility vertical slice。

任务：

1. 复用 exact TaskCompiler；
2. 支持构造器中的任意合法 camera config；
3. 实现 reset/set_init_state/get_sim_state；
4. 输出原式 observation dict；
5. 对齐 RGB/depth/segmentation/proprio/state/time；
6. 实现 task/predicate 所需最小 readout；
7. unsupported methods fail clearly；
8. 不开放 `step()`，或让其明确抛出 action capability error。

覆盖范围：至少选择不同 suite、scene、object、camera/depth/segmentation 组合，不只 pilot task。

Hard gate：

- official-vs-Warp reset/state/camera/predicate oracle 通过；
- 无 silent fallback；
- 返回 key/shape/dtype 与 official 一致；
- no hardcoded task/camera/resolution；
- close/重复构造无失控资源增长。

### G4：7-D OSC action 与 N=1 `step()`

目标：实现真正的 LIBERO action 语义。

任务：

1. 冻结 robosuite 1.5.2 默认 composite `OSC_POSE` contract；
2. 实现/复用 Warp OSC controller；
3. 对齐 action clipping、goal update、gripper、control decimation；
4. 对齐 reward、done、success、horizon；
5. 验证 contact、orientation、grasp、articulated object 行为；
6. 建立 teacher-forced rollout corpus；
7. 对误差使用 task/物理含义明确的 tolerance。

Hard gate：

- one-step state/action parity；
- 多步 teacher-forced suffix；
- task success/reward/predicate 一致；
- 不再使用 raw ctrl 代表 public action；
- 代表性任务簇全部通过后才发布 `step()` capability。

停止条件：若 GPU OSC 不能复现 LIBERO 行为，保持 Warp render/reset-only capability，不把它伪装成
完整环境。

### G5：完整观测、wrapper 和 N=1 drop-in

目标：普通下游把 backend 改成 Warp 后可直接运行。

任务：

1. 完整 robosuite observable cadence/cache；
2. proprio/object observations；
3. `SegmentationRenderEnv` helper 行为；
4. `DemoRenderEnv` 和 state/XML rerender；
5. common `sim` read proxy；
6. public examples 和真实下游 smoke；
7. exception/lifecycle compatibility。

Hard gate：

- 标准 LIBERO walkthrough/evaluation 可运行；
- MVBeliefWM 只通过 backend 选择即可运行普通 rollout；
- LeWM/其他已知下游 smoke 不需要特异 adapter；
- API/behavior compatibility matrix 更新为真实结果。

### G6：Warp-native vector environment

目标：在同一语义上提供 GPU batch，而不是再造一套任务 API。

任务：

1. `[N,7]` batched OSC；
2. full/partial reset；
3. batched cameras/modalities；
4. batched predicates/reward/done；
5. health/overflow/NaN reporting；
6. device-resident observation option；
7. explicit CPU compatibility output option；
8. original `DummyVectorEnv`/`SubprocVectorEnv` regression。

Hard gate：

- N=1 native 与 N=1 compatibility 语义相同；
- N=1 与 N>1 world independence；
- reset subset 不污染其他 worlds；
- batch-size invariance；
- no implicit timed CPU transfer；
- negative/lifecycle/resource tests 通过。

### G7：action-exact continuation

目标：为离线 branching 提供正式、版本化、已验证的 snapshot。

任务和门槛见第 7 节。另需覆盖：

- snapshot before/after gripper transitions；
- contact-heavy boundary；
- articulated objects；
- RNG-dependent continuation；
- N>1 partial restore；
- snapshot serialization roundtrip。

在 G7 前，离线增广可以使用 exact prefix replay，但不能宣称 direct branch。

### G8：130-task 与下游兼容资格测试

目标：从 representative coverage 扩展到 LIBERO 全任务。

分层：

1. compile/reset/camera smoke：130/130；
2. short teacher-forced suffix：130/130；
3. predicate/success corpus：全部 predicate family；
4. representative full demonstrations：每 task 至少一条有效样本；
5. error triage 按 task/predicate/asset/controller 分类。

Hard gate：

- 不能以 aggregate pass rate 掩盖单 task failure；
- 所有 task 有明确 supported/unsupported 状态；
- release-ready 宣称要求 130/130 支持；
- model/asset/provenance 一致性 fail closed。

### G9：离线数据工厂和多 GPU

目标：验证真实主要工作负载，而不是只跑 microbenchmark。

端到端 benchmark 必须包含：

- task/model compile amortization；
- 7-D OSC；
- 物理；
- 所需 RGBD/segmentation；
- predicate/reward；
- branch/prefix replay；
- episode assembly；
- encode/compress/write；
- validation/retry/drop rate。

核心指标：

```text
generated rows/s
behaviorally-valid rows/s
verified branches/s
episodes/hour
GPU-hours per million valid rows
peak host RAM / GPU memory / storage bandwidth
```

资格测试：

- official CPU collector baseline；
- Warp 1 GPU；
- Warp 2 GPU；
- Warp 4 GPU；
- 同 workload、camera、action、writer 和 validity contract；
- 至少三次重复，报告 median/range；
- 多卡 scaling 同时报告 throughput、efficiency 和失败率。

初始目标可设为：

- 2 GPU 相对 1 GPU 有效吞吐 `>=1.7×`；
- 4 GPU 相对 1 GPU 有效吞吐 `>=3.0×`；
- 任一规模不得降低 behavioral-valid 比例；
- 不要求单 world 达到实时 20 Hz，除非 online 部署另有明确需求。

目标是资格线，不是保证值；未达到时先 profile I/O、CPU controller/predicate、kernel launch、
renderer、writer 和 compile，再决定是否继续优化。

### G10：发布

交付物：

- wheel/sdist；
- API compatibility report；
- task support matrix；
- backend capability matrix；
- official/Warp differential report；
- single/vector/multi-GPU examples；
- migration guide；
- known private API limitations；
- benchmark report；
- versioned provenance schema；
- release notes 和 rollback/fallback 指南。

Release hard gate：

- clean environment wheel install；
- public imports/CLI/package data；
- official backend 无回归；
- Warp capability 不夸大；
- 130-task 支持声明与证据一致；
- no silent fallback；
- static/CPU/GPU tests、Ruff、`git diff --check` 全通过；
- 文档中的数字可由保留的 machine-readable artifact 重算。

---

## 9. 测试策略

### 9.1 API conformance

- module/symbol manifest；
- constructor/method signatures；
- default values；
- return arity/types；
- exception types；
- package data/import/CLI；
- public examples。

manifest 只检测表面兼容，不能替代行为测试。

### 9.2 black-box differential

同一 task/config/seed/init-state/action 下分别运行 official 和 Warp，比较：

- obs keys/shapes/dtypes；
- camera calibration；
- state/control boundary；
- reward/success/done；
- predicate inputs/outputs；
- RGB/depth/segmentation；
- rollout suffix；
- reset determinism；
- close/resource lifecycle。

### 9.3 downstream contract corpus

将真实下游用作 smoke/compatibility consumer，但不把其类型放入核心包：

- 标准 LIBERO scripts/notebooks；
- MVBeliefWM LIBERO wrapper；
- LeWM online collector；
- offline augmentation collector；
- 其他用户指定仓库。

每个 consumer 记录：import surface、public calls、private calls、required capabilities、是否需要源码
修改。目标是尽量做到只改 backend selection/config。

### 9.4 negative tests

- unknown backend；
- unsupported controller/robot/camera；
- malformed action/state/XML；
- NaN/Inf；
- device mismatch；
- capacity/overflow；
- stale proxy；
- closed env；
- wrong model/snapshot provenance；
- unsupported private write；
- requested Warp capability unavailable；
- multi-GPU duplicate seed/episode ID。

### 9.5 性能测试的正确边界

保留 microbenchmark 用于定位，但产品结论只使用端到端 workload。计时必须明确：

- warmup/JIT；
- CUDA synchronization；
- compile 是否包含；
- render modality；
- controller/predicate；
- CPU transfer；
- writer/compression；
- dropped/invalid rows。

不允许只报告 aggregate world-Hz 而省略每 world 频率、有效率、资源和端到端吞吐。

---

## 10. 首个 bounded vertical slice

新 thread 的第一轮只执行 G0，最多推进到 G1 的测试骨架。不要直接实现 GPU OSC 或重构
`spike.py`。

推荐首轮变更范围：

```text
docs/LIBERO_API_COMPATIBILITY_MATRIX.md
tests/api_manifest/<target>.json
tests/test_libero_api_compatibility.py
可选：一个只读 manifest 生成脚本
```

首轮必须回答：

1. compatibility target 的准确 commit/tag 是什么；
2. 当前 wheel 缺失哪些原 public modules/symbols/data/CLI；
3. 当前构造器/返回值/异常有哪些漂移；
4. 已知下游具体依赖哪些 public/private calls；
5. `libero.lifelong` 和 configs 如何恢复而不污染 core dependencies；
6. backend selection 放在哪一层；
7. 哪些 API 能严格兼容，哪些只能提供受限 proxy；
8. G1 的最小 diff 是什么。

首轮验收：

- 全部只读审计证据可复现；
- compatibility matrix 无“以后再看”的模糊大项；
- 不改物理、renderer、controller；
- 不混入当前 M1.5 未提交修改；
- 用户审查 G0 后才进入 G1。

---

## 11. 实施纪律

- 使用 `uv` 管理 Python 环境；
- 修改 Python 后运行 `uvx ruff format` 和 `uvx ruff check`；
- 使用 `rg`/`rg --files` 做源码和 API 搜索；
- 保留并隔离工作树中已有修改；
- 不用 `git reset --hard`、`git checkout --` 或其他破坏性回滚；
- 未经用户授权不 commit、不 push、不创建 PR；
- 每一阶段先读源码和真实 runtime，再修改；
- official 是语义 oracle，不是可以为迁就 Warp 而放宽的目标；
- 任何新 tolerance、fallback、proxy 或 capability 必须有 negative test；
- machine-readable artifacts 与人类报告同时保存；
- 每个 benchmark 记录命令、commit、环境、GPU、versions、配置和原始 JSON；
- AutoDL/GPU 验证先探测并复用已有 Torch/CUDA managed environment；没有具体
  incompatibility 证据时禁止用 full `uv sync` 重装另一套 CUDA；
- Python release matrix 与 GPU qualification profile 分离，不能为了验证新 Python
  版本而重建可复用的 GPU 软件栈；
- 阶段失败时在当前 gate 停止并形成证据，不跨 gate 堆功能。

---

## 12. Definition of Done

只有同时满足以下条件，才可称为“通用 LIBERO-Warp 兼容发行版”：

1. 原 LIBERO 核心 import、环境类、benchmark、task/assets/init-state 和常用工具可用；
2. 默认 official backend 行为无回归；
3. `OffScreenRenderEnv(..., backend="warp")` 对支持配置保持原单环境 API；
4. Warp 接收原 7-D LIBERO action，而非 raw ctrl；
5. 原式 observation/reward/done/success/camera/state 语义通过差分门；
6. arbitrary legal camera configuration 不被固定 profile 限制；
7. vector extension 与 N=1 语义一致，world/reset 独立；
8. render-exact 和 action-exact capability 不混淆；
9. 130 tasks 的支持声明有逐 task 证据；
10. 高频 private read 有明确 proxy 或替代 API，不支持写操作 fail loudly；
11. wheel 安装、package data、CLI 和文档完整；
12. 单卡/多卡端到端离线数据吞吐有可复算报告；
13. 下游无需导入项目特异 runtime 类型，只需选择 backend 或 native batch extension；
14. 所有 capability、版本、模型和 backend provenance 可查询且进入数据/report；
15. known limitations 清楚，不以 silent fallback 或模糊“兼容”掩盖差异。

在此之前，应使用准确阶段性名称，例如：

- `MJWarp physics/render spike`；
- `Warp reset/render compatibility preview`；
- `Warp N=1 LIBERO compatibility beta`；
- `Warp vector data-factory preview`。

不得提前称为 drop-in replacement。
