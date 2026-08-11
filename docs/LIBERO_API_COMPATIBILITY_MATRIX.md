# LIBERO-Warp G0 API compatibility matrix

Status: `G0 frozen by user / G1 implementation complete, external qualification pending`

Date: 2026-08-11 (Asia/Shanghai)

This document freezes the compatibility boundary approved by the user on
2026-08-11. The scoped G1 source implementation is complete, but G1 is not
freeze-ready until every hard gate passes. No Warp physics, renderer,
controller, or M1.5 source is changed by the G0/G1 compatibility work.

## 1. Frozen compatibility target

| Field | Value |
|---|---|
| Upstream | `https://github.com/Lifelong-Robot-Learning/LIBERO.git` |
| Ref used to resolve the target | `upstream/master` |
| Proposed frozen commit | `8f1084e3132a39270c3a13ebe37270a43ece2a01` |
| Commit date | `2025-03-15T20:13:56+08:00` |
| Git tree | `99f4ada3f1d62e026fc9ff2390eb4ff8a1760e60` |
| Local branch merge-base | the same commit |

The commit, not the moving ref, is the compatibility identity. It is already the
fork point of this repository and is also the commit pinned by MVBeliefWM's
official LIBERO runtime. A future upstream update requires a new manifest and an
explicit compatibility-target change; it must not silently change this target.

Reproduce the inventory without importing LIBERO or its optional dependencies:

```bash
uv run python scripts/generate_libero_api_manifest.py \
  --target-ref 8f1084e3132a39270c3a13ebe37270a43ece2a01 \
  --output tests/api_manifest/upstream-8f1084e3132a39270c3a13ebe37270a43ece2a01.json
uv run pytest -q tests/test_libero_api_compatibility.py
```

The generated JSON is authoritative for the complete static source inventory. It
covers every Python module below `libero/`, `benchmark_scripts/`, and `scripts/`;
statically declared top-level definitions, imports, assignments, and conditional
bindings; function, class, and method signatures; all source-data files, including
both template files; and all upstream console scripts. A top-level `.py` template
is source data, not an import module.

Every source-data record contains its Git blob OID and byte size, so same-path
content drift is detectable. The G0 test also builds the current wheel and checks
its target Python paths, target data paths, and console-entry-point metadata
against this inventory. A clean compatibility wheel and installed import/data
smoke remain a G1 gate.

## 2. Verified G0 inventory

The table below is the frozen pre-G1 baseline captured during G0. The generated
manifest is intentionally refreshed as G1 restores the working tree; the
post-restoration status is recorded immediately after the baseline.

| Surface | Target | Current working tree | G0 result |
|---|---:|---:|---|
| Python modules | 97 | 81 | 49 statically compatible, 29 missing, 19 with drift |
| Source-data files | 1002 | 972 | 32 missing, 0 same-path content changes, 2 relocated extras |
| Console scripts | 4 | 2 | 3 target commands missing; 1 new command added |

G1 working-tree status after restoration:

| Surface | Target | Current working tree | G1 status |
|---|---:|---:|---|
| Python modules | 97 | 111 | 73 statically compatible, 24 intentional/static drift, 0 missing |
| Source-data files | 1002 | 1004 | 0 missing, 0 same-path content changes, 2 relocated extras retained |
| Console scripts | 4 | 5 | 0 target commands missing; 1 additive command retained |

The target has four console scripts:

- `libero.config_copy` → `scripts.config_copy:main`;
- `libero.create_template` → `scripts.create_template:main`;
- `lifelong.eval` → `libero.lifelong.evaluate:main`;
- `lifelong.main` → `libero.lifelong.main:main`.

The current distribution restores all four target commands and retains the
additive `libero.download_datasets` command. Section 6 records the decision for
each command that was missing in the frozen pre-G1 baseline.

The 32 missing source-data paths are the 30 YAML files below `libero/configs/`
plus `templates/scene_template.xml` and
`templates/problem_class_template.py`. The latter remains a template artifact,
not an importable module. The two current extra paths are the relocated copies
below `libero/templates/`.

Packaging metadata is also frozen as evidence:

- upstream distribution `libero==0.1.0` uses a dynamic
  `find_packages()` expression filtered to `libero*`, declares
  `include_package_data=True`, and provides no explicit package-data mapping in
  `setup.py`;
- current distribution `libero-warp==0.1.0` discovers `libero*`, `scripts*`,
  `benchmark_scripts*`, and `benchmarks*`, and explicitly maps assets, BDDL,
  init states, and `libero/templates/*` as package data;
- therefore upstream source paths, current packaging intent, and actual built
  wheel contents are three separate facts. A G0 build of the current wheel
  confirms that it lacks exactly 29 target Python paths, 32 target data paths,
  and 3 target console scripts; the retained target entry point has the exact
  target value. That pre-G1 result remains reproducible from the frozen target.
  G1 now builds the restored wheel, installs it into a clean temporary
  environment, and checks imports, runtime-data lookup, and all five console
  scripts rather than inferring success from source layout.

## 3. Compatibility levels

### Level A: modules, imports, data, and CLI

| Surface | Compatibility decision | Current status | G1 gate |
|---|---|---|---|
| `libero.libero`, `libero.libero.benchmark` | Strict public compatibility | Present | Direct import plus benchmark-suite static and runtime smoke |
| `libero.libero.envs`, `libero.libero.envs.env_wrapper`, `libero.libero.envs.venv` | Strict documented imports and class names | Present with robosuite-1.5-related static drift | Import and signature contract; official single/vector smoke |
| `libero.libero.utils` and its target modules | Preserve module paths and documented functions | Present | Import and representative utility smoke |
| BDDL, assets, init states | Strict path and package-data compatibility | Present in the current manifest | Clean-wheel lookup and hash/count smoke |
| `benchmark_scripts` and documented CLIs | Preserve target modules and commands; deprecate path-mutation helpers | Restored in G1 working tree; init-path shim is a no-op | Installed CLI smoke; no repository-root assumption |
| `scripts` target modules and documented CLIs | Preserve target modules; allow additive CLI arguments | Restored in G1 working tree; init-path shim is a no-op | Installed module/CLI smoke and target default behavior |
| `libero.configs` | Restore as package data; training dependencies stay optional | Restored in G1 working tree | Wheel data lookup; Hydra composition only with `legacy` extra |
| `libero.lifelong` | Restore exact import paths under `legacy` extra | Restored in G1 working tree with an actionable dependency guard | Import smoke in clean core and `legacy` environments; training reproduction is a separate qualification |
| Original top-level `templates/` | Restore both template files as compatibility package data while retaining the new packaged location | Restored in G1 working tree and wheel data scheme | Wheel/sdist lookup from both supported locations |
| New `libero.libero.runtime` / `libero.libero.warp` | Additive LIBERO-Warp extension, not part of the upstream target | Present | Must not be required by the official legacy imports |

Incidental names imported by upstream modules, such as `List`, `Type`, `glob`,
`random`, `cv2`, and robosuite implementation classes, are recorded in the
machine manifest because they were technically reachable. They are not promoted
to stable Level A API. G1 may retain compatibility aliases where safe, but release
documentation will mark these incidental re-exports deprecated and unsupported
for new callers.

### Level B: constructors, methods, and return shapes

| Contract | Decision | Current evidence | Required G1 proof |
|---|---|---|---|
| `ControlEnv(...)` target constructor | Preserve every target argument and default; later `backend` is optional keyword-only behavior and must not reach the robosuite constructor | Static target/current signature matches | Constructor differential over default and explicit arguments |
| `OffScreenRenderEnv(**kwargs)` | Preserve wrapper identity and forced offscreen flags | Static signature matches | Official golden construction/reset/close |
| `SegmentationRenderEnv(...)` | Preserve constructor and helper names; corrected robot-instance logic is allowed | Static constructor matches | Instance/class/element segmentation fixtures |
| `DemoRenderEnv(**kwargs)` | Restore direct import from `env_wrapper`; no promise that it is re-exported from `envs` because upstream did not do so | Present | XML/state rerender fixture |
| `reset`, `step`, `seed`, `close`, `check_success` | Preserve call signatures, four-value step result, exceptions, horizon, and `ignore_done` behavior | Methods present; robosuite 1.5 implementation differs internally | Black-box golden tests |
| `set_init_state`, `get_sim_state`, `set_state`, `regenerate_obs_from_state`, `reset_from_xml_string` | Strict official behavior; Warp support is capability-gated | Present for official wrapper | State/rerender differential and negative capability tests |
| `obj_of_interest`, `robots`, `sim`, language/problem/domain fields | Official backend exposes the real objects/values; Warp follows Level D restrictions | Present for official wrapper | Identity/freshness tests |
| `DummyVectorEnv`, `SubprocVectorEnv` | Preserve upstream vector semantics and public methods | Present | Single/subprocess reset-step-close smoke |

Signature changes that add an optional CLI `argv=None` are compatible extensions.
The removed `MountedPanda.default_mount` and `OnTheGroundPanda.default_mount`
properties are not accepted drift: G1 restores read-only aliases backed by the
robosuite 1.5 model and verifies the values. `SingleArm` and
`ROBOT_CLASS_MAPPING` compatibility names are restored as deprecated aliases only
when they resolve to the actual robosuite 1.5 robot types; they must not be fake
classes.

#### Current constructor, return, and exception drift

G0 distinguishes static equivalence, intentional migration repair, and behavior
that still needs a golden. It does not label every internal difference unknown.

| Surface | Target versus current finding | Classification and proof |
|---|---|---|
| `ControlEnv(...)`, `OffScreenRenderEnv(**kwargs)`, `SegmentationRenderEnv(...)`, `DemoRenderEnv(**kwargs)` | Target arguments and defaults are unchanged | Statically equivalent; the isolated import contract passes. Object construction remains a G1 official golden because it initializes robosuite/MuJoCo. |
| Controller construction | Target uses `suite.load_controller_config`; current uses robosuite 1.5 `load_part_controller_config` followed by `refactor_composite_controller_config` | Intentional dependency migration. The default `OSC_POSE` construction path has existing integration coverage; non-default controller behavior remains unqualified and is a G1 differential. |
| `seed(seed)` | Target calls the old callable `self.env.seed(seed)`; current sets robosuite 1.5's numeric `seed`, replaces `rng` with `default_rng(seed)`, and maintains a wrapper-local legacy NumPy RNG state | Intentional deterministic-compatibility repair. Signature is equivalent and global NumPy state is isolated by construction; repeated-reset observation/state equivalence still requires the G1 seeded golden. |
| `reset()` | Target retries the whole reset loop; current retries only `RandomizationError` through `_reset_with_legacy_rng` and propagates unexpected errors | Intentional fail-fast bug repair. `test_control_env_reset_retries_only_randomization_errors` proves a placement error retries while an unexpected `ValueError` escapes unchanged. |
| Segmentation `reset()` | Current raises `RuntimeError` if no robot instance ID can be identified, rather than silently deriving an invalid threshold | Intentional fail-closed repair, verified by `test_segmentation_reset_reports_missing_robot_instance`. |
| Segmentation conversion helpers | Current raises `ValueError` for unknown segmentation IDs and `KeyError` when an object of interest is absent | Intentional fail-closed repair by code inspection. Existing migration tests verify explicit robot IDs and interest masks; invalid-ID and missing-interest exceptions are required additions to the G1 golden. |
| `reset()` / `step()` return forms | Public signatures are unchanged; `step` still delegates the upstream four-value result and reset delegates the observation mapping | Statically equivalent call surface, not yet a runtime-equivalence claim. G1 records keys, shapes, dtypes, reward, done, and info from the target/current official paths. |
| State/XML methods and `close()` | Public signatures and delegation targets remain present; no new public exception is introduced by G0 | Statically equivalent call surface. Exact state shape, XML lifecycle, and close/reconstruct failures remain G1 black-box gates. |

### Level C: black-box behavior

All rows below are strict requirements for `backend="official"`. Their current
status is “qualification required”, not “assumed compatible”.

| Behavior | Contract and concrete test artifact |
|---|---|
| Reset and RNG | Same seed/model/init-state placement and exception behavior; serialize observation and state fixtures for repeated seeded resets |
| Step/action | Same 7-D `OSC_POSE` clipping, decimation, reward, done, success, and info; record one-step and short-rollout golden fixtures |
| Observations | Same keys, order where observable, shapes, dtypes, cadence/cache behavior, and absent-modality behavior; record JSON metadata plus NPZ arrays |
| Cameras | Same caller order, resolution, vertical orientation, intrinsics/extrinsics, metric depth, and segmentation mapping; use official outputs as golden fixtures |
| State/XML | Same flattened state, `set_init_state`, rerender, hard reset, and XML reset lifecycle; include repeated close/reconstruct tests |
| Vector env | Same reset/step/result stacking and subprocess failure propagation; run installed-wheel smoke |
| Packaging | Clean wheel imports all Level A core modules, contains target runtime data, and exposes every retained CLI |

No G1 test may import or initialize MJWarp to make the official path pass.

### Level D: factual private access

The corpus confirms each access below in the current working trees. Full
transparent emulation of an arbitrary mutable CPU `MjSim` on a
GPU-authoritative backend is explicitly not a release requirement. “Unsupported”
always means a synchronous `UnsupportedBackendOperation`, never a dummy object,
stale CPU mirror, silent no-op, or official-backend fallback.

| Source and exact access | Mode / required semantics | Official backend | Warp decision | Formal replacement or error |
|---|---|---|---|---|
| LeWM `envs/libero.py`: `action_spec()` or `action_space.low/high` | Read; exact external action bounds | Real robosuite 7-D bounds | Supported metadata once the G2 facade exists | Formal `action_spec` / `action_space` read-only compatibility view; before G2, constructing Warp itself fails |
| MVBeliefWM and LeWM: `wrapper.env`, `env.robots` | Read; task and robot identity | Real robosuite objects | No mutable task-object facade | Stable task/robot metadata plus `backend_info`; direct object access raises `UnsupportedBackendOperation` |
| Both continuation adapters: `robot.controller` fields and mutation | Read/write; action-exact controller continuation | Real controller object | Unsupported before G4/G7 | Versioned controller capability metadata for reads; mutation raises until `capture_continuation_state` / `restore_continuation_state` exists |
| LeWM and MVBeliefWM continuation: `robot.gripper.current_action` read/write | Read/write; action-exact gripper command continuation | Real mutable gripper state | Unsupported before G7 | Included only in the versioned continuation snapshot; direct read/write raises |
| MVBeliefWM continuation: `robot.torques`, `recent_ee_vel_buffer`, and other robot-private buffers; LeWM robot joint-index fields | Read/write; action-boundary robot transients and simulator indexing | Real robot internals | No general robot-private proxy | G7 continuation snapshot owns validated robot transients; static joint/index metadata may use a read-only versioned view, otherwise raise |
| MVBeliefWM runtime: `env.sim.model.get_xml()` | Read; exact episode model and assets | Real model XML | Supported read | `get_model_xml()` returns exact XML plus provenance; no raw model proxy |
| MVBeliefWM/LeWM: `sim.data.time`, qpos/qvel and flattened state reads | Read; live authoritative state | Real live `MjData` | Supported only through explicit views | `get_sim_state()` plus versioned read-only state/episode view with device and freshness metadata |
| Continuation adapters: direct `sim.data` / `sim.model` writes | Write; restore physics/model internals | Existing mutable MuJoCo behavior | Unsupported before G7 | Raise; only versioned continuation capture/restore may become writable |
| MVBeliefWM runtime: `sim.reset(); sim.forward()` after XML reset | Write/lifecycle; reinitialize and propagate model state | Real simulator calls | Direct calls unsupported | Use `reset_from_xml_string()` followed by `regenerate_obs_from_state()` or backend-owned reset; direct `sim.reset/forward` raises |
| MVBeliefWM runtime: camera intrinsic/extrinsic utilities receiving `sim` | Read; caller-ordered calibrated cameras | Real MuJoCo calibration | Supported through formal camera API, not raw `sim` | `get_camera_calibration(camera_names)` returns versioned intrinsics/extrinsics; before G3/G5 capability availability, raise |
| MVBeliefWM/LeWM: `get_real_depth_map(sim, depth)` | Read/transform; metric camera-z depth | Real MuJoCo near/far conversion | Supported only as an observation capability | Request `metric_depth` observations from the backend; raw-sim conversion raises before G5 |
| LeWM restore: write `deterministic_reset` around `reset()` | Temporary write; deterministic placement without resampling | Real robosuite flag | Direct write unsupported | `set_init_state()` / deterministic reset options owned by backend; direct attribute write raises |
| LeWM restore: write `hard_reset` around `reset_from_xml_string()` | Temporary write; force simulator and sensor rebinding | Real robosuite flag | Direct write unsupported | `reset_from_xml_string()` owns the full lifecycle and observable rebinding; direct attribute write raises |
| LeWM capture/replay: `_render_context_offscreen.gl_ctx.make_current()` | Side effect; bind the correct EGL context | Real offscreen GL context | No GL context is exposed | Use backend observation/render API; direct context access raises, and a no-op is forbidden |
| Both continuation adapters: `_observables` and `_obs_cache` reads/writes | Read/write; sensor cadence and cached continuation | Real mutable robosuite state | No mutable cache proxy | Constructor observation config for normal use; G7 continuation snapshot owns cadence/cache state; direct access raises |
| LeWM/MVBeliefWM: `_update_observables(force=True)` and `_get_observations()` | Side effect/read; force fresh sensor evaluation | Real robosuite methods | Direct methods unsupported | `regenerate_obs_from_state()` for restored states and the normal returned observation mapping otherwise; direct calls raise |
| LeWM: `modify_observable(..., "sensor", ...)` for lazy image gating | Write; change sensor evaluation policy | Real robosuite mutation | Direct mutation unsupported | Public observation-selection/configuration API in G5; before then raise |
| Continuation adapters: `cur_time`, `timestep`, `done` reads/writes | Read/write; exact episode boundary | Real wrapper fields | Reads supported, writes unsupported before G7 | Read-only episode-state view; writes only via a versioned continuation snapshot |

The common exception proposed for G1/G2 is
`UnsupportedBackendOperation(RuntimeError)`. It must name the requested operation,
actual backend, missing capability, and supported replacement. Silent fallback to
official is forbidden.

## 4. Known downstream usage report

The audit scanned the current working trees, not only their HEAD commits:

| Corpus | HEAD | Main evidence |
|---|---|---|
| MVBeliefWM | `5152c50d45be095e1ca237f185aa4ec26eb5f8ea` plus local changes | `mvbeliefwm/envs/libero_runtime.py`, `mvbeliefwm/data/continuation.py` |
| LeWM | `79758096b09c59bbe197e680abcc30316996196b` plus local changes | `lewm/envs/libero.py`, `lewm/evaluation/libero.py`, `apps/convert_dataset.py` |
| LIBERO-warp | `d396f2daedce66b5e8bcc4b5765e3050686f213a` plus M1.5 changes | scripts, benchmark scripts, runtime/compiler tests |

Reproducible search shape:

```bash
rg -n --glob '*.py' \
  'libero\.libero|OffScreenRenderEnv|get_benchmark|get_libero_path' \
  /Users/chenhongru/workspace/MVBeliefWM \
  /Users/chenhongru/workspace/LeWM
rg -n --glob '*.py' \
  '\.(env|sim|robots|controller|_observables|_obs_cache|modify_observable)\b' \
  /Users/chenhongru/workspace/MVBeliefWM/mvbeliefwm \
  /Users/chenhongru/workspace/LeWM/lewm
```

Verified usage:

- Both research repositories require `benchmark`, `get_libero_path`, and
  `OffScreenRenderEnv`, plus reset/step/state/XML methods.
- MVBeliefWM requires caller-ordered camera observations, camera geometry,
  `sim.model.get_xml()`, and a pinned action-continuation adapter that inspects
  robot, controller, simulator, observable, and cache state.
- LeWM requires official construction and evaluation, `set_init_state`,
  `get_sim_state`, XML restore, action bounds, lazy observable gating, and direct
  controller/simulator mutation for action-exact replay.
- The private mutation users are version-pinned continuation systems. They do not
  justify a universal mutable Warp proxy. They justify the G7 versioned
  continuation API and fail-closed behavior before it exists.
- No scanned downstream requires `libero.lifelong` for its current environment or
  evaluation path. Restoring lifelong is therefore a distribution-compatibility
  obligation isolated behind the `legacy` extra, not a core runtime dependency.

## 5. Backend selection and provenance proposal

Selection is frozen as:

1. explicit constructor `backend=`;
2. `LIBERO_SIM_BACKEND`;
3. `official`.

Accepted values are exactly `official` and `warp`. Empty, unknown, unavailable, or
unsupported values fail before environment construction. A requested Warp
configuration never falls back to official.

The only G0 schema proposal is a frozen read-only `BackendInfo` value with these
fields:

| Field | Type / meaning |
|---|---|
| `schema_version` | integer, initially `1` |
| `requested_backend` | exactly `official` or `warp` |
| `selection_source` | exactly `constructor`, `environment`, or `default` |
| `actual_backend` | `official` or `warp`; must equal the resolved request |
| `capabilities` | immutable `frozenset[str]`; JSON is a lexicographically sorted array of the frozen names below |
| `libero_warp_version` | installed distribution version |
| `libero_compat_target` | exact commit from section 1 |
| `dependency_versions` | sorted string-to-string-or-null map for robosuite, MuJoCo, MJWarp, Warp, NumPy, and Torch |
| `device` | authoritative compute/render device identity |
| `build` | Python, platform, CUDA, and build provenance |

Schema v1 capability names are exactly:

```text
action_exact_continuation
camera_calibration
cuda_observations
metric_depth
model_xml_read
model_xml_reset
native_batch
partial_reset
predicate_success
private_sim_read_proxy
render_exact_restore
reset
rgb
segmentation
single_env_numpy_api
state_flattened_read
state_flattened_write
step_osc_pose_7d
```

Absent capabilities are omitted from the sorted JSON array; they are never
serialized as present with a false value. New semantics require either a new
capability name or a `BackendInfo.schema_version` change. `build` and `device`
are JSON objects with sorted keys and JSON scalar values; the full value is
read-only after construction.

`BackendInfo` is metadata only. It does not expose a new observation, action, or
batch type to legacy environment users.

### Environment and dependency contract

Python and Torch/CUDA are deployment profiles, not part of the frozen upstream
LIBERO API identity. G1 must replace the current single, fully pinned environment
with these boundaries:

| Concern | Frozen G0 decision | Required G1 proof |
|---|---|---|
| Primary development Python | Python 3.12 | Run static, import, wheel, and official smoke under 3.12 before changing release metadata |
| Python support range | `>=3.12,<3.13` | Qualify and publish Python 3.12 only |
| Manifest tooling | Python 3.12 | Reproduce the saved manifest under the release interpreter |
| Legacy/core LIBERO API | The official MuJoCo/robosuite backend is the compatibility baseline; no Torch, MJWarp, Warp, or CUDA installation requirement | Public import, benchmark, data, `OffScreenRenderEnv`, and NumPy/dict/tuple compatibility smoke in a core environment |
| Official backend | Exact robosuite/MuJoCo versions remain qualification inputs; it must not require MJWarp or CUDA | Official golden and installed-wheel smoke with no CUDA packages installed |
| Tensor runtime | Torch is an optional capability dependency, not a core dependency | Test the minimum supported Torch plus the current qualification profile; fail with an actionable extra-install error when absent |
| Warp backend | Exact MuJoCo/MJWarp/Warp/robosuite profile until requalified; Torch is accepted by tested API/capability range | GPU parity and performance reports record exact Python, Torch, Torch-CUDA, driver, and device versions |
| CUDA runtime | Never pin or install `nvidia-*` packages directly from LIBERO-Warp core metadata | Use the CUDA runtime supplied by the selected Torch build or existing managed environment; require `torch.cuda.is_available()` only for Warp GPU gates |

The implemented candidate Torch range is `>=2.4,<3`, subject to minimum-version
CI.
LIBERO-Warp currently uses public tensor/device/dtype/stream operations and has no
custom Torch C++/CUDA extension, custom operator, or `torch.compile` dependency.
Exact Torch/CUDA versions therefore belong in benchmark provenance and validated
profiles, not in every installation's equality constraint.

The current AutoDL host supplies a working managed profile at
`/root/autodl-tmp/MVBeliefWM/.venv`: Python 3.10.12, Torch 2.13.0+cu130,
CUDA available on an RTX 4090 D. Remote validation must create a source overlay
that reuses this environment; it must not run a full `uv sync`, replace Torch, or
download another CUDA stack unless a test demonstrates a concrete incompatibility.
Python 3.12 release testing is a separate CPU/static/official job and must
not force rebuilding the AutoDL GPU environment.

G1 dependency profiles are:

```text
core       upstream-compatible imports, data, tools, legacy NumPy env API, and exact official MuJoCo/robosuite baseline
official   named compatibility profile for core's official baseline (no additional dependency)
tensor     optional Torch-backed additive runtime API
warp       tensor + exact qualified MJWarp/Warp physics profile
legacy     restored lifelong/training surface and its historical dependencies
```

Moving Torch out of core requires removing eager Torch imports from retained core
utilities and making tensor/Warp modules fail lazily with an actionable extra
message. Merely moving the requirement in `pyproject.toml` while core imports still
raise `ModuleNotFoundError` does not satisfy the gate.

## 6. Every missing public module and command: explicit decision

The following table is exhaustive relative to the generated manifest. “Restore
legacy-extra” means restore the original import path and source, keep its heavy
training dependencies out of core, and qualify it in a clean environment with the
`legacy` extra installed. It does not promise reproduction under modern
dependencies until a separate training qualification passes.

| Missing target module | G1 decision |
|---|---|
| `benchmark_scripts.init_path` | Restore as deprecated installed-package no-op shim; no `sys.path` mutation |
| `libero.configs` | Restore package and all target YAML as package data; Hydra lives in `legacy` extra |
| `libero.lifelong` | Restore legacy-extra import surface |
| `libero.lifelong.algos` | Restore legacy-extra import surface |
| `libero.lifelong.algos.agem` | Restore legacy-extra source |
| `libero.lifelong.algos.base` | Restore legacy-extra source |
| `libero.lifelong.algos.er` | Restore legacy-extra source |
| `libero.lifelong.algos.ewc` | Restore legacy-extra source |
| `libero.lifelong.algos.multitask` | Restore legacy-extra source |
| `libero.lifelong.algos.packnet` | Restore legacy-extra source |
| `libero.lifelong.algos.single_task` | Restore legacy-extra source |
| `libero.lifelong.datasets` | Restore legacy-extra source |
| `libero.lifelong.evaluate` | Restore source and `lifelong.eval`; fail with an actionable extra-install error when dependencies are absent |
| `libero.lifelong.init_path` | Restore as deprecated installed-package no-op shim |
| `libero.lifelong.main` | Restore source and `lifelong.main`; fail with an actionable extra-install error when dependencies are absent |
| `libero.lifelong.metric` | Restore legacy-extra source |
| `libero.lifelong.models` | Restore legacy-extra import surface |
| `libero.lifelong.models.base_policy` | Restore legacy-extra source |
| `libero.lifelong.models.bc_rnn_policy` | Restore legacy-extra source |
| `libero.lifelong.models.bc_transformer_policy` | Restore legacy-extra source |
| `libero.lifelong.models.bc_vilt_policy` | Restore legacy-extra source |
| `libero.lifelong.models.modules.data_augmentation` | Restore legacy-extra source |
| `libero.lifelong.models.modules.language_modules` | Restore legacy-extra source |
| `libero.lifelong.models.modules.rgb_modules` | Restore legacy-extra source |
| `libero.lifelong.models.modules.transformer_modules` | Restore legacy-extra source |
| `libero.lifelong.models.policy_head` | Restore legacy-extra source |
| `libero.lifelong.utils` | Restore legacy-extra source |
| `scripts.config_copy` | Restore core module and `libero.config_copy`; operate on installed package data |
| `scripts.init_path` | Restore as deprecated installed-package no-op shim |

Missing console commands:

| Command | G1 decision |
|---|---|
| `libero.config_copy` | Restore in core and smoke-test in a clean wheel |
| `lifelong.eval` | Restore entry point; actionable `legacy`-extra error without optional dependencies |
| `lifelong.main` | Restore entry point; actionable `legacy`-extra error without optional dependencies |

The current additive `libero.download_datasets` command remains supported. It
does not replace any target command.

## 7. G1 minimal scoped diff after user freeze

Only after this matrix is approved, G1 may touch:

- `.python-version`, `pyproject.toml`, and `uv.lock` for the frozen Python 3.12 profile,
  dependency profiles, package data, target CLIs, and optional extras;
- `.github/workflows/m0-verification.yml` for the frozen Python 3.12 core and
  official gates plus the scheduled legacy-profile import gate;
- restored target files under `libero/configs/`, `libero/lifelong/`, `scripts/`,
  `benchmark_scripts/`, and `templates/`;
- minimal eager-Torch removal in retained benchmark/utility modules so the
  declared core profile is true at import time rather than metadata-only;
- `libero/libero/envs/env_wrapper.py` and `libero/libero/envs/__init__.py` for
  official-default backend selection, read-only metadata, and safe compatibility
  aliases;
- new official golden, packaging, CLI, and vector smoke tests.

G1 must not alter M1.5 runtime/compiler/Warp files, make the official path depend
on MJWarp, implement OSC on GPU, expand to 130-task Warp qualification, or start
multi-GPU work. Existing dirty files remain outside the G0/G1 compatibility diff.

## 8. G0 gate result

The reproducible target, complete generated manifest, deletion decisions,
downstream private-usage boundary, backend-selection rule, and schema are now
frozen. The user explicitly approved the G0 freeze on 2026-08-11 and authorized
G1. The G1 hard gates remain independent and must pass before starting G2.

## 9. G1 implementation and qualification status

The scoped G1 implementation is frozen for the personal-use path approved on
2026-08-11. Release-grade legacy and full-suite qualification are deferred and
do not block G2.

Current local Python 3.12.5 core-profile evidence:

- MuJoCo 3.11.0 and robosuite 1.5.2 are installed; Torch, MJWarp, and Warp are
  absent.
- the maintained 39-file Ruff format and lint gate passes;
- the complete bounded static suite passes `177/177`;
- the wheel/API/core-optional suite passes `28/28`, including a clean temporary
  wheel install, installed resource lookup, zip-safe config copying, and all five
  console entry points;
- the official-enabled migration suite passes `15/15` without a renderer;
- the G1 static wrapper/vector/backend suite passes `8/8`;
- the real default-official headless golden passes `1/1`, covering seeded reset,
  observations, flattened state restore, one zero-action step, reward, done, and
  info;
- the final manifest is reproducible under the local Python 3.12.5 interpreter;
- `uv lock --check` passes, and default `uv sync --dry-run` only proposes
  reinstalling the editable project rather than adding Torch, MJWarp, Warp, or
  CUDA packages.

The schema-v2 manifest generated from the final local source has SHA-256
`183effb79a24c959055e12a082ecf3e161207b3460ee3af48e6c4cb0183c0113`.
It records 97 target versus 111 current modules, 1002 target versus 1004 current
source-data files, and 4 target versus 5 current console scripts. All target
module, data, and console-script paths are present; module coverage is 73 static
compatible plus 24 recorded drift.

The workflow now defines:

- one clean core/package job for Python 3.12;
- a Python 3.12 official pilot with OSMesa, including real instance, class, and
  element segmentation construction/reset/close fixtures;
- manually triggered full official and public-demo jobs, with Torch isolated to
  the `tensor` profile.

GitHub Actions run `31485631224` passed the Python 3.12 core/package job and the
OSMesa official pilot. The former passed all 177 bounded static contracts plus
the real headless golden; the latter passed all 40 controller, migration,
packaging, downloader, renderer, XML/state, and segmentation checks. This is the
frozen G1 gate for personal use. A positive all-module `legacy` import and the
130-task/public-demo qualification remain optional release work; the observed
`libero.lifelong.algos` native-import crash is recorded but is not a blocker for
the official/Warp runtime path.

## 10. G2 backend session seam

`ControlEnv` now constructs an internal `OfficialLiberoSession` and delegates
reset, step, RNG, state, XML reload, observation regeneration, and close through
the `LiberoBackendSession` contract. The legacy `env` attribute remains the same
robosuite task object, so existing official callers keep their current behavior.
The robosuite task remains authoritative for cameras, state, predicates, and
controller state. Local Python 3.12 evidence passes the 8 static G1 wrapper tests,
15 migration/MuJoCo compatibility tests, and the real headless official golden.

## 11. G3 first runnable Warp slice

The first G3 slice is runnable but does not yet claim the complete G3 hard gate.
`make_env(EnvConfig(..., backend="warp", num_worlds=1))` now constructs a
`WarpBatchEnv` backed by the existing exact `TaskCompiler` and `MJWarpSpike`.
It provides:

- trusted init-bank reset for world zero;
- caller-ordered cameras with unequal resolutions, RGB, optional metric depth,
  and optional segmentation;
- CUDA `ObservationBatch` outputs for visuals, reset-boundary proprioception,
  FULLPHYSICS state, and simulator time;
- state and render-exact snapshot reads;
- idempotent close and explicit closed-object failures;
- a clear `NotImplementedError` for policy `step()` until Warp `OSC_POSE` exists.

There is no official physics or rendering fallback in the returned observation.
The official compiler-owned task is used only at reset to obtain authoritative
controller-derived proprioception, which cannot be reconstructed from MuJoCo
qpos/qvel alone.

AutoDL smoke evidence used an RTX 4090 D with the server's existing
Torch 2.13.0+cu130 / CUDA 13 stack plus isolated MuJoCo 3.11.0,
mujoco-warp 3.11.0, Warp 1.16.0, and robosuite 1.5.2. The public GPU test passed
in 14.79 seconds after the one-time Warp kernel cache was populated. It exercised
three unequal-resolution cameras, RGB, depth, instance segmentation, two trusted
resets, state/proprio/time reads, snapshot, step rejection, and repeated close.
All returned observation tensors were finite and on `cuda:0`.

The project runtime remains qualified on Python 3.12 by the G1 workflow; this
fast GPU smoke reused the available Python 3.10 CUDA environment and is recorded
as such rather than presented as a second Python qualification.

The regenerated schema-v2 manifest SHA-256 is
`ff90f81eeb427cbb3358f589b93842d720f0f1a0b2b1225c002a1ab4871166f2`.

Still required before declaring the full G3 gate complete:

- official-vs-Warp reset/state/camera oracle coverage across representative
  suites, scenes, objects, and segmentation modes;
- predicate/minimal task readout;
- the original `ControlEnv` observation-dict compatibility surface;
- repeated-construction resource-growth evidence.
