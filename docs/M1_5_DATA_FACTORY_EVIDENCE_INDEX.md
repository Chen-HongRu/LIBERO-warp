# M1.5 Data Factory Qualification — Evidence Index

Status: `bounded qualification complete; production gate not passed`; host:
AutoDL RTX 4090D; evidence roots:

- baseline and first M1 artifacts: `/tmp/libero-m1-bottleneck-evidence-20260810`
- M1.5 remote artifacts: `/root/autodl-tmp/libero-m15-data-factory-evidence-20260810`

This index deliberately distinguishes an operational transition (25 physics
substeps, three 128px RGB views, and health checks pass) from a behavioral
transition (the task predicate / success criterion passes).  Neither label may
be inferred from the other.

## Frozen benchmark contract

- LIBERO Spatial task 0; 128 worlds; three 128×128 RGB cameras.
- 100 warmup and 1000 measured control operations, unless an artifact says
  otherwise.
- one operation is 25 physics substeps at 20 Hz control.
- all transition rates are aggregate batches, not per-world Hz.

## Evidence ledger

| Question | Direct artifact / result | Status and interpretation |
|---|---|---|
| Correct-layout Graph-1 physics | `formal-graph1-correct-layout-n128.json`: 31.102 batch operations/s, 3981.05 world transitions/s | Operationally valid for the tested trace/health contract; behavioral validity not established. |
| Correct-layout Graph-1 step + 3 RGB renders | same artifact: 16.638 batch operations/s, 2129.69 world transitions/s | Production upper baseline before materialization/storage. |
| Earlier 24/41 Hz Graph figures | superseded | **Invalid.** Graph staging was `[N,K,nu]` while offsets assumed `[K,N,nu]`; it also accepted a zero-stride expanded control source. Do not use. |
| M1.5 materialization smoke | `m15-qualification-io-n128.json` | 3 aggregate RGB batches: GPU→CPU 2.366 GB/s; raw disk 1.965 GB/s; deflate 10.393 MB/s (0.217 compression ratio). At formal rendering rate raw output is about 314 MB/s; synchronous deflate is the bottleneck. The short smoke rate itself is not a formal throughput result. |
| Official multi-process N=8 | `official-mp-attempt2-*` | 8/8 workers, steady aggregate 1403.876 transitions/s and 4211.629 RGB views/s; startup-inclusive 371.855 transitions/s. |
| Official multi-process N=16 | `official-mp-attempt2-*` | 16/16 workers, steady aggregate 2170.200 transitions/s and 6510.601 RGB views/s; startup-inclusive 500.862 transitions/s. This is the provisional production baseline. |
| Official multi-process N=32 | `official-mp-attempt2-*` | 29/32 worker JSON reports; three workers stopped during renderer initialization. Successful subset 1437.601 transitions/s, but this is **not a valid N=32 throughput**. It is evidence of EGL/GPU resource contention. |
| Behavioral replay, direct official environment | `official-positive-demo.json` | `libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5`, `demo_0`, 272 actions reaches official predicate true. |
| TaskCompiler seeded-model replay | `m15-adapter-parity.json`, `m15-adapter-first-divergence.json` | Negative control: controller, frequency, first control and first 50 actions agree, but portable model differs (body position max 0.00267 m) and trajectory first diverges at action 66. HDF fullphysics does not restore static model placement. **Not behavioral parity evidence.** |
| Exact demo model XML | `m15-hdf-exact-model*.json`, `strict-exact-official-demo0.json` | The remapper now selects a complete asset installation and ports bounded robosuite-1.4 single-Panda identifiers to the 1.5 runtime. The exact model loads and remains operational, but the strict official replay never reaches the task predicate. This is a controller/runtime-version qualification failure. |
| Invalid exact-reset diagnostic | `direct-vs-exact-seed4-report.json`, `exact-warp-eager-graph1-report.json` | **Not qualification evidence.** Calling high-level `reset(init_state)` after exact XML reruns the LIBERO placement sampler and changes model-level static fields. It can produce Official success and a Warp trace, but it no longer represents the HDF exact model. The API now rejects this misuse. |
| Strict exact model stability | `strict-exact-model-mutation*.json` | `body_pos` and all physics fields remain stable; only runtime visualization field `site_rgba` changes during the rollout. The pre-rollout portable fingerprint is the compile/Warp handoff gate; post-rollout full fingerprint changes are visual-state evidence, not static-physics drift. |
| Legacy-reference runtime | `legacy-reference/environment-selection.txt`, `existing-legacy-venv-freeze.txt` | Isolated existing uv environment: robosuite 1.4.0, MuJoCo 2.3.2, Python 3.10.12. It matches the HDF-era pins and leaves the current environment unchanged. |
| Legacy exact open-loop action replay | `legacy-reference/legacy-official-replay.{json,log}`, `legacy-exact-demo0-ctrl-trace.npz` | Operational: 272 actions and `[272,25,9]` controls captured, finite state, exact static body positions. Behavioral: never succeeds; divergence from recorded next-state starts at action 0. This proves exact XML + fullphysics is insufficient to reconstruct the historical OSC state. The trace is quarantined from behavioral qualification. |
| HDF state-playback positive control | `legacy-reference/legacy-teacher-forced-anchors.{json,npz,log}` | The exact legacy model and predicate are valid: recorded states first satisfy success at 261, last at 271, final true, 11/272 successful states. |
| Teacher-forced local dynamics | `legacy-reference/current-mujoco-warp-teacher-forced.{json,npz,log}` | Twelve one-control anchors span 5–36 initial contacts. Current MuJoCo 3.11 versus legacy 2.3 stays below 9.5e-5 qpos max error for 11 anchors but reaches 8.32e-3 at the success-boundary anchor 270. MJWarp versus current MuJoCo stays below 1.07e-4 qpos max error and remains healthy. Graph-25 versus eager stays below 7.72e-6. This is Tier-1 evidence only, not rollout/task-success evidence. |
| Closed-loop policy search | `legacy-reference/policy-checkpoint-candidates.txt`, `hf-model-cache.txt`, `policy-named-files.txt` | No executable LIBERO closed-loop policy checkpoint was found. Discovered checkpoints are LeWM/world-model training artifacts. Tier-2/3 needs an official policy checkpoint or a newly trained policy. |

## Async two-cohort spike — proposed, not measured

The valid experiment is two independent cohorts, A and B.  A advances its own
known action sequence while B renders its own previously completed state, then
they exchange roles.  No fullphysics/state crosses cohorts.  Acceptance needs:

1. independent state, renderer, camera buffers, and trajectory IDs;
2. a no-contamination check against sequential A/B trajectories;
3. measured wall time versus sequential `physics + render`;
4. explicit failure if renderer/global stream serialization prevents overlap.

It has **not** been measured because strict behavioral qualification failed; it
must not be used to claim overlap.

## Gates and current limits

- **Operational qualification:** the Graph path and teacher-forced local
  dynamics pass their bounded health/parity contracts, but only for the tested
  task/model/anchors.
- **Behavioral qualification:** **no-go on the current robosuite 1.5 runtime.**
  The published robosuite-1.4 HDF model loads, but strict exact Official replay
  is operationally healthy and behaviorally unsuccessful. The same exact
  open-loop replay also fails in the pinned legacy runtime because controller
  history is not in the HDF. Recorded-state playback proves the demo and
  predicate are positive; teacher-forced anchors prove only local dynamics.
- **Storage qualification:** raw writer bandwidth looks adequate at the formal
  render rate; a bounded asynchronous/multiprocess encoder queue is required
  before compressed-data claims.
- **CPU-vs-GPU:** current default is Official N=16.  Warp is not promoted until
  behavior and pipeline overlap are proven; both paths use the same GPU for
  EGL rendering, so the Official path is not CPU-only.

## Decision summary

| Decision | Verdict | Reason |
|---|---|---|
| M1.5 production data-factory qualification | **NO-GO today; GO as a bounded R&D candidate** | Operational Graph throughput and Tier-1 local dynamics are encouraging, but no closed-loop task-level behavioral qualification exists. |
| CPU versus GPU production default | **Official N=16** | Official steady throughput is 2170.20 transitions/s versus Warp's 2129.69 transitions/s in the current three-view contract (Warp is about 0.981×), and Official is behaviorally authoritative. |
| Async two-cohort pipeline | **NO-GO / unverified** | It was intentionally not measured after the behavioral prerequisite failed. No overlap gain may be claimed. |
| Backend-agnostic WM mainline | **GO for interface migration** | Keep `TrajectoryBatch`/`EnvBackend` semantics independent of LIBERO; do not yet make Warp the production backend. See `docs/design/backend_agnostic_trajectory_boundary.md`. |
| M2 GPU OSC | **Deferred / NO-GO as a milestone blocker** | A graphable GPU controller does not solve the missing task-level oracle and is outside the bounded M1.5 gate. |

## Reproduction commands for the bounded legacy qualification

Run legacy capture and anchors from the isolated environment:

```bash
HDF=/root/autodl-tmp/.stable-wm/datasets/libero_100_raw/libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5
EVID=/root/autodl-tmp/libero-m15-data-factory-evidence-20260810/legacy-reference
cd /root/autodl-tmp/LeWM/repo
MUJOCO_GL=egl uv run --no-sync python \
  /root/autodl-tmp/LIBERO-warp-validation/benchmarks/legacy_reference_replay.py \
  --demo-hdf5 "$HDF" --output-npz "$EVID/legacy-exact-demo0-ctrl-trace.npz" \
  --output-json "$EVID/legacy-official-replay.json"
MUJOCO_GL=egl uv run --no-sync python \
  /root/autodl-tmp/LIBERO-warp-validation/benchmarks/legacy_teacher_forced_anchors.py \
  --demo-hdf5 "$HDF" --output-npz "$EVID/legacy-teacher-forced-anchors.npz" \
  --output-json "$EVID/legacy-teacher-forced-anchors.json"
```

Run the current MuJoCo/MJWarp comparison without changing either environment:

```bash
cd /root/autodl-tmp/LIBERO-warp-validation
PYTHONPATH=$PWD:/root/autodl-tmp/libero-m1-overlay-py310/lib/python3.10/site-packages:/root/LeWM/.venv/lib/python3.10/site-packages \
  uv run --no-sync --project /root/LeWM python \
  benchmarks/current_teacher_forced_compare.py \
  --input-npz "$EVID/legacy-teacher-forced-anchors.npz" \
  --output-npz "$EVID/current-mujoco-warp-teacher-forced.npz" \
  --output-json "$EVID/current-mujoco-warp-teacher-forced.json"
```
