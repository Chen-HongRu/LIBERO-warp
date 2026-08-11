"""Batched, device-resident MJWarp spike for one compiled LIBERO task.

This module deliberately stops at actuator-control replay.  Mapping LIBERO's
7-D policy action to OSC torques belongs to M2; M1 validates the lower physics
and rendering boundary by replaying official actuator ``ctrl`` values.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco_warp as mjw
import torch
import warp as wp
from mujoco_warp._src.types import ObjType, OverflowType

from libero.libero.runtime.compiler import CompiledTask
from libero.libero.runtime.types import CameraConfig

if TYPE_CHECKING:
    from collections.abc import Sequence


@wp.kernel
def _unpack_abgr_to_uint8_nhwc(
    packed: wp.array2d(dtype=wp.uint32),
    rgb_adr: wp.array1d(dtype=wp.int32),
    camera_index: int,
    rgb_out: wp.array4d(dtype=wp.uint8),
) -> None:
    """Unpack MJWarp's ABGR framebuffer directly into public NHWC uint8."""
    world_id, pixel_id = wp.tid()
    width = rgb_out.shape[2]
    y = pixel_id // width
    x = pixel_id % width
    value = packed[world_id, rgb_adr[camera_index] + pixel_id]
    rgb_out[world_id, y, x, 0] = wp.uint8(value >> wp.uint32(16))
    rgb_out[world_id, y, x, 1] = wp.uint8(value >> wp.uint32(8))
    rgb_out[world_id, y, x, 2] = wp.uint8(value)


@dataclass(frozen=True, slots=True)
class WarpRenderBatch:
    """GPU render outputs in the frozen runtime image conventions."""

    rgb: OrderedDict[str, torch.Tensor]
    depth: OrderedDict[str, torch.Tensor]
    segmentation: OrderedDict[str, torch.Tensor]


class MJWarpSpike:
    """One task/model batched MJWarp physics and rendering boundary.

    All hot-path tensors stay on ``device``.  Checked operations synchronize at
    their explicit diagnostics boundary; benchmark runners use unchecked paths
    and synchronize only around the timed region.
    """

    def __init__(
        self,
        compiled: CompiledTask,
        *,
        num_worlds: int,
        device: torch.device | str = "cuda",
        nconmax: int = 512,
        nccdmax: int = 256,
        njmax: int = 4096,
        njmax_nnz: int = 65536,
        naconmax: int | None = None,
        nvmax: int | None = None,
    ) -> None:
        if not isinstance(compiled, CompiledTask):
            raise TypeError("compiled must be a CompiledTask.")
        self._assert_compiled_model_current(compiled)
        if num_worlds <= 0:
            raise ValueError("num_worlds must be positive.")
        self.compiled = compiled
        self.metadata = compiled.metadata
        self.num_worlds = num_worlds
        requested_device = torch.device(device)
        if requested_device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("MJWarpSpike requires an available CUDA device.")
        cuda_index = (
            torch.cuda.current_device()
            if requested_device.index is None
            else requested_device.index
        )
        self.device = torch.device("cuda", cuda_index)
        if self.metadata.target_num_worlds not in {1, num_worlds}:
            raise ValueError(
                "Compiled task world count does not match spike world count: "
                f"{self.metadata.target_num_worlds} != {num_worlds}."
            )
        wp.init()
        self._warp_device = wp.get_device(str(self.device))
        # Dynamics state is batched by ``make_data(nworld=...)``.  The model is
        # static for M1, so keep every optional model field shared explicitly.
        with wp.ScopedDevice(self._warp_device):
            self.model = mjw.put_model(compiled.model, batch_sizes={})
            make_data_kwargs = {
                "nworld": num_worlds,
                "nconmax": nconmax,
                "nccdmax": nccdmax,
                "njmax": njmax,
                "njmax_nnz": njmax_nnz,
            }
            # ``nconmax`` is the compatibility spelling retained by
            # mujoco-warp's wrapper.  These optional current-version knobs let
            # the investigation harness size shared contacts / compact solver
            # workspace from measured occupancy without changing M1 defaults.
            if naconmax is not None:
                make_data_kwargs["naconmax"] = naconmax
            if nvmax is not None:
                make_data_kwargs["nvmax"] = nvmax
            self.data = mjw.make_data(compiled.model, **make_data_kwargs)
        self._state_signature = int(mjw.State.FULLPHYSICS)
        # M1 reset/replay only needs FULLPHYSICS.  Avoid retaining redundant
        # qpos/qvel/act/time/init-state copies for every large batch runner.
        self._init_fullphysics = compiled.init_state_bank.fullphysics.to(
            self.device, dtype=torch.float32
        )
        self._reset_staging = torch.empty(
            (num_worlds, self._init_fullphysics.shape[1]),
            device=self.device,
            dtype=torch.float32,
        )
        self._active = torch.zeros(num_worlds, device=self.device, dtype=torch.bool)
        self._render_context = None
        self._render_index: dict[str, int] = {}
        self._render_cameras: tuple[CameraConfig, ...] = ()
        self._depth_offsets: dict[str, int] = {}
        self._seg_offsets: dict[str, int] = {}
        self._geometry_instance: torch.Tensor | None = None
        self._geometry_class: torch.Tensor | None = None
        self._rgb_tensors: dict[str, torch.Tensor] = {}
        self._rgb_buffers: dict[str, wp.array] = {}
        self._ctrl_graph = None
        self._ctrl_graph_mode: str | None = None
        self._ctrl_graph_staging: torch.Tensor | None = None
        self._ctrl_graph_staging_array: wp.array | None = None
        self._ctrl_graph_source: wp.array | None = None
        self._ctrl_graph_target: wp.array | None = None
        self._ctrl_graph_substeps: int | None = None
        self._closed = False

    @property
    def nu(self) -> int:
        self._ensure_open()
        return self.metadata.nu

    def reset_all(
        self, state_indices: torch.Tensor, *, check_health: bool = True
    ) -> None:
        """Reset all worlds from pre-uploaded init states without a host copy."""
        self._ensure_open()
        self._validate_indices(
            state_indices, expected=self.num_worlds, name="state_indices"
        )
        self.reset_all_prevalidated(state_indices)
        if check_health:
            self.assert_healthy("full reset")

    def reset_all_prevalidated(self, state_indices: torch.Tensor) -> None:
        """Unchecked device-only full reset for a prevalidated benchmark request."""
        self._ensure_open()
        state = self._init_fullphysics.index_select(0, state_indices)
        with wp.ScopedDevice(self._warp_device):
            mjw.reset_data(self.model, self.data)
            mjw.set_state(
                self.model, self.data, wp.from_torch(state), self._state_signature
            )
            mjw.forward(self.model, self.data)

    def reset_partial(
        self,
        world_ids: torch.Tensor,
        state_indices: torch.Tensor,
        *,
        check_health: bool = True,
    ) -> None:
        """Reset a prevalidated device subset; no gather/scatter helper is hot-path."""
        self._ensure_open()
        self._validate_indices(world_ids, upper=self.num_worlds, name="world_ids")
        self._validate_indices(
            state_indices, expected=world_ids.numel(), name="state_indices"
        )
        if torch.unique(world_ids).numel() != world_ids.numel():
            raise ValueError(
                "world_ids must be unique for a deterministic partial reset."
            )
        self.reset_partial_prevalidated(world_ids, state_indices)
        if check_health:
            self.assert_healthy("partial reset")

    def reset_partial_prevalidated(
        self, world_ids: torch.Tensor, state_indices: torch.Tensor
    ) -> None:
        """Unchecked partial reset with only device kernels for timed code."""
        self._ensure_open()
        self._reset_staging.index_copy_(
            0, world_ids, self._init_fullphysics.index_select(0, state_indices)
        )
        self._active.zero_()
        self._active.index_fill_(0, world_ids, True)
        with wp.ScopedDevice(self._warp_device):
            mjw.reset_data(self.model, self.data, wp.from_torch(self._active))
            mjw.set_state(
                self.model,
                self.data,
                wp.from_torch(self._reset_staging),
                self._state_signature,
                wp.from_torch(self._active),
            )
            mjw.forward(self.model, self.data)

    def reset_fullphysics(
        self, states: torch.Tensor, *, check_health: bool = True
    ) -> None:
        """Reset to trusted CUDA FULLPHYSICS states for trace replay only."""
        self._ensure_open()
        if (
            not isinstance(states, torch.Tensor)
            or states.device != self.device
            or states.dtype != torch.float32
            or states.shape != self._reset_staging.shape
        ):
            raise ValueError(
                "states must be float32 CUDA FULLPHYSICS [num_worlds, state_size]."
            )
        if not torch.isfinite(states).all():
            raise ValueError("states contains non-finite values.")
        self.reset_fullphysics_prevalidated(states)
        if check_health:
            self.assert_healthy("FULLPHYSICS reset")

    def reset_fullphysics_prevalidated(self, states: torch.Tensor) -> None:
        """Unchecked device-only trusted state reset for timed trace cycling."""
        self._ensure_open()
        with wp.ScopedDevice(self._warp_device):
            mjw.reset_data(self.model, self.data)
            mjw.set_state(
                self.model, self.data, wp.from_torch(states), self._state_signature
            )
            mjw.forward(self.model, self.data)

    def replay_ctrl(
        self,
        ctrl_sequence: torch.Tensor,
        *,
        check_health: bool = True,
        validate: bool = True,
    ) -> None:
        """Replay CUDA ``[N, K, nu]`` actuator controls one physics step at a time."""
        self._ensure_open()
        self._validate_ctrl_sequence(ctrl_sequence, validate=validate)
        with wp.ScopedDevice(self._warp_device):
            ctrl_target = wp.to_torch(self.data.ctrl)
            for substep in range(ctrl_sequence.shape[1]):
                ctrl_target.copy_(ctrl_sequence[:, substep].to(dtype=ctrl_target.dtype))
                mjw.step(self.model, self.data)
        if check_health:
            self.assert_healthy("ctrl replay")

    def capture_ctrl_replay_graph(
        self, ctrl_template: torch.Tensor, *, mode: str
    ) -> None:
        """Capture one exact fixed-length raw-ctrl replay using Warp CUDA Graphs.

        ``ctrl_template`` defines a stable device buffer shape only; every replay
        copies fresh values into an internal CUDA staging tensor.  ``graph-1``
        captures a single physics step and launches it 25 times, whereas
        ``graph-25`` captures all 25 distinct control-row copies plus steps.
        """
        self._ensure_open()
        if mode not in {"graph-1", "graph-25"}:
            raise ValueError("mode must be 'graph-1' or 'graph-25'.")
        self._validate_ctrl_sequence(ctrl_template, validate=False)
        if self._ctrl_graph is not None:
            raise RuntimeError(
                "A ctrl replay graph is already captured for this spike."
            )
        # Graph copy nodes address one physics substep across all worlds, so
        # keep the persistent source in [K, N, nu] order.  The public control
        # contract remains [N, K, nu].  Flattening [N, K, nu] and offsetting by
        # ``K`` would otherwise walk successive substeps of world 0 rather
        # than the same substep of every world.
        staging = torch.empty(
            (
                ctrl_template.shape[1],
                ctrl_template.shape[0],
                ctrl_template.shape[2],
            ),
            device=self.device,
            dtype=ctrl_template.dtype,
        )
        staging.copy_(ctrl_template.permute(1, 0, 2))
        source = wp.from_torch(staging.flatten())
        target = wp.from_torch(wp.to_torch(self.data.ctrl).flatten())
        row_size = self.num_worlds * self.nu
        with wp.ScopedCapture(
            device=self._warp_device, force_module_load=False
        ) as capture:
            if mode == "graph-1":
                mjw.step(self.model, self.data)
            else:
                for substep in range(ctrl_template.shape[1]):
                    wp.copy(
                        target,
                        source,
                        src_offset=substep * row_size,
                        count=row_size,
                    )
                    mjw.step(self.model, self.data)
        self._ctrl_graph = capture.graph
        self._ctrl_graph_mode = mode
        self._ctrl_graph_staging = staging
        self._ctrl_graph_staging_array = wp.from_torch(staging)
        self._ctrl_graph_source = source
        self._ctrl_graph_target = target
        self._ctrl_graph_substeps = int(ctrl_template.shape[1])

    def replay_captured_ctrl(
        self,
        ctrl_sequence: torch.Tensor,
        *,
        check_health: bool = True,
        validate: bool = True,
    ) -> None:
        """Replay a previously captured exact raw-ctrl graph with fresh CUDA input."""
        self._ensure_open()
        if (
            self._ctrl_graph is None
            or self._ctrl_graph_staging is None
            or self._ctrl_graph_staging_array is None
        ):
            raise RuntimeError("Call capture_ctrl_replay_graph before graph replay.")
        self._validate_ctrl_sequence(ctrl_sequence, validate=validate)
        if ctrl_sequence.shape[1] != self._ctrl_graph_substeps:
            raise ValueError("ctrl_sequence substeps do not match the captured graph.")
        # ``runner.trace`` is intentionally an ``expand`` view across worlds.
        # Torch materializes that broadcast and transposes it into the [K,N,nu]
        # graph-source layout.  Warp's ``from_torch`` import would not perform
        # either operation.  Torch and Warp use distinct CUDA streams here;
        # the explicit fence is required until a supported cross-stream event
        # hand-off is available.
        self._ctrl_graph_staging.copy_(ctrl_sequence.permute(1, 0, 2))
        torch.cuda.current_stream(self.device).synchronize()
        if self._ctrl_graph_mode == "graph-1":
            if self._ctrl_graph_source is None or self._ctrl_graph_target is None:
                raise RuntimeError("Captured graph control buffers are unavailable.")
            row_size = self.num_worlds * self.nu
            for substep in range(ctrl_sequence.shape[1]):
                wp.copy(
                    self._ctrl_graph_target,
                    self._ctrl_graph_source,
                    src_offset=substep * row_size,
                    count=row_size,
                )
                wp.capture_launch(self._ctrl_graph)
        else:
            wp.capture_launch(self._ctrl_graph)
        if check_health:
            self.assert_healthy("captured ctrl replay")

    def physics_readout(self) -> dict[str, torch.Tensor]:
        """Return zero-copy CUDA views for parity checks (qpos/qvel/body/site/time)."""
        self._ensure_open()
        return {
            "qpos": wp.to_torch(self.data.qpos),
            "qvel": wp.to_torch(self.data.qvel),
            "act": wp.to_torch(self.data.act),
            "time": wp.to_torch(self.data.time),
            "body_xpos": wp.to_torch(self.data.xpos),
            "body_xquat": wp.to_torch(self.data.xquat),
            "site_xpos": wp.to_torch(self.data.site_xpos),
            "site_xmat": wp.to_torch(self.data.site_xmat),
        }

    def fullphysics_state(self) -> torch.Tensor:
        """Materialize the current FULLPHYSICS state on CUDA for render snapshots."""
        self._ensure_open()
        state = torch.empty_like(self._reset_staging)
        with wp.ScopedDevice(self._warp_device):
            mjw.get_state(
                self.model, self.data, wp.from_torch(state), self._state_signature
            )
        return state

    def configure_renderer(self, cameras: Sequence[CameraConfig]) -> None:
        """Create one renderer for caller-ordered cameras of arbitrary resolutions."""
        self._ensure_open()
        self._assert_compiled_model_current(self.compiled)
        cameras = tuple(cameras)
        if not cameras:
            raise ValueError("At least one CameraConfig is required.")
        names = [camera.name for camera in cameras]
        if len(set(names)) != len(names):
            raise ValueError("CameraConfig names must be unique.")
        missing = [name for name in names if name not in self.metadata.camera_ids]
        if missing:
            raise ValueError(
                "Unknown cameras "
                f"{missing}; available={sorted(self.metadata.camera_ids)}."
            )
        # MJWarp keeps active cameras in model-ID order.  Preserve the public
        # caller order in output maps while translating each name to that index.
        ordered_by_id = sorted(
            cameras, key=lambda camera: self.metadata.camera_ids[camera.name]
        )
        active_ids = {self.metadata.camera_ids[camera.name] for camera in ordered_by_id}
        active_mask = [
            camera_id in active_ids for camera_id in range(self.compiled.model.ncam)
        ]
        with wp.ScopedDevice(self._warp_device):
            self._render_context = mjw.create_render_context(
                self.compiled.model,
                nworld=self.num_worlds,
                cam_res=[(camera.width, camera.height) for camera in ordered_by_id],
                render_rgb=[True] * len(ordered_by_id),
                render_depth=[camera.depth for camera in ordered_by_id],
                render_seg=[
                    camera.segmentation is not None for camera in ordered_by_id
                ],
                cam_active=active_mask,
                # Robosuite's offscreen context enables MuJoCo visual groups 1
                # and 2 (``vopt.geomgroup == [0, 1, 1, 0, 0, 0]``).  Group 0
                # holds semi-transparent collision proxies in the LIBERO
                # object assets; ray-casting them would incorrectly occlude
                # the visual mesh in RGB, depth, and element segmentation.
                enabled_geom_groups=[1, 2],
            )
        self._render_cameras = cameras
        self._render_index = {
            camera.name: index for index, camera in enumerate(ordered_by_id)
        }
        self._depth_offsets, self._seg_offsets = self._render_offsets(ordered_by_id)
        self._geometry_instance, self._geometry_class = self._make_geom_lookup_tables()
        with wp.ScopedDevice(self._warp_device):
            self._rgb_tensors = {
                camera.name: torch.empty(
                    (self.num_worlds, camera.height, camera.width, 3),
                    device=self.device,
                    dtype=torch.uint8,
                )
                for camera in cameras
            }
            self._rgb_buffers = {
                name: wp.from_torch(image) for name, image in self._rgb_tensors.items()
            }

    def render(self, *, check_health: bool = True) -> WarpRenderBatch:
        """Refit, render, and expose top-left NHWC GPU tensors without host copies."""
        self._ensure_open()
        if self._render_context is None:
            raise RuntimeError("Call configure_renderer(cameras) before render().")
        if check_health:
            self.assert_healthy("render")
        rc = self._render_context
        with wp.ScopedDevice(self._warp_device):
            mjw.refit_bvh(self.model, self.data, rc)
            mjw.render(self.model, self.data, rc)
        rgb: OrderedDict[str, torch.Tensor] = OrderedDict()
        depth: OrderedDict[str, torch.Tensor] = OrderedDict()
        segmentation: OrderedDict[str, torch.Tensor] = OrderedDict()
        raw_depth = (
            wp.to_torch(rc.depth_data)
            if any(camera.depth for camera in self._render_cameras)
            else None
        )
        raw_seg = (
            wp.to_torch(rc.seg_data)
            if any(camera.segmentation is not None for camera in self._render_cameras)
            else None
        )
        for camera in self._render_cameras:
            index = self._render_index[camera.name]
            image = self._rgb_buffers[camera.name]
            with wp.ScopedDevice(self._warp_device):
                wp.launch(
                    _unpack_abgr_to_uint8_nhwc,
                    dim=(self.num_worlds, camera.height * camera.width),
                    inputs=[rc.rgb_data, rc.rgb_adr, index],
                    outputs=[image],
                    device=self._warp_device,
                )
            rgb[camera.name] = self._rgb_tensors[camera.name]
            pixels = camera.height * camera.width
            if camera.depth:
                if raw_depth is None:
                    raise RuntimeError(
                        "Renderer did not allocate requested depth data."
                    )
                start = self._depth_offsets[camera.name]
                # ``rc.depth_data`` is metric planar camera-Z.  Do not use
                # get_depth(), which divides and clamps to normalized [0, 1].
                depth[camera.name] = (
                    raw_depth[:, start : start + pixels]
                    .reshape(self.num_worlds, camera.height, camera.width, 1)
                    .to(torch.float32)
                )
            if camera.segmentation is not None:
                if raw_seg is None:
                    raise RuntimeError(
                        "Renderer did not allocate requested segmentation data."
                    )
                start = self._seg_offsets[camera.name]
                raw_pair = raw_seg[:, start : start + pixels].reshape(
                    self.num_worlds, camera.height, camera.width, 2
                )
                # MJWarp stores (object id, object type).  For geom masks the
                # id is channel 0; background is (-1, -1), while flex and any
                # future non-geom object is deliberately treated as background
                # to preserve robosuite's geom-based segmentation convention.
                raw_geom = torch.where(
                    raw_pair[..., 1] == int(ObjType.GEOM),
                    raw_pair[..., 0],
                    torch.full_like(raw_pair[..., 0], -1),
                )
                segmentation[camera.name] = self._segmentation_tensor(
                    raw_geom, camera.segmentation
                )
        return WarpRenderBatch(rgb=rgb, depth=depth, segmentation=segmentation)

    def assert_healthy(self, operation: str) -> None:
        """Synchronizing diagnostic check, for explicit non-timed boundaries only."""
        self._ensure_open()
        readout = self.physics_readout()
        nonfinite = [
            name for name, value in readout.items() if not torch.isfinite(value).all()
        ]
        overflow = wp.to_torch(self.data.overflow)
        if nonfinite or torch.any(overflow != 0):
            # Python branching on CUDA reductions already synchronizes.  This
            # method is intentionally called only at explicit benchmark timing
            # boundaries and before returning a checked public result.
            torch.cuda.synchronize(self.device)
            worlds = (
                torch.nonzero(overflow != 0, as_tuple=False).flatten().cpu().tolist()
            )
            bits = overflow.detach().cpu().tolist()
            decoded = {
                world: [flag.name for flag in OverflowType if int(bitmask) & int(flag)]
                for world, bitmask in enumerate(bits)
                if bitmask
            }
            raise RuntimeError(
                f"MJWarp {operation} failed: nonfinite={nonfinite}, "
                f"overflow_worlds={worlds}, overflow_bits={bits}, "
                f"overflow_flags={decoded}."
            )

    def synchronize(self) -> None:
        self._ensure_open()
        torch.cuda.synchronize(self.device)

    def close(self) -> None:
        """Idempotently release model, renderer, CUDA buffers, and official EGL."""
        if self._closed:
            return
        self._closed = True
        compiled = self.compiled
        self._render_context = None
        self._rgb_tensors.clear()
        self._rgb_buffers.clear()
        self._ctrl_graph = None
        self._ctrl_graph_mode = None
        self._ctrl_graph_staging = None
        self._ctrl_graph_staging_array = None
        self._ctrl_graph_source = None
        self._ctrl_graph_target = None
        self._ctrl_graph_substeps = None
        self._geometry_instance = None
        self._geometry_class = None
        self._render_cameras = ()
        self._render_index.clear()
        self._depth_offsets.clear()
        self._seg_offsets.clear()
        self._init_fullphysics = None
        self._reset_staging = None
        self._active = None
        self.data = None
        self.model = None
        self.metadata = None
        self.compiled = None
        compiled.close()

    def _validate_indices(
        self,
        indices: torch.Tensor,
        *,
        name: str,
        upper: int | None = None,
        expected: int | None = None,
    ) -> None:
        if not isinstance(indices, torch.Tensor) or indices.device != self.device:
            raise TypeError(f"{name} must be a CUDA tensor on {self.device}.")
        if indices.dtype != torch.int64 or indices.ndim != 1:
            raise TypeError(f"{name} must be a one-dimensional torch.int64 tensor.")
        if indices.numel() == 0:
            raise ValueError(f"{name} must not be empty.")
        if expected is not None and indices.numel() != expected:
            raise ValueError(f"{name} must contain {expected} values.")
        bound = self._init_fullphysics.shape[0] if upper is None else upper
        if torch.any(indices < 0) or torch.any(indices >= bound):
            raise IndexError(f"{name} contains an out-of-range index.")

    def _validate_ctrl_sequence(
        self, ctrl_sequence: torch.Tensor, *, validate: bool
    ) -> None:
        if not isinstance(ctrl_sequence, torch.Tensor):
            raise TypeError("ctrl_sequence must be a CUDA torch.Tensor.")
        if ctrl_sequence.device != self.device or ctrl_sequence.ndim != 3:
            raise ValueError(
                "ctrl_sequence must have shape [N, K, nu] on the spike device."
            )
        if (
            ctrl_sequence.shape[0] != self.num_worlds
            or ctrl_sequence.shape[2] != self.nu
        ):
            raise ValueError(
                "ctrl_sequence must have shape "
                f"[{self.num_worlds}, K, {self.nu}], got {list(ctrl_sequence.shape)}."
            )
        if not torch.is_floating_point(ctrl_sequence):
            raise TypeError("ctrl_sequence must use a floating-point dtype.")
        if validate and not torch.isfinite(ctrl_sequence).all():
            raise ValueError("ctrl_sequence contains non-finite values.")

    @staticmethod
    def _assert_compiled_model_current(compiled: CompiledTask) -> None:
        """Reject closed or hard-reset official environments before model handoff."""
        official_env = compiled.official_env
        if getattr(official_env, "_closed", True):
            raise RuntimeError(
                "CompiledTask official environment is closed; compile a fresh task "
                "before constructing or configuring MJWarpSpike."
            )
        try:
            current_model = official_env._env.sim.model._model
        except AttributeError as error:
            raise RuntimeError(
                "CompiledTask no longer exposes its current official MuJoCo model; "
                "compile a fresh task before using MJWarpSpike."
            ) from error
        if compiled.model is not current_model:
            raise RuntimeError(
                "CompiledTask model is stale after an official hard reset; compile a "
                "fresh task before using MJWarpSpike."
            )

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("MJWarpSpike is closed and cannot be used.")

    @staticmethod
    def _render_offsets(
        cameras: Sequence[CameraConfig],
    ) -> tuple[dict[str, int], dict[str, int]]:
        depth_offsets: dict[str, int] = {}
        seg_offsets: dict[str, int] = {}
        depth_offset = 0
        seg_offset = 0
        for camera in cameras:
            pixels = camera.height * camera.width
            if camera.depth:
                depth_offsets[camera.name] = depth_offset
                depth_offset += pixels
            if camera.segmentation is not None:
                seg_offsets[camera.name] = seg_offset
                seg_offset += pixels
        return depth_offsets, seg_offsets

    def _make_geom_lookup_tables(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compile robosuite geom-to-instance/class maps once onto CUDA."""
        model = self.compiled.official_env._env.env.model
        default = -torch.ones(self.compiled.model.ngeom, dtype=torch.int32)
        instance = default.clone()
        class_ids = default.clone()
        for source, ordered_names, target in (
            (
                getattr(model, "geom_ids_to_instances", {}),
                getattr(model, "instances_to_ids", {}),
                instance,
            ),
            (
                getattr(model, "geom_ids_to_classes", {}),
                getattr(model, "classes_to_ids", {}),
                class_ids,
            ),
        ):
            # robosuite maps geom ids to semantic *names*.  Its public
            # segmentation conversion enumerates the insertion-ordered keys of
            # instances_to_ids/classes_to_ids and then adds one for background.
            name_to_id = {name: index for index, name in enumerate(ordered_names)}
            for geom_id, value in source.items():
                if value not in name_to_id:
                    continue
                target[int(geom_id)] = name_to_id[value]
        return instance.to(self.device), class_ids.to(self.device)

    def _segmentation_tensor(self, raw_geom: torch.Tensor, kind: str) -> torch.Tensor:
        raw_geom = raw_geom.to(torch.int64)
        if kind == "element":
            mapped = raw_geom
        else:
            lookup = (
                self._geometry_instance if kind == "instance" else self._geometry_class
            )
            if lookup is None:
                raise RuntimeError("Segmentation lookup tables were not initialized.")
            mapped = torch.zeros_like(raw_geom)
            valid = (raw_geom >= 0) & (raw_geom < lookup.numel())
            mapped[valid] = lookup[raw_geom[valid]].to(torch.int64) + 1
            mapped = torch.clamp(mapped, min=0)
        return mapped.to(torch.int32).unsqueeze(-1)
