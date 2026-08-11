## Working agreements

- Use `uv` for Python environments and dependency operations.
- After changing Python files, run Ruff through `uvx` for formatting and linting.
- This repository is maintained primarily for the owner's personal research use.
  Prioritize a working vertical slice and real runtime evidence over exhaustive
  release qualification unless the user explicitly requests release hardening.

## AutoDL default workflow

- Read `docs/AUTODL_WORKFLOW.md` before any AutoDL setup, dependency install,
  source transfer, or GPU validation for this repository.
- Use the SSH alias `autodl` and place all project work below `/root/autodl-tmp/`.
- For ordinary commands that do not need the proxy tunnel, use
  `ssh -o ClearAllForwardings=yes -S none autodl` so a stale remote port 1080
  forwarding cannot block the connection.
- For GitHub access, prefer one deliberately managed local reverse-proxy tunnel.
  Do not repeatedly try raw GitHub clone, codeload, and unrelated mirrors.
- For PyPI, use `https://mirrors.aliyun.com/pypi/simple` by default. For a known
  large wheel, prefer the corresponding Aliyun `/pypi/packages/...` direct URL
  and verify its SHA-256 against `uv.lock` or trusted package metadata.
- Never start with a full `uv sync` on AutoDL. Inventory Python, Torch, CUDA,
  MuJoCo, robosuite, Warp, MJWarp, disk, and caches first. Reuse the installed
  Torch/CUDA stack and add only missing or incompatible packages into a
  project-owned environment or target directory.
- Keep downloaded wheels and a stable runtime environment under
  `/root/autodl-tmp/`; do not recreate them for each validation run.
- Transfer only changed files with checksum-aware `rsync`. Do not copy the full
  repository, assets, `.git`, virtual environments, or caches when the remote
  already has the baseline.
- Use `MUJOCO_GL=egl` and `PYOPENGL_PLATFORM=egl` for headless rendering.
- After an interrupted SSH command, check for and stop only the remote processes
  started by that command; closing the local SSH process may leave remote `uv`,
  `curl`, or `rsync` processes alive.

## Multi-agent preferences

- Delegate only concrete, bounded work that can proceed independently.
- Workers must preserve other agents' edits, avoid unrelated refactors, and
  report changed files, validation results, and remaining risks.
