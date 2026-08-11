# LIBERO-warp AutoDL workflow

This is the default path for personal development and GPU smoke tests. Its goal
is to reach a real runtime result quickly without rebuilding a release matrix or
redownloading the CUDA stack.

## 1. Connect in the right mode

The `autodl` SSH alias may define a reverse forward on remote port 1080. A second
connection can fail with `remote port forwarding failed for listen port 1080`.

For normal shell commands that do not need proxy access:

```bash
ssh -o ClearAllForwardings=yes -S none autodl
```

For GitHub, establish one intentional reverse-proxy connection and reuse it.
Verify the remote proxy before fetching:

```bash
curl -x http://127.0.0.1:1080 -I https://github.com
```

Then set `HTTPS_PROXY` / `https_proxy` for that command. Prefer this trusted local
proxy over cycling through public GitHub mirrors. If a repository-specific mirror
is already configured and verified, it is the fallback; raw GitHub over a known
slow link is not the first attempt.

## 2. Inventory before installing

Run one probe before any dependency operation:

```bash
python3 --version
nvidia-smi
uv --version
df -h /root/autodl-tmp
python3 - <<'PY'
import importlib.util

for name in ("torch", "mujoco", "robosuite", "warp", "mujoco_warp"):
    spec = importlib.util.find_spec(name)
    print(name, spec.origin if spec else "MISSING")
PY
```

Also inspect known project environments and uv caches. The required compatibility
checks are small:

- Torch must see CUDA and the selected GPU.
- The CUDA driver must support the Torch build; do not pin or install individual
  `nvidia-*` wheels manually.
- `mujoco` and `mujoco-warp` must be API-compatible. A missing symbol such as
  `mujoco.mjMINAWAKE` means MuJoCo is too old for MJWarp; upgrade MuJoCo in the
  isolated project target instead of replacing Torch/CUDA.
- Use Python 3.12 for project qualification. A quick GPU smoke may reuse an
  existing compatible interpreter, but record that difference explicitly. If a
  Python 3.12 GPU environment is needed, create it once and retain it.

## 3. Install through the fast path

Default PyPI index:

```bash
uv pip install --index-url https://mirrors.aliyun.com/pypi/simple ...
```

Do not begin with `uv sync` or install every optional extra. Install only the
profile needed by the current test. Reuse the server's working Torch/CUDA stack.

For a large wheel already named in `uv.lock`, use its exact `/packages/...` path
on the Aliyun mirror, then verify its SHA-256 before installing offline:

```bash
curl -L --fail --retry 3 -o package.whl \
  https://mirrors.aliyun.com/pypi/packages/.../package.whl
sha256sum package.whl
uv pip install --no-deps package.whl
```

This repository observed roughly 55 KiB/s from `files.pythonhosted.org` versus
about 18 MiB/s for the same 155 MiB Warp wheel through Aliyun. There is no reason
to wait several minutes before trying the established fast path.

Keep reusable wheels in a stable directory such as:

```text
/root/autodl-tmp/libero-warp-runtime/wheels/
```

Keep the environment or isolated target next to it. Do not store it in another
project's checkout and do not mutate another project's virtual environment.

## 4. Transfer only the delta

If the remote has a known baseline, send only changed files while preserving
their relative paths:

```bash
rsync -avR -e 'ssh -o ClearAllForwardings=yes -S none' \
  path/to/changed.py tests/test_changed.py \
  autodl:/root/autodl-tmp/libero-warp-worktree/
```

Before transferring, compare hashes. Avoid full `rsync`, GitHub codeload archives,
or cloning a second copy merely to move a handful of source files. Never include
`.git`, `.venv`, uv caches, build outputs, or unchanged assets in an incremental
smoke-test transfer.

## 5. Run the smallest real smoke first

For Warp, the order is:

1. import Torch, MuJoCo, robosuite, Warp, and MJWarp;
2. print versions, `torch.cuda.is_available()`, and GPU name;
3. construct one task and one world;
4. reset to one trusted init state;
5. verify CUDA state, proprioception, RGB, optional depth/segmentation, and sim time;
6. close the environment;
7. only then run broader parity, benchmarks, or multi-world tests.

Headless rendering uses:

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

## 6. Handle interruption cleanly

Interrupting local SSH does not guarantee that its remote child exited. After a
cancel or timeout, inspect the exact processes you started:

```bash
ps -eo pid,etime,stat,cmd | grep -E 'uv|curl|rsync'
```

Stop only those confirmed processes. Do not leave parallel package downloads
competing for the same slow link, and do not remove shared caches or unrelated
AutoDL data while cleaning up.

## Lessons from the G3 setup

- The server already had a CUDA-capable Torch build; trying a full frozen sync
  would have downloaded several GiB unnecessarily.
- A previous isolated environment was Python 3.11 because the old project pin
  asked for it. The maintained project now qualifies Python 3.12; do not rebuild
  3.11 for new work.
- The remote G2 source baseline was already current. Four changed files were
  enough for G3; a multi-megabyte historical patch was unnecessary.
- PyPI mirror selection should be a policy, not an experiment performed during
  each install. Aliyun is the default; use another source only after a concrete
  failure.
- Torch/CUDA flexibility does not imply all simulator packages are interchangeable.
  MuJoCo and MJWarp share a direct API boundary and must be checked together.
