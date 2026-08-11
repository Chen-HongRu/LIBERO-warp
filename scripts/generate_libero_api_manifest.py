"""Generate a deterministic static API and package-surface manifest for LIBERO.

The compatibility target is read directly from Git objects. The current snapshot is
read from the working tree, so the generated report also exposes removals and local
signature drift without importing LIBERO or any of its optional dependencies.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA_VERSION = 2
MODULE_ROOTS = ("libero", "benchmark_scripts", "scripts")
SOURCE_ROOTS = MODULE_ROOTS + ("templates",)
UPSTREAM_URL = "https://github.com/Lifelong-Robot-Learning/LIBERO.git"
DEFAULT_TARGET_COMMIT = "8f1084e3132a39270c3a13ebe37270a43ece2a01"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def resolve_commit(repo: Path, ref: str) -> str:
    return _git(repo, "rev-parse", "--verify", f"{ref}^{{commit}}").strip()


def _module_name(path: str) -> str | None:
    pure = PurePosixPath(path)
    if pure.suffix != ".py":
        return None
    parts = list(pure.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts) if parts else None


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    signature = f"({ast.unparse(node.args)})"
    if node.returns is not None:
        signature += f" -> {ast.unparse(node.returns)}"
    return signature


def _decorators(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    return [ast.unparse(decorator) for decorator in node.decorator_list]


def _assigned_names(target: ast.expr) -> list[str]:
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        return [name for item in target.elts for name in _assigned_names(item)]
    return []


def _literal_string_list(node: ast.AST) -> list[str] | None:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _literal_string_list(node.left)
        right = _literal_string_list(node.right)
        if left is not None and right is not None:
            return left + right
        return None
    try:
        value = ast.literal_eval(node)
    except (ValueError, TypeError):
        return None
    if isinstance(value, (list, tuple)) and all(
        isinstance(item, str) for item in value
    ):
        return list(value)
    return None


def _resolve_import(module: str, is_package: bool, level: int, name: str | None) -> str:
    if level == 0:
        return name or ""
    package = module if is_package else module.rpartition(".")[0]
    relative = "." * level + (name or "")
    return importlib.util.resolve_name(relative, package)


def _parse_module(path: str, source: str) -> dict[str, Any]:
    module = _module_name(path)
    if module is None:
        raise ValueError(f"not a Python module: {path}")
    is_package = PurePosixPath(path).name == "__init__.py"
    tree = ast.parse(source, filename=path)
    symbols: dict[str, dict[str, Any]] = {}
    star_imports: list[str] = []
    explicit_all: list[str] | None = None
    explicit_all_unresolved = False

    def record_symbol(
        name: str, symbol: dict[str, Any], *, conditional: bool = False
    ) -> None:
        if conditional:
            symbol = {**symbol, "availability": "conditional"}
        existing = symbols.get(name)
        if existing is None:
            symbols[name] = symbol
            return
        alternatives = (
            existing["alternatives"]
            if existing.get("kind") == "conditional_binding"
            else [existing]
        )
        candidates = alternatives + [symbol]
        unique = {
            json.dumps(candidate, sort_keys=True): candidate for candidate in candidates
        }
        if len(unique) == 1:
            symbols[name] = next(iter(unique.values()))
            return
        symbols[name] = {
            "kind": "conditional_binding",
            "alternatives": [unique[key] for key in sorted(unique)],
        }

    def visit_statements(
        statements: list[ast.stmt], *, conditional: bool = False
    ) -> None:
        nonlocal explicit_all, explicit_all_unresolved
        for node in statements:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                record_symbol(
                    node.name,
                    {
                        "kind": "async_function"
                        if isinstance(node, ast.AsyncFunctionDef)
                        else "function",
                        "signature": _signature(node),
                        "decorators": _decorators(node),
                    },
                    conditional=conditional,
                )
            elif isinstance(node, ast.ClassDef):
                members: dict[str, dict[str, Any]] = {}
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        members[child.name] = {
                            "kind": "async_method"
                            if isinstance(child, ast.AsyncFunctionDef)
                            else "method",
                            "signature": _signature(child),
                            "decorators": _decorators(child),
                        }
                init = members.get("__init__")
                record_symbol(
                    node.name,
                    {
                        "kind": "class",
                        "signature": init["signature"] if init else "()",
                        "bases": [ast.unparse(base) for base in node.bases],
                        "members": members,
                    },
                    conditional=conditional,
                )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    exported_name = alias.asname or alias.name.split(".", 1)[0]
                    record_symbol(
                        exported_name,
                        {"kind": "module_import", "origin_module": alias.name},
                        conditional=conditional,
                    )
            elif isinstance(node, ast.ImportFrom):
                origin_module = _resolve_import(
                    module, is_package, node.level, node.module
                )
                for alias in node.names:
                    if alias.name == "*":
                        star_imports.append(origin_module)
                        continue
                    exported_name = alias.asname or alias.name
                    record_symbol(
                        exported_name,
                        {
                            "kind": "symbol_import",
                            "origin_module": origin_module,
                            "origin_name": alias.name,
                        },
                        conditional=conditional,
                    )
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
                value = node.value
                names = [name for target in targets for name in _assigned_names(target)]
                if "__all__" in names and value is not None:
                    parsed_all = _literal_string_list(value)
                    if conditional or parsed_all is None:
                        explicit_all_unresolved = True
                    else:
                        explicit_all = parsed_all
                for name in names:
                    if name == "__all__":
                        continue
                    symbol: dict[str, Any] = {"kind": "assignment"}
                    if value is not None:
                        try:
                            literal = ast.literal_eval(value)
                        except (ValueError, TypeError):
                            literal = None
                        if literal is not None and isinstance(
                            literal, (str, int, float, bool, type(None))
                        ):
                            symbol["value"] = literal
                    record_symbol(name, symbol, conditional=conditional)
            elif isinstance(node, ast.AugAssign):
                names = _assigned_names(node.target)
                if "__all__" in names and isinstance(node.op, ast.Add):
                    items = _literal_string_list(node.value)
                    if conditional or explicit_all is None or items is None:
                        explicit_all_unresolved = True
                    else:
                        explicit_all.extend(items)
                for name in names:
                    if name != "__all__":
                        record_symbol(
                            name, {"kind": "assignment"}, conditional=conditional
                        )
            elif (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and isinstance(node.value.func.value, ast.Name)
                and node.value.func.value.id == "__all__"
                and node.value.func.attr in {"append", "extend"}
            ):
                call = node.value
                if conditional or explicit_all is None or len(call.args) != 1:
                    explicit_all_unresolved = True
                    continue
                if call.func.attr == "append":
                    try:
                        item = ast.literal_eval(call.args[0])
                    except (ValueError, TypeError):
                        item = None
                    if isinstance(item, str):
                        explicit_all.append(item)
                    else:
                        explicit_all_unresolved = True
                else:
                    items = _literal_string_list(call.args[0])
                    if items is None:
                        explicit_all_unresolved = True
                    else:
                        explicit_all.extend(items)
            elif isinstance(node, ast.If):
                visit_statements(node.body, conditional=True)
                visit_statements(node.orelse, conditional=True)
            elif isinstance(node, (ast.For, ast.AsyncFor)):
                for name in _assigned_names(node.target):
                    record_symbol(name, {"kind": "loop_target"}, conditional=True)
                visit_statements(node.body, conditional=True)
                visit_statements(node.orelse, conditional=True)
            elif isinstance(node, ast.While):
                visit_statements(node.body, conditional=True)
                visit_statements(node.orelse, conditional=True)
            elif isinstance(node, (ast.With, ast.AsyncWith)):
                for item in node.items:
                    if item.optional_vars is None:
                        continue
                    for name in _assigned_names(item.optional_vars):
                        record_symbol(name, {"kind": "with_target"}, conditional=True)
                visit_statements(node.body, conditional=True)
            elif isinstance(node, ast.Try):
                visit_statements(node.body, conditional=True)
                for handler in node.handlers:
                    visit_statements(handler.body, conditional=True)
                visit_statements(node.orelse, conditional=True)
                visit_statements(node.finalbody, conditional=True)
            elif isinstance(node, ast.Match):
                for case in node.cases:
                    visit_statements(case.body, conditional=True)

    visit_statements(tree.body)

    return {
        "path": path,
        "is_package": is_package,
        "sha256": hashlib.sha256(source.encode()).hexdigest(),
        "declared_symbols": dict(sorted(symbols.items())),
        "star_imports": sorted(set(star_imports)),
        "explicit_all": explicit_all,
        "explicit_all_unresolved": explicit_all_unresolved,
    }


def _copy_origin(symbol: dict[str, Any], module: str, name: str) -> dict[str, Any]:
    copied = json.loads(json.dumps(symbol, sort_keys=True))
    copied["origin"] = f"{module}:{name}"
    return copied


def _resolve_exports(modules: dict[str, dict[str, Any]]) -> None:
    cache: dict[str, dict[str, dict[str, Any]]] = {}

    def exports(module: str, stack: frozenset[str]) -> dict[str, dict[str, Any]]:
        if module in cache:
            return cache[module]
        if module in stack or module not in modules:
            return {}
        record = modules[module]
        declared = record["declared_symbols"]
        names = record["explicit_all"]
        if names is None:
            names = [name for name in declared if not name.startswith("_")]
        resolved: dict[str, dict[str, Any]] = {}
        next_stack = stack | {module}
        for name in names:
            symbol = declared.get(name)
            if symbol is None:
                continue
            if symbol["kind"] == "symbol_import":
                origin_module = symbol["origin_module"]
                origin_name = symbol["origin_name"]
                origin = exports(origin_module, next_stack).get(origin_name)
                if origin is None:
                    origin = (
                        modules.get(origin_module, {})
                        .get("declared_symbols", {})
                        .get(origin_name)
                    )
                resolved[name] = (
                    _copy_origin(origin, origin_module, origin_name)
                    if origin is not None
                    else symbol
                )
            else:
                resolved[name] = symbol
        if record["explicit_all"] is None:
            for origin_module in record["star_imports"]:
                for name, symbol in exports(origin_module, next_stack).items():
                    if not name.startswith("_"):
                        resolved.setdefault(
                            name, _copy_origin(symbol, origin_module, name)
                        )
        cache[module] = dict(sorted(resolved.items()))
        return cache[module]

    for module in sorted(modules):
        modules[module]["exports"] = exports(module, frozenset())


@dataclass(frozen=True)
class Snapshot:
    files: dict[str, str]
    entries: dict[str, dict[str, int | str]]
    metadata: dict[str, Any]


def _git_snapshot(repo: Path, ref: str, roots: tuple[str, ...]) -> Snapshot:
    commit = resolve_commit(repo, ref)
    raw_entries = _git(repo, "ls-tree", "-r", "-l", commit, "--", *roots)
    entries: dict[str, dict[str, int | str]] = {}
    for line in raw_entries.splitlines():
        metadata, path = line.split("\t", 1)
        _mode, object_type, oid, size = metadata.split()
        if object_type != "blob":
            continue
        entries[path] = {"git_blob_oid": oid, "size": int(size)}
    files = {
        path: _git(repo, "show", f"{commit}:{path}") if path.endswith(".py") else ""
        for path in entries
    }
    committed_at = _git(repo, "show", "-s", "--format=%cI", commit).strip()
    tree = _git(repo, "rev-parse", f"{commit}^{{tree}}").strip()
    return Snapshot(
        files=files,
        entries=entries,
        metadata={
            "ref": ref,
            "commit": commit,
            "tree": tree,
            "committed_at": committed_at,
        },
    )


def _working_tree_blob_metadata(path: Path) -> dict[str, int | str]:
    size = path.stat().st_size
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {size}\0".encode())
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"git_blob_oid": digest.hexdigest(), "size": size}


def _working_tree_snapshot(repo: Path, roots: tuple[str, ...]) -> Snapshot:
    files: dict[str, str] = {}
    entries: dict[str, dict[str, int | str]] = {}
    for root_name in roots:
        root = repo / root_name
        if not root.exists():
            continue
        candidates = [root] if root.is_file() else root.rglob("*")
        for path in candidates:
            if not path.is_file() or any(
                part in {".git", "__pycache__", ".pytest_cache", ".ruff_cache"}
                for part in path.parts
            ):
                continue
            relative = path.relative_to(repo).as_posix()
            files[relative] = path.read_text() if path.suffix == ".py" else ""
            entries[relative] = _working_tree_blob_metadata(path)
    return Snapshot(
        files=files,
        entries=entries,
        metadata={
            "head_commit": resolve_commit(repo, "HEAD"),
        },
    )


def _under_roots(path: str, roots: tuple[str, ...]) -> bool:
    return any(path == root or path.startswith(f"{root}/") for root in roots)


def _modules(
    snapshot: Snapshot, module_roots: tuple[str, ...]
) -> dict[str, dict[str, Any]]:
    modules: dict[str, dict[str, Any]] = {}
    for path, source in sorted(snapshot.files.items()):
        if path == "setup.py" or not _under_roots(path, module_roots):
            continue
        module = _module_name(path)
        if module is None:
            continue
        try:
            modules[module] = _parse_module(path, source)
        except SyntaxError as exc:
            modules[module] = {
                "path": path,
                "parse_error": f"{exc.msg} at {exc.lineno}:{exc.offset}",
                "sha256": hashlib.sha256(source.encode()).hexdigest(),
                "declared_symbols": {},
                "star_imports": [],
                "explicit_all": None,
                "explicit_all_unresolved": False,
            }
    _resolve_exports(modules)
    return dict(sorted(modules.items()))


def _target_cli(snapshot: Snapshot) -> dict[str, str]:
    source = snapshot.files.get("setup.py")
    if source is None:
        return {}
    tree = ast.parse(source, filename="setup.py")
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id != "setup":
            continue
        for keyword in node.keywords:
            if keyword.arg != "entry_points":
                continue
            try:
                entry_points = ast.literal_eval(keyword.value)
            except (ValueError, TypeError):
                return {}
            scripts = entry_points.get("console_scripts", [])
            return dict(sorted(item.split("=", 1) for item in scripts))
    return {}


def _target_distribution(snapshot: Snapshot) -> dict[str, Any]:
    source = snapshot.files.get("setup.py")
    if source is None:
        return {}
    tree = ast.parse(source, filename="setup.py")
    selected = {
        "name",
        "version",
        "python_requires",
        "include_package_data",
        "eager_resources",
        "packages",
    }
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id != "setup":
            continue
        metadata: dict[str, Any] = {}
        for keyword in node.keywords:
            if keyword.arg not in selected:
                continue
            try:
                metadata[keyword.arg] = ast.literal_eval(keyword.value)
            except (ValueError, TypeError):
                metadata[keyword.arg] = {"expression": ast.unparse(keyword.value)}
        return metadata
    return {}


def _current_cli(repo: Path) -> dict[str, str]:
    pyproject = repo / "pyproject.toml"
    if not pyproject.exists():
        return {}
    with pyproject.open("rb") as handle:
        project = tomllib.load(handle).get("project", {})
    return dict(sorted(project.get("scripts", {}).items()))


def _current_distribution(repo: Path) -> dict[str, Any]:
    with (repo / "pyproject.toml").open("rb") as handle:
        pyproject = tomllib.load(handle)
    project = pyproject.get("project", {})
    setuptools = pyproject.get("tool", {}).get("setuptools", {})
    return {
        "name": project.get("name"),
        "version": project.get("version"),
        "python_requires": project.get("requires-python"),
        "package_discovery": setuptools.get("packages", {}),
        "package_data": setuptools.get("package-data", {}),
    }


def _is_source_data(path: str) -> bool:
    return (
        path.startswith("templates/")
        or "/templates/" in path
        or not path.endswith(".py")
    )


def _source_data_files(snapshot: Snapshot) -> dict[str, dict[str, int | str]]:
    return {
        path: snapshot.entries[path]
        for path in sorted(snapshot.entries)
        if _under_roots(path, SOURCE_ROOTS) and _is_source_data(path)
    }


def _semantic_symbol_descriptor(
    symbol: dict[str, Any], *, include_members: bool = True
) -> dict[str, Any]:
    descriptor = {
        key: symbol[key]
        for key in (
            "kind",
            "signature",
            "decorators",
            "bases",
            "origin",
            "origin_module",
            "origin_name",
            "value",
            "availability",
        )
        if key in symbol
    }
    if "alternatives" in symbol:
        descriptor["alternatives"] = [
            _semantic_symbol_descriptor(alternative)
            for alternative in symbol["alternatives"]
        ]
    if include_members and "members" in symbol:
        descriptor["members"] = {
            name: _semantic_symbol_descriptor(member)
            for name, member in sorted(symbol["members"].items())
        }
    return descriptor


def _symbol_change(
    target_symbol: dict[str, Any], current_symbol: dict[str, Any], name: str
) -> dict[str, Any] | None:
    target = _semantic_symbol_descriptor(target_symbol, include_members=False)
    current = _semantic_symbol_descriptor(current_symbol, include_members=False)
    target_members = target_symbol.get("members", {})
    current_members = current_symbol.get("members", {})
    missing_members = sorted(set(target_members) - set(current_members))
    changed_members = {
        member: {
            "target": _semantic_symbol_descriptor(target_members[member]),
            "current": _semantic_symbol_descriptor(current_members[member]),
        }
        for member in sorted(set(target_members) & set(current_members))
        if _semantic_symbol_descriptor(target_members[member])
        != _semantic_symbol_descriptor(current_members[member])
    }
    if target == current and not missing_members and not changed_members:
        return None
    return {
        "symbol": name,
        "target": target,
        "current": current,
        "missing_members": missing_members,
        "changed_members": changed_members,
    }


def _coverage(
    target_modules: dict[str, dict[str, Any]],
    current_modules: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    coverage: dict[str, dict[str, Any]] = {}
    for module, target in target_modules.items():
        current = current_modules.get(module)
        if current is None:
            coverage[module] = {"status": "missing_module"}
            continue
        target_exports = target["exports"]
        current_exports = current["exports"]
        missing = sorted(set(target_exports) - set(current_exports))
        changed: list[dict[str, Any]] = []
        for name in sorted(set(target_exports) & set(current_exports)):
            change = _symbol_change(target_exports[name], current_exports[name], name)
            if change is not None:
                changed.append(change)
        status = "compatible_static_surface"
        if missing or changed or target.get("explicit_all_unresolved"):
            status = "static_drift"
        coverage[module] = {
            "status": status,
            "missing_exports": missing,
            "symbol_changes": changed,
            "target_exports_unresolved": target.get("explicit_all_unresolved", False),
        }
    return coverage


def build_manifest(repo: Path, target_ref: str) -> dict[str, Any]:
    repo = repo.resolve()
    roots = SOURCE_ROOTS + ("setup.py",)
    target = _git_snapshot(repo, target_ref, roots)
    current = _working_tree_snapshot(repo, SOURCE_ROOTS)
    target_modules = _modules(target, MODULE_ROOTS)
    current_modules = _modules(current, MODULE_ROOTS)
    target_cli = _target_cli(target)
    current_cli = _current_cli(repo)
    target_distribution = _target_distribution(target)
    current_distribution = _current_distribution(repo)
    target_data = _source_data_files(target)
    current_data = _source_data_files(current)
    return {
        "schema_version": SCHEMA_VERSION,
        "generator": "scripts/generate_libero_api_manifest.py",
        "generator_contract": {"python": ">=3.11,<3.13", "ast_renderer": "stdlib"},
        "scope": {
            "module_roots": list(MODULE_ROOTS),
            "source_roots": list(SOURCE_ROOTS),
            "python_policy": (
                "all Python files under module_roots; top-level templates are "
                "source data, not import modules"
            ),
            "source_data_policy": (
                "all non-Python files under source_roots plus every file in a "
                "templates directory; content-addressed by Git blob OID"
            ),
            "export_policy": (
                "static top-level exports with recursive relative star-import "
                "resolution"
            ),
        },
        "compatibility_target": {
            **target.metadata,
            "remote_url": UPSTREAM_URL,
        },
        "target": {
            "modules": target_modules,
            "console_scripts": target_cli,
            "distribution": target_distribution,
            "source_data_files": target_data,
        },
        "current": {
            **current.metadata,
            "modules": current_modules,
            "console_scripts": current_cli,
            "distribution": current_distribution,
            "source_data_files": current_data,
        },
        "comparison": {
            "module_coverage": _coverage(target_modules, current_modules),
            "extra_modules": sorted(set(current_modules) - set(target_modules)),
            "missing_console_scripts": sorted(set(target_cli) - set(current_cli)),
            "extra_console_scripts": sorted(set(current_cli) - set(target_cli)),
            "changed_console_scripts": {
                name: {"target": target_cli[name], "current": current_cli[name]}
                for name in sorted(set(target_cli) & set(current_cli))
                if target_cli[name] != current_cli[name]
            },
            "missing_source_data_files": sorted(set(target_data) - set(current_data)),
            "extra_source_data_files": sorted(set(current_data) - set(target_data)),
            "changed_source_data_files": {
                path: {"target": target_data[path], "current": current_data[path]}
                for path in sorted(set(target_data) & set(current_data))
                if target_data[path]["git_blob_oid"]
                != current_data[path]["git_blob_oid"]
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--target-ref", default=DEFAULT_TARGET_COMMIT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = build_manifest(args.repo, args.target_ref)
    encoded = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(encoded, end="")
        return
    output = args.output if args.output.is_absolute() else args.repo / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(encoded)


if __name__ == "__main__":
    main()
