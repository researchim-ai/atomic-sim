"""Read-only check of staged paths and blob sizes before committing.

No hooks are installed and no index/worktree changes are performed.
"""

import argparse
import json
import os
import subprocess
from pathlib import Path


def git(repo, *args, input_bytes=None, allowed=(0,)):
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode not in allowed:
        raise RuntimeError(os.fsdecode(result.stderr).strip())
    return result.stdout


def check(repo, max_bytes):
    changed = git(
        repo, "diff", "--cached", "--name-only", "--no-renames", "--diff-filter=ACMT", "-z"
    )
    paths = [p for p in changed.split(b"\0") if p]
    entries = {}
    for entry in git(repo, "ls-files", "--stage", "-z").split(b"\0"):
        if entry:
            metadata, path = entry.split(b"\t", 1)
            mode, oid, stage = metadata.split()
            if stage != b"0":
                raise RuntimeError("Сначала разрешите конфликты в индексе")
            entries[path] = (mode, oid)
    if not paths:
        return [], 0
    ignored = set(
        git(
            repo,
            "check-ignore",
            "--no-index",
            "--stdin",
            "-z",
            input_bytes=b"\0".join(paths) + b"\0",
            allowed=(0, 1),
        ).split(b"\0")
    )
    oids = sorted({entries[p][1] for p in paths if entries[p][0] != b"160000"})
    sizes = {}
    if oids:
        records = git(
            repo,
            "cat-file",
            "--batch-check=%(objectname) %(objecttype) %(objectsize)",
            input_bytes=b"\n".join(oids) + b"\n",
        )
        for record in records.splitlines():
            oid, kind, size = record.split()
            if kind != b"blob":
                raise RuntimeError("В индексе ожидался файловый blob")
            sizes[oid] = int(size)
    issues = []
    for path in paths:
        label = json.dumps(os.fsdecode(path), ensure_ascii=False)
        if path in ignored:
            issues.append(f"{label}: исключён правилами ignore; не добавляйте через -f")
        mode, oid = entries[path]
        if mode != b"160000" and sizes[oid] > max_bytes:
            issues.append(f"{label}: {sizes[oid]} байт, больше лимита {max_bytes}")
    return issues, len(paths)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--max-size-mib", type=int, default=2)
    args = parser.parse_args(argv)
    if args.max_size_mib <= 0:
        parser.error("--max-size-mib должен быть положительным")
    try:
        issues, count = check(args.repo, args.max_size_mib * 1024 * 1024)
    except (RuntimeError, OSError) as exc:
        parser.exit(2, str(exc) + "\n")
    if issues:
        print("Проверка индекса не пройдена:")
        for issue in issues:
            print("- " + issue)
        return 1
    print(f"Проверено файлов в индексе: {count}. Исключённых и слишком больших файлов нет.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
