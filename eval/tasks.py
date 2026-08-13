"""评测任务集 — 4 个"日常重复型"任务，全部在临时 sandbox 内执行。

每个任务定义:
    TASK_PROMPT: 给 agent 的任务描述
    init_sandbox(sandbox_dir): 初始化任务环境（预埋问题）
    verify(sandbox_dir) -> (passed: bool, detail: str): 验证任务是否完成
"""

from __future__ import annotations

import random
import subprocess
import sys
from pathlib import Path

TASKS = ("T1", "T2", "T3", "T4")

# 评测环境解释器（agent conda env），verify 与 agent 的 bash 工具保持一致
PY = sys.executable


# ── T1: 修复 Python 导入错误 ─────────────────────────


def init_t1(sb: Path) -> None:
    (sb / "app").mkdir()
    (sb / "app" / "__init__.py").write_text("", encoding="utf-8")
    (sb / "app" / "main.py").write_text(
        "from app.utils import helper\n\nprint(helper())\n", encoding="utf-8"
    )
    # 预埋问题：utils.py 导入了不存在的模块
    (sb / "app" / "utils.py").write_text(
        "import nonexistent_lib_xyz\n\n"
        "def helper():\n"
        "    return 'helper works'\n",
        encoding="utf-8",
    )


T1_PROMPT = (
    "运行 `python app/main.py` 时出现 ImportError。请定位并修复错误，"
    "使得 `python app/main.py` 能成功输出 'helper works'。"
)


def verify_t1(sb: Path) -> tuple[bool, str]:
    r = subprocess.run(
        [PY, "app/main.py"], cwd=str(sb), capture_output=True,
        text=True, timeout=60, encoding="utf-8", errors="replace",
    )
    if r.returncode == 0 and "helper works" in (r.stdout or ""):
        return True, f"stdout={r.stdout.strip()!r}"
    return False, f"exit={r.returncode} stdout={r.stdout.strip()!r} stderr={r.stderr.strip()[-300:]!r}"


# ── T2: 创建虚拟环境并验证运行 ───────────────────────


def init_t2(sb: Path) -> None:
    (sb / "demo.py").write_text(
        "import platform, sys\n"
        "print('python:', platform.python_version())\n"
        "print('demo OK')\n",
        encoding="utf-8",
    )
    (sb / "test_demo.py").write_text(
        "import demo\n",
        encoding="utf-8",
    )


T2_PROMPT = (
    "目录中有 demo.py（纯标准库脚本，无任何第三方依赖）。"
    "请为它创建一个隔离的虚拟环境：在目录中执行 `python -m venv .venv`，"
    "然后用 `.venv\\Scripts\\python.exe demo.py` 运行脚本，"
    "确认输出包含 'demo OK'。"
    "（这是 Windows 环境，venv 的解释器在 .venv\\Scripts\\python.exe）"
)


def verify_t2(sb: Path) -> tuple[bool, str]:
    venv_py = sb / ".venv" / "Scripts" / "python.exe"
    if not venv_py.is_file():
        return False, ".venv/Scripts/python.exe 不存在（未创建 venv 或路径不对）"
    r = subprocess.run(
        [str(venv_py), "demo.py"], cwd=str(sb), capture_output=True,
        text=True, timeout=60, encoding="utf-8", errors="replace",
    )
    if r.returncode == 0 and "demo OK" in (r.stdout or ""):
        return True, f"venv python run: {r.stdout.strip()!r}"
    return False, f"exit={r.returncode} stdout={r.stdout.strip()!r} stderr={r.stderr.strip()[-300:]!r}"


# ── T3: 批量重命名文件 ────────────────────────────────


def init_t3(sb: Path) -> None:
    rng = random.Random(42)
    for i in range(1, 9):
        (sb / f"IMG_{i:03d}.jpg").write_text(f"fake jpeg #{i}\n", encoding="utf-8")
    (sb / "notes.txt").write_text("do not touch this file\n", encoding="utf-8")
    (sb / "archive").mkdir()
    (sb / "archive" / "readme.md").write_text("archive", encoding="utf-8")


T3_PROMPT = (
    "目录中有 8 个 IMG_*.jpg 文件。请编写并执行一个 Python 脚本，"
    "将它们批量重命名为 photo_1.jpg 到 photo_8.jpg（按原文件名中的数字顺序）。"
    "注意：不得修改 notes.txt 和 archive/ 目录中的任何内容。"
    "完成后运行 `dir /b` 确认结果。"
)


def verify_t3(sb: Path) -> tuple[bool, str]:
    jpgs = sorted(sb.glob("*.jpg"))
    photos = sorted(sb.glob("photo_*.jpg"))
    notes_ok = (sb / "notes.txt").read_text(encoding="utf-8").startswith("do not touch")
    archive_ok = (sb / "archive" / "readme.md").exists()
    if len(jpgs) == 0 and len(photos) == 8 and notes_ok and archive_ok:
        return True, f"8 photos renamed, notes/archive intact"
    return False, f"jpg={len(jpgs)} photo_*.jpg={len(photos)} notes_ok={notes_ok} archive_ok={archive_ok}"


# ── T4: 运行测试并修复失败 ────────────────────────────


def init_t4(sb: Path) -> None:
    (sb / "math_utils.py").write_text(
        "def add(a, b):\n    return a + b\n\n"
        "def divide(a, b):\n    return a / b\n",
        encoding="utf-8",
    )
    # 预埋问题：divide(1, 0) 抛 ZeroDivisionError，但测试期望 ValueError → 初始失败
    (sb / "test_math.py").write_text(
        "import pytest\n"
        "from math_utils import add, divide\n\n"
        "def test_add():\n    assert add(2, 3) == 5\n\n"
        "def test_divide_by_zero():\n"
        "    with pytest.raises(ValueError):\n"
        "        divide(1, 0)\n",
        encoding="utf-8",
    )


T4_PROMPT = (
    "目录中有 math_utils.py 和 test_math.py。运行 `python -m pytest test_math.py -q`，"
    "测试 test_divide_by_zero 失败：divide(1, 0) 抛出的是 ZeroDivisionError 而测试期望 ValueError。"
    "请阅读失败原因，修改 math_utils.py 使 divide(1, 0) 抛出 ValueError('cannot divide by zero')，"
    "并同步修改测试。最终 `python -m pytest test_math.py -q` 必须全部通过。"
)


def verify_t4(sb: Path) -> tuple[bool, str]:
    r = subprocess.run(
        [PY, "-m", "pytest", "test_math.py", "-q"], cwd=str(sb),
        capture_output=True, text=True, timeout=60,
        encoding="utf-8", errors="replace",
    )
    out = (r.stdout or "") + (r.stderr or "")
    if r.returncode == 0 and "passed" in out:
        return True, out.strip().splitlines()[-1]
    return False, f"exit={r.returncode} {out.strip()[-300:]!r}"


# ── 注册表 ────────────────────────────────────────────


TASKS_REGISTRY = {
    "T1": {"prompt": T1_PROMPT, "init": init_t1, "verify": verify_t1},
    "T2": {"prompt": T2_PROMPT, "init": init_t2, "verify": verify_t2},
    "T3": {"prompt": T3_PROMPT, "init": init_t3, "verify": verify_t3},
    "T4": {"prompt": T4_PROMPT, "init": init_t4, "verify": verify_t4},
}
