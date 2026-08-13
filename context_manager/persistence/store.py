"""Workflow + Step 元数据存储抽象基类与实现。"""

from __future__ import annotations

import datetime
import sqlite3
from abc import ABC, abstractmethod

from ..models import Workflow, Step


class WorkflowStoreBase(ABC):
    """Workflow 元数据存储接口。"""

    @abstractmethod
    def create_workflow(self, workflow_id: str, name: str,
                        description: str = "", source_thread_id: str = "",
                        tags: str = "") -> None:
        ...

    @abstractmethod
    def get_workflow(self, workflow_id: str) -> Workflow | None:
        ...

    @abstractmethod
    def list_workflows(self, status: str | None = None) -> list[Workflow]:
        ...

    @abstractmethod
    def update_status(self, workflow_id: str, status: str) -> None:
        ...

    @abstractmethod
    def update_name(self, workflow_id: str, name: str) -> None:
        ...

    @abstractmethod
    def update_description(self, workflow_id: str, description: str) -> None:
        ...

    @abstractmethod
    def delete_workflow(self, workflow_id: str) -> None:
        ...

    @abstractmethod
    def add_step(self, step_id: str, workflow_id: str, step_index: int,
                 type: str, name: str, arguments: str = "", result: str = "",
                 status: str = "success", duration_ms: int = 0,
                 error_message: str = "", timestamp: str = "") -> None:
        ...

    @abstractmethod
    def get_steps(self, workflow_id: str) -> list[Step]:
        ...

    @abstractmethod
    def update_step_pruned(self, step_id: str, is_pruned: bool) -> None:
        ...

    @abstractmethod
    def update_step_fields(self, step_id: str, **fields) -> None:
        ...

    @abstractmethod
    def delete_step(self, step_id: str) -> None:
        ...

    @abstractmethod
    def reorder_steps(self, workflow_id: str, step_index_map: dict[str, int]) -> None:
        ...

    @abstractmethod
    def close(self) -> None:
        ...


class SQLiteWorkflowStore(WorkflowStoreBase):
    """基于 SQLite 的 Workflow 元数据持久化存储。"""

    def __init__(self, db_path: str = "context_manager.db"):
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS workflows (
                workflow_id     TEXT PRIMARY KEY,
                name            TEXT NOT NULL,
                description     TEXT DEFAULT '',
                status          TEXT DEFAULT 'RAW',
                source_thread_id TEXT DEFAULT '',
                tags            TEXT DEFAULT '',
                created_at      TEXT NOT NULL,
                updated_at      TEXT NOT NULL
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS steps (
                step_id         TEXT PRIMARY KEY,
                workflow_id     TEXT NOT NULL,
                step_index      INTEGER NOT NULL,
                type            TEXT NOT NULL,
                name            TEXT NOT NULL,
                arguments       TEXT DEFAULT '',
                result          TEXT DEFAULT '',
                status          TEXT DEFAULT 'success',
                duration_ms     INTEGER DEFAULT 0,
                error_message   TEXT DEFAULT '',
                is_pruned       INTEGER DEFAULT 0,
                timestamp       TEXT NOT NULL,
                FOREIGN KEY (workflow_id) REFERENCES workflows(workflow_id)
            )
        """)
        self._conn.commit()

    # ── Workflow CRUD ────────────────────────────────

    def create_workflow(self, workflow_id: str, name: str,
                        description: str = "", source_thread_id: str = "",
                        tags: str = "") -> None:
        now = datetime.datetime.now().isoformat()
        self._conn.execute(
            "INSERT INTO workflows (workflow_id, name, description, status, "
            "source_thread_id, tags, created_at, updated_at) "
            "VALUES (?, ?, ?, 'RAW', ?, ?, ?, ?)",
            (workflow_id, name, description, source_thread_id, tags, now, now),
        )
        self._conn.commit()

    def get_workflow(self, workflow_id: str) -> Workflow | None:
        row = self._conn.execute(
            "SELECT workflow_id, name, description, status, source_thread_id, "
            "tags, created_at, updated_at FROM workflows WHERE workflow_id = ?",
            (workflow_id,),
        ).fetchone()
        if row is None:
            return None
        wf = self._row_to_workflow(row)
        wf.steps = self.get_steps(workflow_id)
        return wf

    def list_workflows(self, status: str | None = None) -> list[Workflow]:
        if status:
            rows = self._conn.execute(
                "SELECT workflow_id, name, description, status, source_thread_id, "
                "tags, created_at, updated_at FROM workflows WHERE status = ? ORDER BY created_at",
                (status,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT workflow_id, name, description, status, source_thread_id, "
                "tags, created_at, updated_at FROM workflows ORDER BY created_at"
            ).fetchall()
        return [self._row_to_workflow(r) for r in rows]

    def update_status(self, workflow_id: str, status: str) -> None:
        now = datetime.datetime.now().isoformat()
        self._conn.execute(
            "UPDATE workflows SET status = ?, updated_at = ? WHERE workflow_id = ?",
            (status, now, workflow_id),
        )
        self._conn.commit()

    def update_name(self, workflow_id: str, name: str) -> None:
        now = datetime.datetime.now().isoformat()
        self._conn.execute(
            "UPDATE workflows SET name = ?, updated_at = ? WHERE workflow_id = ?",
            (name, now, workflow_id),
        )
        self._conn.commit()

    def update_description(self, workflow_id: str, description: str) -> None:
        now = datetime.datetime.now().isoformat()
        self._conn.execute(
            "UPDATE workflows SET description = ?, updated_at = ? WHERE workflow_id = ?",
            (description, now, workflow_id),
        )
        self._conn.commit()

    def delete_workflow(self, workflow_id: str) -> None:
        self._conn.execute("DELETE FROM steps WHERE workflow_id = ?", (workflow_id,))
        self._conn.execute("DELETE FROM workflows WHERE workflow_id = ?", (workflow_id,))
        self._conn.commit()

    # ── Step CRUD ────────────────────────────────────

    def add_step(self, step_id: str, workflow_id: str, step_index: int,
                 type: str, name: str, arguments: str = "", result: str = "",
                 status: str = "success", duration_ms: int = 0,
                 error_message: str = "", timestamp: str = "") -> None:
        self._conn.execute(
            "INSERT INTO steps (step_id, workflow_id, step_index, type, name, "
            "arguments, result, status, duration_ms, error_message, is_pruned, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)",
            (step_id, workflow_id, step_index, type, name, arguments, result,
             status, duration_ms, error_message, timestamp),
        )
        self._conn.commit()

    def get_steps(self, workflow_id: str) -> list[Step]:
        rows = self._conn.execute(
            "SELECT step_id, workflow_id, step_index, type, name, arguments, result, "
            "status, duration_ms, error_message, is_pruned, timestamp "
            "FROM steps WHERE workflow_id = ? ORDER BY step_index",
            (workflow_id,),
        ).fetchall()
        return [Step.from_dict(self._row_to_step_dict(r)) for r in rows]

    def update_step_pruned(self, step_id: str, is_pruned: bool) -> None:
        self._conn.execute(
            "UPDATE steps SET is_pruned = ? WHERE step_id = ?",
            (1 if is_pruned else 0, step_id),
        )
        self._conn.commit()

    def update_step_fields(self, step_id: str, **fields) -> None:
        allowed = {"name", "arguments", "result", "status", "error_message", "duration_ms", "type", "timestamp"}
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [step_id]
        self._conn.execute(
            f"UPDATE steps SET {set_clause} WHERE step_id = ?",
            values,
        )
        self._conn.commit()

    def delete_step(self, step_id: str) -> None:
        self._conn.execute("DELETE FROM steps WHERE step_id = ?", (step_id,))
        self._conn.commit()

    def reorder_steps(self, workflow_id: str, step_index_map: dict[str, int]) -> None:
        with self._conn:
            for step_id, new_index in step_index_map.items():
                self._conn.execute(
                    "UPDATE steps SET step_index = ? WHERE step_id = ? AND workflow_id = ?",
                    (new_index, step_id, workflow_id),
                )

    def close(self) -> None:
        self._conn.close()

    # ── 内部 ────────────────────────────────────────

    @staticmethod
    def _row_to_workflow(row: tuple) -> Workflow:
        return Workflow(
            workflow_id=row[0],
            name=row[1],
            description=row[2],
            status=row[3],
            source_thread_id=row[4],
            tags=row[5],
            created_at=row[6],
            updated_at=row[7],
        )

    @staticmethod
    def _row_to_step_dict(row: tuple) -> dict:
        return {
            "step_id": row[0],
            "workflow_id": row[1],
            "step_index": row[2],
            "type": row[3],
            "name": row[4],
            "arguments": row[5],
            "result": row[6],
            "status": row[7],
            "duration_ms": row[8],
            "error_message": row[9],
            "is_pruned": bool(row[10]),
            "timestamp": row[11],
        }


class MemoryWorkflowStore(WorkflowStoreBase):
    """基于内存 dict 的 Workflow 元数据存储。"""

    def __init__(self):
        self._workflows: dict[str, Workflow] = {}

    # ── Workflow CRUD ────────────────────────────────

    def create_workflow(self, workflow_id: str, name: str,
                        description: str = "", source_thread_id: str = "",
                        tags: str = "") -> None:
        now = datetime.datetime.now().isoformat()
        self._workflows[workflow_id] = Workflow(
            workflow_id=workflow_id,
            name=name,
            description=description,
            status="RAW",
            source_thread_id=source_thread_id,
            tags=tags,
            created_at=now,
            updated_at=now,
        )

    def get_workflow(self, workflow_id: str) -> Workflow | None:
        return self._workflows.get(workflow_id)

    def list_workflows(self, status: str | None = None) -> list[Workflow]:
        result = list(self._workflows.values())
        if status:
            result = [w for w in result if w.status == status]
        return result

    def update_status(self, workflow_id: str, status: str) -> None:
        wf = self._workflows.get(workflow_id)
        if wf:
            wf.status = status
            wf.updated_at = datetime.datetime.now().isoformat()

    def update_name(self, workflow_id: str, name: str) -> None:
        wf = self._workflows.get(workflow_id)
        if wf:
            wf.name = name
            wf.updated_at = datetime.datetime.now().isoformat()

    def update_description(self, workflow_id: str, description: str) -> None:
        wf = self._workflows.get(workflow_id)
        if wf:
            wf.description = description
            wf.updated_at = datetime.datetime.now().isoformat()

    def delete_workflow(self, workflow_id: str) -> None:
        self._workflows.pop(workflow_id, None)

    # ── Step CRUD ────────────────────────────────────

    def add_step(self, step_id: str, workflow_id: str, step_index: int,
                 type: str, name: str, arguments: str = "", result: str = "",
                 status: str = "success", duration_ms: int = 0,
                 error_message: str = "", timestamp: str = "") -> None:
        wf = self._workflows.get(workflow_id)
        if wf is None:
            return
        step = Step(
            step_id=step_id,
            workflow_id=workflow_id,
            step_index=step_index,
            type=type,
            name=name,
            arguments=arguments,
            result=result,
            status=status,
            duration_ms=duration_ms,
            error_message=error_message,
            timestamp=timestamp,
        )
        wf.steps.append(step)
        wf.steps.sort(key=lambda s: s.step_index)

    def get_steps(self, workflow_id: str) -> list[Step]:
        wf = self._workflows.get(workflow_id)
        if wf is None:
            return []
        return sorted(wf.steps, key=lambda s: s.step_index)

    def update_step_pruned(self, step_id: str, is_pruned: bool) -> None:
        for wf in self._workflows.values():
            for s in wf.steps:
                if s.step_id == step_id:
                    s.is_pruned = is_pruned
                    return

    def update_step_fields(self, step_id: str, **fields) -> None:
        allowed = {"name", "arguments", "result", "status", "error_message", "duration_ms", "type", "timestamp", "is_pruned"}
        for wf in self._workflows.values():
            for s in wf.steps:
                if s.step_id == step_id:
                    for k, v in fields.items():
                        if k in allowed:
                            setattr(s, k, v)
                    return

    def delete_step(self, step_id: str) -> None:
        for wf in self._workflows.values():
            wf.steps = [s for s in wf.steps if s.step_id != step_id]

    def reorder_steps(self, workflow_id: str, step_index_map: dict[str, int]) -> None:
        wf = self._workflows.get(workflow_id)
        if wf is None:
            return
        for s in wf.steps:
            if s.step_id in step_index_map:
                s.step_index = step_index_map[s.step_id]
        wf.steps.sort(key=lambda s: s.step_index)

    def close(self) -> None:
        pass