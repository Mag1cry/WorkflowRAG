"""SQLite Workflow + Step 元数据存储。"""

import datetime
import sqlite3

from .base import WorkflowStoreBase


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

    def get_workflow(self, workflow_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT workflow_id, name, description, status, source_thread_id, "
            "tags, created_at, updated_at FROM workflows WHERE workflow_id = ?",
            (workflow_id,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_workflow(row)

    def list_workflows(self, status: str | None = None) -> list[dict]:
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

    def get_steps(self, workflow_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT step_id, workflow_id, step_index, type, name, arguments, result, "
            "status, duration_ms, error_message, is_pruned, timestamp "
            "FROM steps WHERE workflow_id = ? ORDER BY step_index",
            (workflow_id,),
        ).fetchall()
        return [self._row_to_step(r) for r in rows]

    def update_step_pruned(self, step_id: str, is_pruned: bool) -> None:
        self._conn.execute(
            "UPDATE steps SET is_pruned = ? WHERE step_id = ?",
            (1 if is_pruned else 0, step_id),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ── 内部 ────────────────────────────────────────

    @staticmethod
    def _row_to_workflow(row: tuple) -> dict:
        return {
            "workflow_id": row[0],
            "name": row[1],
            "description": row[2],
            "status": row[3],
            "source_thread_id": row[4],
            "tags": row[5],
            "created_at": row[6],
            "updated_at": row[7],
        }

    @staticmethod
    def _row_to_step(row: tuple) -> dict:
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