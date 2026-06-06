from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Protocol


class WorkQueueError(RuntimeError):
    pass


@dataclass(frozen=True)
class QueueMessage:
    queue_name: str
    message_id: str
    job_id: str
    work_id: str
    payload: dict[str, Any]


class WorkQueue(Protocol):
    def enqueue(self, queue_name: str, payload: dict[str, Any]) -> str:
        ...

    def read(self, queue_names: list[str], *, worker_id: str, block_ms: int = 0, count: int = 1) -> list[QueueMessage]:
        ...

    def ack(self, queue_name: str, message_id: str) -> None:
        ...

    def dead_letter(self, payload: dict[str, Any]) -> str:
        ...

    def health(self) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class RedisQueueConfig:
    redis_url: str
    namespace: str = "neurokernel"
    group: str = "neurokernel-workers"
    socket_timeout: float | None = None


class RedisWorkQueue:
    def __init__(self, config: RedisQueueConfig):
        self.config = config
        try:
            import redis
        except ImportError as exc:
            raise WorkQueueError("redis package is not installed") from exc
        self._redis = redis.Redis.from_url(config.redis_url, decode_responses=True, socket_timeout=config.socket_timeout)
        self._known_groups: set[str] = set()

    def enqueue(self, queue_name: str, payload: dict[str, Any]) -> str:
        key = self._queue_key(queue_name)
        self._ensure_group(key)
        fields = {
            "job_id": str(payload.get("job_id") or ""),
            "work_id": str(payload.get("work_id") or ""),
            "payload_json": json.dumps(payload, ensure_ascii=False, sort_keys=True),
            "enqueued_at_ms": str(int(time.time() * 1000)),
        }
        return str(self._redis.xadd(key, fields))

    def read(self, queue_names: list[str], *, worker_id: str, block_ms: int = 0, count: int = 1) -> list[QueueMessage]:
        keys = [self._queue_key(queue_name) for queue_name in queue_names]
        key_to_name = {self._queue_key(queue_name): queue_name for queue_name in queue_names}
        for key in keys:
            self._ensure_group(key)
        streams = {key: ">" for key in keys}
        try:
            raw = self._redis.xreadgroup(self.config.group, worker_id, streams, count=max(1, count), block=max(0, block_ms))
        except Exception as exc:
            if "timeout" in str(exc).lower():
                return []
            raise
        messages: list[QueueMessage] = []
        for key, entries in raw or []:
            queue_name = key_to_name.get(str(key), str(key))
            for message_id, fields in entries:
                payload = _loads_payload(fields)
                messages.append(
                    QueueMessage(
                        queue_name=queue_name,
                        message_id=str(message_id),
                        job_id=str(payload.get("job_id") or fields.get("job_id") or ""),
                        work_id=str(payload.get("work_id") or fields.get("work_id") or ""),
                        payload=payload,
                    )
                )
        return messages

    def ack(self, queue_name: str, message_id: str) -> None:
        self._redis.xack(self._queue_key(queue_name), self.config.group, message_id)

    def dead_letter(self, payload: dict[str, Any]) -> str:
        key = f"{self.config.namespace}:queue:dead_letter"
        fields = {
            "job_id": str(payload.get("job_id") or ""),
            "work_id": str(payload.get("work_id") or ""),
            "payload_json": json.dumps(payload, ensure_ascii=False, sort_keys=True),
            "created_at_ms": str(int(time.time() * 1000)),
        }
        return str(self._redis.xadd(key, fields))

    def health(self) -> dict[str, Any]:
        pong = self._redis.ping()
        return {"available": bool(pong), "backend": "redis_streams", "namespace": self.config.namespace, "group": self.config.group}

    def _queue_key(self, queue_name: str) -> str:
        safe = "".join(char if char.isalnum() or char in {"_", "-", ":"} else "_" for char in queue_name).strip("_")
        if not safe:
            safe = "default"
        return f"{self.config.namespace}:queue:{safe}"

    def _ensure_group(self, key: str) -> None:
        if key in self._known_groups:
            return
        try:
            self._redis.xgroup_create(key, self.config.group, id="0", mkstream=True)
        except Exception as exc:
            text = str(exc).lower()
            if "busygroup" not in text:
                raise
        self._known_groups.add(key)


class InMemoryWorkQueue:
    def __init__(self):
        self.messages: list[tuple[str, str, dict[str, Any]]] = []
        self.acked: list[tuple[str, str]] = []
        self.dead_letters: list[dict[str, Any]] = []
        self._counter = 0

    def enqueue(self, queue_name: str, payload: dict[str, Any]) -> str:
        self._counter += 1
        message_id = f"{self._counter}-0"
        self.messages.append((queue_name, message_id, dict(payload)))
        return message_id

    def read(self, queue_names: list[str], *, worker_id: str, block_ms: int = 0, count: int = 1) -> list[QueueMessage]:
        result: list[QueueMessage] = []
        kept: list[tuple[str, str, dict[str, Any]]] = []
        for queue_name, message_id, payload in self.messages:
            if queue_name in queue_names and len(result) < count:
                result.append(QueueMessage(queue_name=queue_name, message_id=message_id, job_id=str(payload.get("job_id") or ""), work_id=str(payload.get("work_id") or ""), payload=dict(payload)))
            else:
                kept.append((queue_name, message_id, payload))
        self.messages = kept
        return result

    def ack(self, queue_name: str, message_id: str) -> None:
        self.acked.append((queue_name, message_id))

    def dead_letter(self, payload: dict[str, Any]) -> str:
        self.dead_letters.append(dict(payload))
        return f"dead-{len(self.dead_letters)}"

    def health(self) -> dict[str, Any]:
        return {"available": True, "backend": "memory"}


def build_work_queue_from_env() -> WorkQueue | None:
    redis_url = os.environ.get("NEUROKERNEL_REDIS_URL")
    enabled = os.environ.get("NEUROKERNEL_QUEUE_ENABLED", "").lower() in {"1", "true", "yes", "y"}
    if not redis_url and not enabled:
        return None
    if not redis_url:
        raise WorkQueueError("NEUROKERNEL_REDIS_URL is required when work queue is enabled")
    namespace = os.environ.get("NEUROKERNEL_QUEUE_NAMESPACE", "neurokernel")
    group = os.environ.get("NEUROKERNEL_QUEUE_GROUP", "neurokernel-workers")
    raw_timeout = os.environ.get("NEUROKERNEL_REDIS_SOCKET_TIMEOUT")
    socket_timeout = float(raw_timeout) if raw_timeout else None
    return RedisWorkQueue(RedisQueueConfig(redis_url=redis_url, namespace=namespace, group=group, socket_timeout=socket_timeout))


def queue_name_for_work_type(work_type: str) -> str:
    mapping = {
        "self_patch": "self_patch",
        "external_work": "external_work",
        "runtime_task": "runtime_task",
    }
    return mapping.get(str(work_type or "").strip(), "external_work")


def _loads_payload(fields: dict[str, Any]) -> dict[str, Any]:
    raw = fields.get("payload_json") if isinstance(fields, dict) else None
    if isinstance(raw, str):
        try:
            payload = json.loads(raw)
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass
    return dict(fields)
