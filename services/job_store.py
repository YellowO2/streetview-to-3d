import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Job:
    job_id: str
    status: str  # pending | running | done | error
    target_pano_id: str
    output_dir: str
    support_pano_ids: list[str] = field(default_factory=list)
    ply_files: list[str] = field(default_factory=list)
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)


_jobs: dict[str, Job] = {}
_lock = threading.Lock()


def create_job(target_pano_id: str) -> Job:
    job_id = uuid.uuid4().hex
    job = Job(
        job_id=job_id,
        status="pending",
        target_pano_id=target_pano_id,
        output_dir=f"splats/{job_id}",
    )
    with _lock:
        _jobs[job_id] = job
    return job


def get_job(job_id: str) -> Optional[Job]:
    return _jobs.get(job_id)


def update_job(job_id: str, **kwargs) -> None:
    with _lock:
        job = _jobs.get(job_id)
        if job:
            for k, v in kwargs.items():
                setattr(job, k, v)
