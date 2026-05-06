import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Job:
    job_id: str
    status: str  # pending | running | done | error
    pano_ids: list[str]
    output_dir: str
    ply_files: list[str] = field(default_factory=list)
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)


_jobs: dict[str, Job] = {}
_lock = threading.Lock()


def create_job(pano_ids: list[str]) -> Job:
    job_id = uuid.uuid4().hex
    job = Job(
        job_id=job_id,
        status="pending",
        pano_ids=pano_ids,
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
