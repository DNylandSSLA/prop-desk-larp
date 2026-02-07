"""
Walpole — Job orchestrator with scheduling and dependencies.

Inspired by the batch job schedulers described in Bank Python. Walpole manages
jobs that can be one-shot, periodic, or long-running services, with dependency
tracking and optional Barbara integration for persisting results.

Job modes:
- one-shot (run_once): execute once and record success/failure
- periodic: execute on a fixed interval
- service: long-running, restart on crash
"""

import threading
import time
import logging
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class JobMode(Enum):
    RUN_ONCE = "run_once"
    PERIODIC = "periodic"
    SERVICE = "service"


class JobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass
class JobConfig:
    """Configuration for a Walpole job."""
    name: str
    callable: object  # the function to run
    mode: JobMode = JobMode.RUN_ONCE
    interval: float = 5.0  # seconds between runs (periodic mode)
    depends_on: list = field(default_factory=list)  # list of job names
    retries: int = 0
    barbara_key: str = None  # optional key to persist results in Barbara


class JobRunner:
    """
    Orchestrator that manages and runs jobs according to their configuration.
    """

    def __init__(self, barbara=None):
        self._jobs = {}  # name -> JobConfig
        self._status = {}  # name -> JobStatus
        self._results = {}  # name -> last result
        self._threads = {}  # name -> Thread
        self._stop_event = threading.Event()
        self._barbara = barbara
        self._lock = threading.Lock()
        self._ever_succeeded = set()  # tracks jobs that have succeeded at least once

    def add_job(self, config):
        """Register a job configuration."""
        self._jobs[config.name] = config
        self._status[config.name] = JobStatus.PENDING

    def _deps_satisfied(self, job_name):
        """Check if all dependencies for a job have succeeded (at least once)."""
        config = self._jobs[job_name]
        for dep in config.depends_on:
            status = self._status.get(dep)
            if status != JobStatus.SUCCEEDED and dep not in self._ever_succeeded:
                return False
        return True

    def _run_job(self, config):
        """Execute a single job, handling retries."""
        attempts = 0
        max_attempts = config.retries + 1

        while attempts < max_attempts:
            if self._stop_event.is_set():
                return
            try:
                with self._lock:
                    self._status[config.name] = JobStatus.RUNNING
                logger.info(f"[Walpole] Running job: {config.name} (attempt {attempts + 1})")

                result = config.callable()

                with self._lock:
                    self._status[config.name] = JobStatus.SUCCEEDED
                    self._results[config.name] = result
                    self._ever_succeeded.add(config.name)

                if config.barbara_key and self._barbara:
                    self._barbara[config.barbara_key] = result

                logger.info(f"[Walpole] Job {config.name} succeeded")
                return result

            except Exception as e:
                attempts += 1
                logger.warning(
                    f"[Walpole] Job {config.name} failed (attempt {attempts}/{max_attempts}): {e}"
                )
                if attempts >= max_attempts:
                    with self._lock:
                        self._status[config.name] = JobStatus.FAILED
                        self._results[config.name] = e
                    return None

    def _run_periodic(self, config):
        """Run a job repeatedly on a fixed interval."""
        while not self._stop_event.is_set():
            if self._deps_satisfied(config.name):
                self._run_job(config)
                # Reset to PENDING so it can run again after the interval
                # but only if it succeeded (don't reset failures)
                if self._status[config.name] == JobStatus.SUCCEEDED:
                    with self._lock:
                        self._status[config.name] = JobStatus.PENDING
            self._stop_event.wait(config.interval)

    def _run_service(self, config):
        """Run a job as a service — restart on crash."""
        while not self._stop_event.is_set():
            if self._deps_satisfied(config.name):
                self._run_job(config)
                if self._status[config.name] == JobStatus.FAILED:
                    logger.info(f"[Walpole] Service {config.name} crashed, restarting...")
                    with self._lock:
                        self._status[config.name] = JobStatus.PENDING
                    continue
            self._stop_event.wait(config.interval)

    def start(self):
        """Start all registered jobs in background threads."""
        self._stop_event.clear()

        for name, config in self._jobs.items():
            if config.mode == JobMode.RUN_ONCE:
                t = threading.Thread(
                    target=self._run_once_with_deps,
                    args=(config,),
                    name=f"walpole-{name}",
                    daemon=True,
                )
            elif config.mode == JobMode.PERIODIC:
                t = threading.Thread(
                    target=self._run_periodic,
                    args=(config,),
                    name=f"walpole-{name}",
                    daemon=True,
                )
            elif config.mode == JobMode.SERVICE:
                t = threading.Thread(
                    target=self._run_service,
                    args=(config,),
                    name=f"walpole-{name}",
                    daemon=True,
                )
            else:
                continue

            self._threads[name] = t
            t.start()

    def _run_once_with_deps(self, config):
        """Run a one-shot job, waiting for dependencies first."""
        while not self._stop_event.is_set():
            if self._deps_satisfied(config.name):
                self._run_job(config)
                return
            self._stop_event.wait(0.5)

    def stop(self, timeout=5.0):
        """Signal all jobs to stop and wait for threads to finish."""
        self._stop_event.set()
        for name, t in self._threads.items():
            t.join(timeout=timeout)
            if t.is_alive():
                logger.warning(f"[Walpole] Thread for {name} did not stop cleanly")
        with self._lock:
            for name in self._jobs:
                if self._status[name] == JobStatus.RUNNING:
                    self._status[name] = JobStatus.STOPPED

    def get_status(self, job_name):
        return self._status.get(job_name)

    def get_result(self, job_name):
        return self._results.get(job_name)

    def status_report(self):
        lines = ["Walpole Job Status:"]
        for name, config in self._jobs.items():
            status = self._status.get(name, JobStatus.PENDING)
            lines.append(f"  {name}: {status.value} (mode={config.mode.value})")
        return "\n".join(lines)

    def __repr__(self):
        return f"JobRunner(jobs={len(self._jobs)})"
