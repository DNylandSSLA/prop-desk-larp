"""Tests for Walpole — job orchestrator."""

import time
import threading
from bank_python.walpole import JobRunner, JobConfig, JobMode, JobStatus


class TestJobRunner:
    def test_run_once(self):
        results = []

        def job():
            results.append("done")
            return "ok"

        runner = JobRunner()
        runner.add_job(JobConfig(name="test_job", callable=job, mode=JobMode.RUN_ONCE))
        runner.start()
        time.sleep(1)
        runner.stop()

        assert results == ["done"]
        assert runner.get_status("test_job") == JobStatus.SUCCEEDED
        assert runner.get_result("test_job") == "ok"

    def test_periodic(self):
        counter = {"n": 0}

        def job():
            counter["n"] += 1
            return counter["n"]

        runner = JobRunner()
        runner.add_job(JobConfig(
            name="periodic_job", callable=job,
            mode=JobMode.PERIODIC, interval=0.3,
        ))
        runner.start()
        time.sleep(1.5)
        runner.stop()

        # Should have run multiple times
        assert counter["n"] >= 3

    def test_dependency_checking(self):
        order = []

        def job_a():
            order.append("A")
            return "A done"

        def job_b():
            order.append("B")
            return "B done"

        runner = JobRunner()
        runner.add_job(JobConfig(
            name="job_a", callable=job_a, mode=JobMode.RUN_ONCE,
        ))
        runner.add_job(JobConfig(
            name="job_b", callable=job_b, mode=JobMode.RUN_ONCE,
            depends_on=["job_a"],
        ))
        runner.start()
        time.sleep(2)
        runner.stop()

        assert "A" in order
        assert "B" in order
        # A should run before B
        assert order.index("A") < order.index("B")

    def test_retry_on_failure(self):
        attempts = {"n": 0}

        def flaky_job():
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise ValueError("not yet")
            return "finally"

        runner = JobRunner()
        runner.add_job(JobConfig(
            name="flaky", callable=flaky_job,
            mode=JobMode.RUN_ONCE, retries=3,
        ))
        runner.start()
        time.sleep(2)
        runner.stop()

        assert runner.get_status("flaky") == JobStatus.SUCCEEDED
        assert runner.get_result("flaky") == "finally"

    def test_failed_job(self):
        def bad_job():
            raise RuntimeError("boom")

        runner = JobRunner()
        runner.add_job(JobConfig(
            name="bad", callable=bad_job,
            mode=JobMode.RUN_ONCE, retries=0,
        ))
        runner.start()
        time.sleep(1)
        runner.stop()

        assert runner.get_status("bad") == JobStatus.FAILED

    def test_status_report(self):
        runner = JobRunner()
        runner.add_job(JobConfig(name="j1", callable=lambda: None))
        report = runner.status_report()
        assert "j1" in report
        assert "pending" in report

    def test_clean_shutdown(self):
        def slow_job():
            time.sleep(0.5)
            return "ok"

        runner = JobRunner()
        runner.add_job(JobConfig(
            name="slow", callable=slow_job,
            mode=JobMode.PERIODIC, interval=0.2,
        ))
        runner.start()
        time.sleep(0.3)
        runner.stop(timeout=3.0)
        # Should stop without hanging

    def test_barbara_integration(self):
        from bank_python.barbara import BarbaraDB

        db = BarbaraDB.open("default")

        def job():
            return {"result": 42}

        runner = JobRunner(barbara=db)
        runner.add_job(JobConfig(
            name="persist_job", callable=job,
            mode=JobMode.RUN_ONCE,
            barbara_key="/Jobs/test_result",
        ))
        runner.start()
        time.sleep(1)
        runner.stop()

        assert db["/Jobs/test_result"] == {"result": 42}
        db.close()
