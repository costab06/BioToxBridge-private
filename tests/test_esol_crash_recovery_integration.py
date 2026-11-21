"""Real Redis / subprocess ESOL crash-recovery regression test.

Opt in:
    BIOTOXBRIDGE_RUN_CRASH_TEST=1 \
      python -m pytest -v -s tests/test_esol_crash_recovery_integration.py

Requires OAF on PYTHONPATH, Redis on port 6381, and access to
the ESOL dataset and any services required by the ESOL stages.
"""

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest
import redis


ROOT = Path(__file__).resolve().parents[1]
REDIS_URL = os.environ.get(
    "BIOTOXBRIDGE_TEST_REDIS_URL",
    "redis://localhost:6381/0",
)
WORKER_GROUP = "oaf.agent.biotoxbridge-esol"
COORDINATOR_GROUP = "oaf.agent.biotoxbridge-esol-lifecycle"
TIMEOUT = 300
MIN_IDLE_MS = 30000


def wait_for(predicate, description, timeout=TIMEOUT):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(0.5)
    raise AssertionError(f"Timed out waiting for {description}")


def start_process(script, args, log_path, env):
    log = open(log_path, "w", encoding="utf-8")
    try:
        process = subprocess.Popen(
            [sys.executable, "-u", str(ROOT / "scripts" / script), *args],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        log.close()
    return process


def stop_process(process):
    if process.poll() is not None:
        return

    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return

    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def output_files(run_dir, stage):
    return sorted(
        p for p in run_dir.glob(f"{stage}-*.json")
        if not p.name.endswith("-input-binding.json")
    )


def group_info(client, stream, group):
    if not client.exists(stream):
        return None
    groups = client.xinfo_groups(stream)
    return next(
        (g for g in groups if g["name"] == group),
        None,
    )


@pytest.mark.integration
def test_esol_recovers_after_sigkill_before_commit(tmp_path):
    if os.environ.get("BIOTOXBRIDGE_RUN_CRASH_TEST") != "1":
        pytest.skip("Set BIOTOXBRIDGE_RUN_CRASH_TEST=1 to enable")

    client = redis.Redis.from_url(
        REDIS_URL,
        decode_responses=True,
        socket_timeout=5,
    )
    client.ping()

    unique = uuid.uuid4().hex
    run_id = f"esol-crash-pytest-{unique}"
    stream = f"biotoxbridge.esol.pytest.{unique}"
    artifact_root = tmp_path / "artifacts"
    run_dir = artifact_root / run_id
    store = tmp_path / "store"
    marker = tmp_path / "crash.injected"

    env = os.environ.copy()
    oaf_path = (
        Path.home()
        / "Projects/PyCharmProjects/OpenAgentFabric/framework"
    )
    python_paths = [str(oaf_path), str(ROOT / "src")]
    if env.get("PYTHONPATH"):
        python_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_paths)

    common = [
        "--redis-url", REDIS_URL,
        "--stream", stream,
        "--artifact-root", str(artifact_root),
        "--min-idle-ms", str(MIN_IDLE_MS),
        "--heartbeat-interval", "10",
        "--recovery-interval", "2",
    ]

    coordinator_log = tmp_path / "coordinator.log"
    crashed_worker_log = tmp_path / "crashed-worker.log"
    recovered_worker_log = tmp_path / "recovered-worker.log"

    # Initialize the isolated stream before consumers start.
    # This avoids an XINFO GROUPS race on a nonexistent Redis key.
    initial_id = client.xadd(stream, {"test_init": "1"})
    client.xdel(stream, initial_id)

    processes = []

    def launch(script, args, log_path):
        process = start_process(script, args, log_path, env)
        processes.append(process)
        return process

    def diagnostics():
        for path in (
            coordinator_log,
            crashed_worker_log,
            recovered_worker_log,
        ):
            if path.exists():
                print(f"\n=== {path.name} ===")
                print(path.read_text(errors="replace")[-12000:])

    try:
        coordinator = launch(
            "esol_oaf_lifecycle_coordinator.py",
            common,
            coordinator_log,
        )

        crash_worker = launch(
            "esol_oaf_worker.py",
            [
                *common,
                "--store", str(store),
                "--crash-after-artifact-run-id", run_id,
                "--crash-stage", "esol_acquire",
                "--crash-marker", str(marker),
            ],
            crashed_worker_log,
        )

        wait_for(
            lambda: (
                group_info(client, stream, WORKER_GROUP)
                and group_info(client, stream, COORDINATOR_GROUP)
            ),
            "both Redis consumer groups",
            timeout=30,
        )

        assert coordinator.poll() is None
        assert crash_worker.poll() is None

        submitted = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/esol_oaf_lifecycle_submit.py"),
                "--redis-url", REDIS_URL,
                "--stream", stream,
                "--artifact-root", str(artifact_root),
                "--run-id", run_id,
            ],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )

        submission = json.loads(submitted.stdout)
        assert submission["created"] is True
        assert submission["run_id"] == run_id

        wait_for(
            lambda: crash_worker.poll() is not None,
            "deliberate worker termination",
        )

        assert crash_worker.returncode == -signal.SIGKILL
        assert marker.read_text().strip() == "injected"
        assert (
            "INJECTING SIGKILL AFTER ARTIFACT, BEFORE OAF COMMIT"
            in crashed_worker_log.read_text()
        )

        acquisition = output_files(run_dir, "esol_acquire")
        assert len(acquisition) == 1
        original_digest = digest(acquisition[0])

        pending = client.xpending(stream, WORKER_GROUP)
        assert pending["pending"] == 1

        # Verify the interrupted event has not yet been acknowledged.
        pending_detail = client.xpending_range(
            stream, WORKER_GROUP, "-", "+", 10
        )
        assert len(pending_detail) == 1
        assert (
            pending_detail[0]["message_id"]
            == submission["message_id"]
        )

        # Restart without crash injection.
        recovered_worker = launch(
            "esol_oaf_worker.py",
            [
                *common,
                "--store", str(store),
            ],
            recovered_worker_log,
        )

        def completed():
            if coordinator.poll() is not None:
                raise AssertionError("Coordinator exited unexpectedly")
            if recovered_worker.poll() is not None:
                raise AssertionError("Recovered worker exited unexpectedly")

            if len(output_files(run_dir, "esol_train")) != 1:
                return False
            if len(output_files(run_dir, "esol_promote")) != 1:
                return False

            for group in (WORKER_GROUP, COORDINATOR_GROUP):
                info = group_info(client, stream, group)
                if not info:
                    return False
                if info["pending"] != 0 or info["lag"] != 0:
                    return False

            return True

        wait_for(completed, "recovered ESOL lifecycle completion")

        acquisition_after = output_files(run_dir, "esol_acquire")
        assert len(acquisition_after) == 1
        assert digest(acquisition_after[0]) == original_digest

        promotion = json.loads(
            output_files(run_dir, "esol_promote")[0].read_text()
        )
        assert isinstance(promotion["accepted"], bool)
        assert "candidate_version" in promotion

        # Locate the deterministic acquisition event identity.
        acquisition_id = uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"biotoxbridge:lifecycle:{run_id}:esol_acquire",
        )
        event_id = str(acquisition_id)

        terminal = (
            f"oaf:terminal:{stream}:{WORKER_GROUP}:{event_id}"
        )
        terminal_message = (
            f"oaf:terminal-message:{stream}:{WORKER_GROUP}:{event_id}"
        )

        assert client.exists(terminal) == 1
        assert client.exists(terminal_message) == 1

        step_keys = list(
            client.scan_iter(
                match=f"oaf:workflow:step:*:biotoxbridge-esol:{event_id}"
            )
        )
        assert len(step_keys) == 1

        assert client.xpending(stream, WORKER_GROUP)["pending"] == 0
        assert client.xpending(stream, COORDINATOR_GROUP)["pending"] == 0

        print(
            "\nPASS: SIGKILL recovery, immutable acquisition artifact, "
            "durable OAF records, completed lifecycle, zero pending."
        )

    except Exception:
        diagnostics()
        raise

    finally:
        for process in reversed(processes):
            stop_process(process)

        # Remove only this test's isolated stream and consumer-group
        # metadata. Do not touch any shared or historical streams.
        client.delete(stream)
        client.close()
