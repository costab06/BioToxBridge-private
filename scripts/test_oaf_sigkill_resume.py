#!/usr/bin/env python3
"""Opt-in destructive-to-worker (not Redis) OAF crash/resume integration probe."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from uuid import uuid4

STAGES = ('toxicity_target', 'chemistry_resolution', 'structure_grouping', 'tox21_scoring', 'master_assembly', 'dataset_split')

def hashes(folder):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.glob('*.json')}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='config/experiments/liver_loael_v1.yaml')
    ap.add_argument('--redis-url', default='redis://localhost:6381/0')
    ap.add_argument('--run-id', default=None)
    ap.add_argument('--artifact-root', default='runs/oaf-artifacts')
    ap.add_argument('--work-root', default='runs/oaf-work')
    ap.add_argument('--wait-seconds', type=float, default=240)
    ap.add_argument('--resume-timeout', type=float, default=900)
    ap.add_argument('--kill-delay', type=float, default=0.05)
    args = ap.parse_args()
    run_id = args.run_id or ('oaf-crash-' + uuid4().hex[:12])
    folder = Path(args.artifact_root).resolve() / run_id
    if folder.exists() or (Path(args.work_root) / run_id).exists():
        ap.error('Run ID already has state. Use a fresh run ID; this script never removes artifacts or Redis data.')
    cmd = [sys.executable, '-u', '-m', 'biotoxbridge.oaf.runtime', '--config', args.config,
           '--run-id', run_id, '--redis-url', args.redis_url, '--artifact-root', args.artifact_root,
           '--work-root', args.work_root, '--timeout', str(args.resume_timeout)]
    print('RUN_ID:', run_id, flush=True)
    print('Starting initial worker; waiting for structure_grouping output before SIGKILL.', flush=True)
    log_dir = Path('runs/oaf-crash-test-logs'); log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / f'{run_id}-initial.log').open('wb') as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic() + args.wait_seconds
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    raise RuntimeError(f'Initial worker exited prematurely: {proc.returncode}; inspect log')
                if list(folder.glob('structure_grouping-*.json')):
                    break
                time.sleep(0.05)
            else:
                raise TimeoutError('Structure grouping not completed before deadline; inspect log')
            time.sleep(args.kill_delay)
            if proc.poll() is not None:
                raise RuntimeError('Worker finished before kill; rerun with a fresh ID and smaller --kill-delay')
            if list(folder.glob('dataset_split-*.json')):
                raise RuntimeError('Workflow already finished before kill; no crash-recovery assertion possible')
            before = hashes(folder)
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=15)
            print(f'SIGKILL sent; {len(before)} immutable JSON artifacts recorded.', flush=True)
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=15)
    print('Resuming with --resume; waiting for completion.', flush=True)
    with (log_dir / f'{run_id}-resume.log').open('wb') as log:
        try:
            resumed = subprocess.run(cmd + ['--resume'], stdout=log, stderr=subprocess.STDOUT,
                                     timeout=args.resume_timeout + 30)
        except subprocess.TimeoutExpired:
            raise RuntimeError('Resume timed out; inspect Redis pending messages and resume log')
    if resumed.returncode:
        raise RuntimeError(f'Resume exited {resumed.returncode}; inspect resume log')
    after = hashes(folder)
    altered = [name for name, digest in before.items() if after.get(name) != digest]
    if altered:
        raise AssertionError(f'Existing artifacts changed or disappeared: {altered}')
    final = [
        p for p in (
            Path("runs/oaf-artifacts") / run_id
        ).glob("dataset_split-*.json")
        if not p.name.endswith("-input-binding.json")
    ]
    if len(final) != 1:
        raise AssertionError(f'Expected exactly one dataset_split output, found {len(final)}')
    payload = json.loads(final[0].read_text())
    train, test = payload.get('train'), payload.get('test')
    if not isinstance(train, list) or not isinstance(test, list):
        raise AssertionError('Final artifact missing train/test lists')
    report = {'run_id': run_id, 'pre_crash_artifacts': len(before),
              'post_resume_artifacts': len(after), 'unchanged_pre_crash_artifacts': len(before),
              'train_count': len(train), 'test_count': len(test),
              'initial_log': str(log_dir / f'{run_id}-initial.log'),
              'resume_log': str(log_dir / f'{run_id}-resume.log')}
    report_path = log_dir / f'{run_id}-report.json'
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print('PASS: SIGKILL / resume / artifact preservation / split completion')
    print(json.dumps(report, indent=2))
    print('NOTE: This does not prove kill occurred inside Tox21 or exactly-once side effects.')

if __name__ == '__main__':
    main()
