# BioToxBridge OAF runtime integration (development)

This package adds a Redis-backed development runner. It does not edit OpenAgentFabric.

## Install
From BioToxBridge root:

    unzip -o ~/Downloads/biotoxbridge_oaf_runtime_v1.zip
    .venv/bin/python -m pip install -e ../OpenAgentFabric
    export PYTHONPATH="$PWD/src:$HOME/Projects/PyCharmProjects/OpenAgentFabric/framework"
    .venv/bin/python -m pytest -q

Start Redis separately, then:

    .venv/bin/python -m biotoxbridge.oaf.runtime \\
      --config config/experiments/liver_loael_v1.yaml \\
      --run-id oaf-test-001

The real scientific pipeline requires configured source data, a reachable
chemistry resolver and Tox21 scoring service. Use a NEW run-id for a new
experiment. A timed-out run may be resumed with --resume.

## Known limitations
- This is a development runner, not a validated end-to-end deployment.
- Bootstrap SET NX and event publication are not atomic. A crash in that gap
  requires inspecting Redis and manually resolving the bootstrap marker.
- The scientific functions write side-effect files; a crash/retry can repeat
  external scoring and file writes. Confirm replay semantics before production.
- Correlation records expire after the OAF registry TTL; long-running jobs
  need a deliberate lifecycle policy.
- The runner currently starts both workers in one process and polls the OAF
  completion index. It does not provide independent worker supervision.
- No scientific equivalence test against standalone pipeline has been run.
