---
category: discovery
timestamp: '2026-01-18T15:41:01.676363+00:00'
project_id: f959454c-12b6-4062-b411-59fa58f83777
files_affected: []
---

## Discovery
Templates located in the repository root are not bundled into the wheel package, causing them to be inaccessible when installed globally via pipx.

## Reasoning
Standard Hatch/Pip packaging only includes files within the specified package directory or explicitly listed shared-data. Internal code expecting relative paths to the repo root fails in a global install.

## Correction
Moved templates into `src/nexus_dev/templates` and updated build settings.