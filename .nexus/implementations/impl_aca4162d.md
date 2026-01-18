---
title: Isolated Global Installation Support
timestamp: '2026-01-18T15:41:09.321153+00:00'
project_id: f959454c-12b6-4062-b411-59fa58f83777
files_changed:
- README.md
- docs/quick-start.md
- pyproject.toml
- src/nexus_dev/cli.py
- src/nexus_dev/templates/pre-commit-hook (moved)
related_plan: ''
---

# Isolated Global Installation Support

## Summary
Enabled isolated global installation of Nexus-Dev via pipx or uv by fixing resource packaging and path resolution issues.

## Technical Approach
Reorganized resource files into the package source tree, updated CLI path resolution using package-relative paths, and revised build metadata in pyproject.toml. Recommended pipx/uv for isolated global installation.

## Design Decisions

- Moved templates to `src/nexus_dev/templates` to ensure seamless packaging without complex data-file mapping.
- Used `Path(__file__).parent` for robust path resolution regardless of installation location.
- Recommended `pipx` to avoid the 'sourcing venv' friction reported by the user.

## Files Changed

- `README.md`
- `docs/quick-start.md`
- `pyproject.toml`
- `src/nexus_dev/cli.py`
- `src/nexus_dev/templates/pre-commit-hook (moved)`