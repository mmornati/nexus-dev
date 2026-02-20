import subprocess
import sys

res = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/unit/test_cli.py::TestCliStatus::test_status_shows_project_info", "-v", "-s", "-W", "default"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    cwd="/Users/mmornati/Projects/nexus-dev/.venv/bin/.."
)
print("STDOUT:")
print(res.stdout)
print("STDERR:")
print(res.stderr)
