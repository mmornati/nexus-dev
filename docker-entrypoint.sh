#!/bin/bash
set -e

# If the first argument is "nexus-dev" or looks like a flag, run the server
if [ "$1" = "nexus-dev" ]; then
    shift
    exec python -m nexus_dev.server "$@"
elif [ "${1:0:1}" = "-" ]; then
    exec python -m nexus_dev.server "$@"
fi

# If the command is a known nexus CLI tool, execute it
if [[ "$1" == nexus-* ]]; then
    COMMAND="$1"
    shift
    # Replace hyphens with underscores for module path (except nexus-dev)
    # e.g. nexus-init -> nexus_dev.cli:init_command
    # But currently the project.scripts maps them conveniently.
    # We can just call the executable directly since they are installed in the path.
    exec "$COMMAND" "$@"
fi

# Fallback: just execute the command
exec "$@"
