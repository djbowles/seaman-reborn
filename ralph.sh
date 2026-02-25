#!/usr/bin/env bash
# Ralph Loop - Autonomous development driver for Seaman Reborn
# Usage: bash ralph.sh --tool claude <iterations>
#
# Each iteration:
#   1. Launches Claude with --dangerously-skip-permissions
#   2. Claude reads CLAUDE.md instructions and prd.json
#   3. Picks next pending story, implements, tests, commits
#   4. Exits, loop continues to next iteration

set -euo pipefail

TOOL="claude"
ITERATIONS=1

while [[ $# -gt 0 ]]; do
    case $1 in
        --tool)
            TOOL="$2"
            shift 2
            ;;
        *)
            ITERATIONS="$1"
            shift
            ;;
    esac
done

echo "=== Ralph Loop ==="
echo "Tool: $TOOL"
echo "Iterations: $ITERATIONS"
echo ""

for i in $(seq 1 "$ITERATIONS"); do
    echo "--- Iteration $i/$ITERATIONS ---"
    echo "$(date '+%Y-%m-%d %H:%M:%S') Starting iteration $i" >> progress.txt

    if [ "$TOOL" = "claude" ]; then
        claude --dangerously-skip-permissions -p "Read CLAUDE.md for instructions. Execute one complete iteration of the Ralph loop: pick the next pending story from prd.json, implement it, write tests, ensure quality gates pass, commit, and update prd.json."
    else
        echo "Unknown tool: $TOOL"
        exit 1
    fi

    echo "$(date '+%Y-%m-%d %H:%M:%S') Completed iteration $i" >> progress.txt
    echo ""
done

echo "=== Ralph Loop Complete ==="
echo "Ran $ITERATIONS iterations."
echo "Check progress.txt and git log for results."
