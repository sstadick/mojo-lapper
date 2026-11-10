#!/usr/bin/env bash
set -euo pipefail

count=0
while IFS= read -r file; do
    mojo run -D ASSERT=all -I . "$file" "$@"
    count=$((count + 1))
done < <(find tests -name 'test_*.mojo' -type f | sort)
if [ "$count" -eq 0 ]; then
    echo "No Mojo test files found" >&2
    exit 1
fi
