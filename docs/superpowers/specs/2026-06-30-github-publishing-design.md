# GitHub publishing design

Prepare the existing desktop application as a small, directly runnable GitHub repository.

- Replace stale package imports with local-module imports.
- Add a concise README, dependency list, and Python/macOS ignore rules.
- Remove generated caches and exclude local scan data and application outputs.
- Keep the current flat source layout; do not add packaging or PyPI scaffolding.
- Add no license.
- Verify with the existing test suite and a clean-tree artifact scan.
