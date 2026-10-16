# Infrastructure release scope

This release preserves the current matrix solver implementations and numerical
policies while separating them from study execution and deployment machinery.
It is a substantial API update from public version 0.1.0, not a promise that old
benchmark scripts run unchanged.

Included:

- Matrix preparation and preconditioner interfaces, including optioned
  hierarchies that share setup across application configurations.
- All six solvers, charged certification, bounded continuation, and one-shot
  correction restarts with global-solution diagnostics.
- Versioned raw and coefficient storage, convergence tables, work counters,
  and generic execution/resource reporting.
- Exact external source pins, portable synthetic regression tests, and examples.

Not included:

- Series-of-systems ingestion, storage, and rebuild-policy orchestration.
- Production experiment grids, dataset selections, results, and analyses.
- Scheduler/fleet configuration, sysimage bundles, or cluster maintenance tools.
- Historical result migration/repair tools and exploratory numerical notebooks.

The source retains optional native-bundle lookup support, but building and
deploying such bundles is outside this release. Ordinary setup uses local
libraries.

Generic solver policy code is public in this scope. Omitting experiment grids
does not conceal certification, continuation, or correction-restart mechanisms.
