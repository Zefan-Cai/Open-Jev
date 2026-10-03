# V8 controlled operational imports

This separate CPU preparation imports the no-borrow adapter and its process and
resource parser dependencies from verified source bytes. The scientific and
native checkouts remain at their frozen commits. The existing source loader,
adapter, resource episode and execution refusals are unchanged.

Use a fresh trusted `python -B -I -S` interpreter with three separate clean Git
checkouts and external declarations. The operational declaration includes the
new bridge and its dependency closure. The bridge bootstraps the unchanged,
pinned source verifier from captured bytes, checks all three declared source
inventories, and compiles the operational helpers from the captured verified
bytes. It does not use their cached bytecode or accept a module merely because
its `__file__` claims the expected path.

The import scope audits module and loader identity, actual origins, functions,
class methods and critical dependency references. Conflicting namespaces,
optional model packages, foreign import paths and hooks, changed sources and
substituted callables must cause refusal. Import state is restored when the
scope exits, including on failure.

CPython's direct file startup may leave a negative (`None`) importer-cache
entry for the exact bridge script. That one entry is permitted; a positive
finder at the same key and other foreign entries remain rejected. A separate
front finder hard-denies the exact `org` namespace with `ModuleNotFoundError`
so older CPython standard-library `copy` code can handle its optional Jython
probe as absent. It never searches a fallback path or supplies a positive spec.
Preloaded `org` modules are refused, the denial hook is audited, and all other
names continue through the unchanged pinned source finder.

A supplied `assert_owned` callback brackets verification, loading and scoped
CPU inspection. This is a trusted predicate dependency, not a GPU lease or
independent resource attestation. A no-op callback in a CPU source proof is
explicitly a fixture; it does not prove live ownership. The scoped wrapper
refuses use after exit. Python module or function references retained by trusted
inspection code remain ordinary callables: this is provenance verification,
not a sandbox, a revocation mechanism or a restriction on arbitrary Python.

The CPU entrypoint performs imports and source audits only. It does not call
the adapter's live probe, create a cooperative reservation, run an owned worker,
pause a queue, import Torch, initialize CUDA or allocate a model. Production
execution remains unavailable before reading a request. Importing a resource
episode module does not demonstrate that episode or its lazy restoration path.

Future execution still requires independently reviewed launcher and recovery
integration, fresh actual host and physical resource authority, full runtime,
data and weight validation, native numerical and tensor observations, and the
original first-step gradient checks. Existing model, installer, proof and replay
attempts remain consumed. The fixed scientific plan, data membership, 733-step
pass and comparison gates are unchanged. CPU source provenance establishes no
model improvement, natural trial or business action.
