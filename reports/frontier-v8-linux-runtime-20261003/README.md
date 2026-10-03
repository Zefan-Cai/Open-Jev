# Actual isolated Linux runtime installation

A fresh Linux x86-64 / Python 3.11 environment installed all **76 exact
locked distribution versions**. Both wheel installation reports contain
the expected package sets and artifact hashes. All distribution roots and
six core module locations resolve inside the new venv; user and system
site packages are disabled. `pip check` returned **No broken requirements
found**. The single installation took **93.346 seconds**.

The installer was committed and pushed as
`360a9ef584352a876dbd9eaccf7a0b7332dea73c` before execution. Its SHA256 is
`acf804c8af5fbdf9357bb808482e0af0186cc0e78cc869fd80e15ca7fad5e606`.
The original 76-version lock remains
`d1f238a916dfdfbc8b89f3de4c28bb3b7220cdd97ecf609ebb726afab0ca8221`.
The frozen v8 scientific source, plan and data were not changed.

The original host Python binary and boot identity were unchanged after
installation. The owned installer was reaped, and the postflight read-only
process check found no command referencing this episode's directory.
Existing environments were not installation targets. No models, GPU jobs,
queue controls, training, inference or external customer trials were run.

## Evidence

- [Verification and exact-copy hashes](verification.json)
- [Actual installation receipt](actual-install-r1/attempts/install-r1/installation-receipt.json)
- [Torch wheel provenance](actual-install-r1/attempts/install-r1/torch-install.report.json)
- [Other 75 wheel artifacts](actual-install-r1/attempts/install-r1/pypi-install.report.json)
- [Package versions and isolated locations](actual-install-r1/attempts/install-r1/metadata.log)
- [Dependency check](actual-install-r1/attempts/install-r1/pip-check.log)
- [Actual launch command and scrubbed environment](actual-install-r1/launch.json)
- [Postflight check](postflight.json)
- [Installer review](installer-independent-final-review-r1.json)
- [Independent actual-install review](independent-actual-install-review-r1.json):
  425 checks passed, with no blockers.
- [Regression receipt](root-regression-r2.json): 12 focused tests; 1,185 total,
  including 1,100 passed and 85 skipped.

The retained contract findings describe the pre-install fixes: include
already-satisfied bootstrap packages in wheel reports, isolate bootstrap
startup with `-I -S`, clean up children during catchable signals, reject
other environment ancestors, and accept the exact official PyTorch CDN
host selected by its CUDA 12.8 index. The lock was not relaxed.

The final installer review preceded the installation request. The separate
transport review receipt was recorded after that request; it is retained
as a review of the transport code, rather than prelaunch approval.
The reviewer also retained an audit-only checker correction: the prepared
data manifest is an external input bound by frozen hashes, not a Git blob.
The installation and original evidence were not rerun or changed.

## Scope and next step

This is an actual isolated **CPU installation** result. The existing host
Python, standard library and operating system remain inputs. The original
lock specifies versions; the pip reports separately identify the wheel
artifacts actually fetched. They are not a predeclared artifact hash lock.
The independent artifact check compares saved reports and index metadata;
it does not independently rehash retained wheel bytes or installed package
files, and it does not refresh that index metadata.

Metadata and `find_spec` checks do not import Torch or establish package
ABI compatibility, CUDA initialization, model loading, model quality or
GPU ownership. The controller has an installation deadline while alive;
this helper is not an independent guard against abrupt controller loss.

The v8 733-step training and fixed comparison have **not run**. A separately
reviewed native loader, owned launcher and independent restoration guard
remain required before the model phase. This installation supplies no
GPU lease, model promotion or benchmark improvement.
