# Isolated CPU installation of the locked Linux runtime

The v8 scientific protocol fixes 76 installed distribution versions in
`requirements-linux-py311-cu128-20261002.lock` (SHA256
`d1f238a916dfdfbc8b89f3de4c28bb3b7220cdd97ecf609ebb726afab0ca8221`).
A shared environment with different first-match versions cannot satisfy this
contract. Create a new environment; leave existing environments unchanged.

`scripts/install_isolated_runtime_cpu.py` is a separate operational helper.
It installs packages only. It does not download models, datasets or repositories,
import Torch or a model, initialize CUDA, allocate a GPU, control queues,
train, evaluate or start a server. Its code and lock hashes must match the
explicit expected hashes supplied by the operator. Before remote execution,
push and independently review the helper's exact source. Preserve the
scientific source and plan frozen by PR26.

Use an existing Linux x86-64 Python 3.11 interpreter with `-I -S` to create
a fresh venv inside a new absolute attempt directory under the user's
workspace. These bootstrap flags disable inherited Python paths and site
startup before the fresh environment exists. The
underlying Python and standard library are host inputs, whose identity is
recorded; this does not rebuild Python or the operating system. The venv
must disable system site packages and user site packages. Installed
third-party distributions and the six core import locations must resolve
inside that new venv. The helper does not install the Open-Jev checkout.

The attempt and consumed marker are exclusive. Existing outputs, failed
attempts and ambiguous paths are rejected. Caught errors and catchable
interruptions keep their logs and failure receipt and do not resume. An
uncatchable kill or host loss can leave a consumed attempt without a final
receipt; that attempt is still not reusable. The operation has a 1,800-second
deadline while its controller is alive. Timeout and catchable-signal cleanup
control only the helper's newly created subprocess session, never a
pre-existing task. A surviving child after abrupt controller loss requires
separate verification; this helper is not an independent restoration guard.

Every pip invocation uses the isolated interpreter, a scrubbed environment,
`PIP_CONFIG_FILE=/dev/null`, no inherited proxy, Python path, pip settings
or credentials, and disabled keyring lookup. Installations additionally
use explicit public indexes, no dependency resolution, wheels only, no
cache and zero automatic network retries. The final `pip check` does not
contact a package index. Torch 2.8.0+cu128 comes from the
CUDA 12.8 PyTorch index; the remaining 75 locked distributions come from
PyPI. Source distributions and their build hooks are excluded. CUDA device
visibility is empty for these CPU installation subprocesses.

Record the original lock and the derived PyPI-only lock separately. Pip's
actual install reports identify downloaded wheel URLs and SHA256 hashes.
These observed artifact hashes improve reproducibility but do not turn the
original version lock into a predeclared artifact lock. Index availability
and compatible wheel selection are observed at installation time.

A successful result requires all 76 exact metadata versions, isolated
package and distribution positions, disabled inherited/user site packages,
and a successful `pip check`. It records interpreter/venv/host identity,
install reports and logs. Metadata and `find_spec` checks do not import the
model runtime or prove that a GPU model can load. A version mismatch,
missing compatible wheel, timeout or failed `pip check` is a failed result;
do not substitute a different version under the same lock.

Actual installation evidence belongs in a separately reviewed report.
Preparing this helper or passing mock tests is not a clean installation
result. Even a successful CPU install supplies no current GPU ownership,
restoration evidence, model capability, latency or memory result. A fresh
native loader, owned launcher and independent restoration guard remain
required before the v8 model phase.
