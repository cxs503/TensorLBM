# Model Zoo — a registry of trained artifacts

The AI side of TensorLBM has four independent model-persistence conventions
(`save_drag_regressor`, `save_fno2d`, `save_model`, `save_flow_transformer_model`),
each writing a weight file plus a JSON sidecar. None of them records *provenance*:
which task a trained artifact serves, which dataset (and split) produced it, which
metrics it achieved, and which code revision trained it. The zoo
(`tensorlbm.zoo`, roadmap item **C1**) adds that missing layer for publishable
weight sets — without inventing a fifth serialization format.

## Quickstart

```python
from tensorlbm.zoo import ModelZoo, SUGGESTED_LOADERS

# 1. train and save exactly as before — no new save convention
from tensorlbm.ai.model import EddyViscosityMLP, save_model
model = train_your_eddy_mlp(...)          # tensorlbm.ai.train works fine here
save_model(model, "/tmp/eddy_v3.pt")      # writes eddy_v3.pt + eddy_v3.pt.json

# 2. register into the zoo (default root ~/.tensorlbm/zoo, or pass root=...)
zoo = ModelZoo("/data/zoos/tensorlbm")    # ModelZoo() uses the default root
info = zoo.register(
    "/tmp/eddy_v3.pt",
    model_id="eddy-viscosity-cyl-v3",
    task="eddy-viscosity",
    loader=SUGGESTED_LOADERS["eddy-viscosity"],
    metrics={"test_mape": 4.2, "n_test_samples": 8},
    dataset={
        "path": "/data/campaigns/cyl_re_scan_v3",
        "split": "train=64/val=8/test=8 (point-level, seed 7)",
    },
    notes="15-epoch Adam run; best of 3 seeds",
)

# 3. later — anywhere the zoo directory is available
models = zoo.list_models(task="eddy-viscosity")
model = zoo.load("eddy-viscosity-cyl-v3")  # restored via load_model()
report = zoo.validate("eddy-viscosity-cyl-v3")
assert report.ok
```

The zoo root is resolved in this order: an explicit `root=` argument, then the
`TENSORLBM_ZOO_ROOT` environment variable, then `~/.tensorlbm/zoo`
(`tensorlbm.zoo.resolve_zoo_root`).

## Zoo layout

A zoo is just a directory — there is **no index database**, so it stays
self-describing and shareable (`rsync`/`zip` the folder and it works elsewhere):

```
<zoo_root>/
  eddy-viscosity-cyl-v3/     # entry directory == model_id
    eddy_v3.pt               # the artifact (copied in by register; move=True moves it)
    eddy_v3.pt.json          # the saver's sidecar, travels with the weights
    model.json               # the zoo manifest — the single source of truth
```

## Manifest schema (`model.json`)

| Field | Type | Meaning |
|---|---|---|
| `schema_version` | `int` | Must be `1` (`ZOO_SCHEMA_VERSION`); any other value is rejected |
| `model_id` | `str` | Unique id, lowercase kebab/snake (`^[a-z0-9][a-z0-9_-]{0,63}$`); equals the entry directory name |
| `task` | `str` | Task label, e.g. `drag-surrogate` / `eddy-viscosity` / `flow-transformer` / `fno2d` (`KNOWN_TASKS`) |
| `loader` | `str` | `"module:attr"` import string of the `load_*` function, e.g. `"tensorlbm.ai.model:load_model"` |
| `artifact` | `str` | Weight filename inside the entry dir (bare filename — no path components) |
| `artifact_sha256` | `str` | SHA-256 of the artifact, recorded at registration (integrity check) |
| `artifact_bytes` | `int` | Artifact size in bytes |
| `artifact_companion` | `str \| null` | The saver's JSON sidecar (`<artifact>.json`) when it travelled with the weights |
| `metrics` | `dict[str, scalar]` | Standard evaluation numbers, e.g. `{"test_mape": 4.2}` (flat, finite scalars) |
| `dataset` | `dict \| null` | Dataset lineage; recommended keys `path` and `split`, further keys free |
| `code_sha` | `str` | Git SHA that produced the artifact (auto-captured, `+dirty` suffix when the tree was modified) |
| `created_at` | `str` | ISO-8601 UTC timestamp |
| `notes` | `str` | Free-form context |

The schema is enforced **fail-closed on write and on read**: unknown keys,
wrong types, non-scalar metrics, NaN values, path-like `artifact`/`model_id`
values (traversal guard), or timestamp formats that don't parse are all
rejected with a `ZooError`/`ZooManifestError` describing the violation.

## API

`ModelZoo(root=None)` (or the module-level functions operating on the default
root; the curated aliases live in `tensorlbm.api` as `zoo_register`,
`zoo_load`, `zoo_list_models`, `zoo_info`, `zoo_validate`):

| Call | Behaviour |
|---|---|
| `register(path, model_id, task, loader, *, metrics, dataset, notes, code_sha, move=False, overwrite=False) -> ModelInfo` | Copies (or moves) the artifact + sidecar into `<root>/<model_id>/`, writes the manifest, returns the entry. Fails closed on duplicate ids (unless `overwrite=True`), missing files, invalid metadata, and loader strings that do not resolve. |
| `list_models(task=None) -> list[ModelInfo]` | Lists entries (oldest first). Directories without a manifest are ignored; a present-but-invalid manifest raises instead of being hidden. |
| `info(model_id) -> ModelInfo` | Returns the validated manifest as a `ModelInfo`. `KeyError` when absent. |
| `load(model_id) -> Any` | Imports the manifest's `loader` and calls it with the artifact path. Returns whatever the loader returns — a bare module for eddy/FNO entries, a `(model, norm)` tuple for drag surrogates. |
| `validate(model_id) -> ZooValidation` | Per-check report: `manifest_schema`, `artifact_present` (incl. the declared companion), `integrity` (SHA-256 + size), `loader_importable`, `model_loads`. `report.ok` is the single verdict; nothing raises on entry problems. |

## Relation to the existing `save_*` conventions

The zoo deliberately **reuses** the existing persistence layer and adds no
serialization of its own:

| Task | Save (unchanged) | Loader string for the manifest |
|---|---|---|
| `drag-surrogate` | `tensorlbm.ai.drag_surrogate.save_drag_regressor` | `tensorlbm.ai.drag_surrogate:load_drag_regressor` |
| `eddy-viscosity` | `tensorlbm.ai.model.save_model` | `tensorlbm.ai.model:load_model` |
| `flow-transformer` | `tensorlbm.ai.transformer.save_flow_transformer_model` | `tensorlbm.ai.transformer:load_flow_transformer_model` |
| `fno2d` | `tensorlbm.ai.fno.save_fno2d` | `tensorlbm.ai.fno:load_fno2d` |

All four conventions store their architecture/normalisation metadata in a
`<weights>.json` sidecar next to the weight file; `register` detects that
sidecar, moves it into the entry directory with the weights, and records it as
`artifact_companion` so `validate` can check it is still present (loaders fall
back to default architectures without it and then fail on `load_state_dict`).

The suggested loader strings above are also available programmatically as
`tensorlbm.zoo.SUGGESTED_LOADERS` (keyed by `KNOWN_TASKS`).

## Relation to `tensorlbm.ml.model_registry`

Both layers sit above the same `save_*`/`load_*` conventions:

- **`ml.model_registry.ModelAssetRegistry`** is the platform-side asset layer:
  a SQLite index with stage lifecycle (`development → staging → production`),
  serving cross-links, and training-job/product lineage. It answers "which
  checkpoint assets exist for the platform".
- **`zoo.ModelZoo`** is the lighter, file-manifest layer for *publishable*
  weight sets: no database, explicit loader import strings, dataset lineage,
  standard metric numbers, and SHA-256 integrity — a directory you can hand to
  someone. It answers "which weights should I use, and why".

An artifact can move between the layers without re-saving — both point at
files produced by the same savers.

## Security considerations

- **`load()` executes code.** The manifest's `loader` field is resolved with
  `importlib.import_module` and called — importing the target module runs its
  import-time code. Treat a zoo directory like a package: only register
  artifacts from trusted sources, and inspect unfamiliar manifests with
  `info()` before loading. Prefer the curated `SUGGESTED_LOADERS` strings.
- **Integrity, not authenticity.** `artifact_sha256` detects accidental
  corruption and tampering *after* registration; it is not a signature and
  does not establish who authored the entry. `code_sha` records provenance on
  a best-effort basis (`"unknown"` outside a git checkout).
- **Path confinement.** `model_id`, `artifact`, and `artifact_companion` are
  validated as bare filenames / confined identifiers, so a hostile manifest
  cannot reference files outside its entry directory.
- Weights are loaded with `weights_only=True` by the underlying loaders where
  applicable (no pickle code execution for the four bundled families).
