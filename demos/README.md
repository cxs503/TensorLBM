# Drag-echo interactive demo (`echo_slider.html`)

A single-file, dependency-free web page that drives the TensorLBM
drag-surrogate **echo API** of PR #241 (branch `exp/b4-echo`, commit
`98c617f7`, `app/backend/routers/drag_echo.py`). It exists to make one
product claim tangible: **a geometry change during hull design gets an
immediate C_D answer with an uncertainty band and an honest verdict**.

What you can do on the page:

- move the SUBOFF design sliders (`l_over_d_mult`, `nose_len_mult`,
  `stern_len_mult`, `sail_x_mult`, `sail_scale`, `fin_scale`), pick the hull
  type (`bare_hull` / `with_sail` / `full`), set `u_in` and the Reynolds
  range (log-spaced, 20–900 by default) — every change fires one
  `POST /api/drag/echo/params` call and redraws the C_D-vs-Re curve with the
  deep-ensemble min/max band;
- run a **sweep**: pick an axis, one batched `POST /api/drag/echo/sweep`
  (16 geometries, 3 Re levels) renders C_D-vs-axis curves;
- watch the **verdict banner**: the worst guard verdict over the whole query
  batch (`reject > review > ok`), the guardrail reasons verbatim, the
  `confident` flag and the ensemble member-σ summary;
- hover both charts for exact values; open *data table* under each chart for
  the numbers behind it. Dark mode follows the OS setting (toggle in the
  header). Everything is hand-rolled inline SVG — no CDN, no chart library,
  no build step, works in an offline lab.

## The honesty contract (why a beautiful curve can say REVIEW)

Every answer the backend serves carries a guardrail verdict, and the page
promotes it to the most prominent element on screen instead of burying it in
a payload. This matters because the served surrogate is a deep ensemble
trained on a finite corpus: a *hull-form variant* (any of the CAD deformation
axes off 1.0) produces a perfectly smooth, plausible-looking C_D curve —
while being outside the training corpus. The channel-space guard alone would
answer `ok` there, so PR #241 downgrades every non-mother design to at least
`review` and makes `confident` require `ok` with no unsupported channels.
The page mirrors that contract visually: an `ok` answer draws a solid band;
any `review`/`reject` answer keeps the curve but marks it — the badge turns
amber/red, the tag **out of served corpus** appears on the chart, and the
uncertainty band switches to a dashed/hatched rendering. The numbers are
never hidden, and neither is their epistemic status.

The second half of the contract is the reasons list, shown **verbatim** from
`guard.reasons` (e.g. the hull-form downgrade text naming exactly which axes
are out of corpus, or the STL path listing every channel that is not
derivable from an arbitrary mask). If the backend refuses or fails, the error
banner shows the exact HTTP detail and the charts keep their last good data —
degraded, never silently wrong. Worst verdict wins across the whole batch:
one bad geometry in a 16-point sweep turns the whole sweep amber, because the
operator must not average away an out-of-corpus point.

## Running it

### Mode (a) — standalone dev mode (today, needs branch `exp/b4-echo`)

1. Start the backend (CPU is fine; ensemble load takes a moment):

   ```bash
   cd /nfs/wangxi/worktrees/b4_echo/app
   PYTHONPATH=src:app \
     TENSORLBM_DRAG_CKPT_DIR=/nfs/wangxi/runs/b4_serve_20260824/ckpts \
     TENSORLBM_DRAG_RUN_DIR=/nfs/wangxi/runs/b4_v4_20260824 \
     TENSORLBM_DRAG_ECHO_DEVICE=cpu \
     python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
   ```

   (Env vars follow PR #239: `TENSORLBM_DRAG_CKPT_DIR` picks the 5-member
   serving ensemble, `TENSORLBM_DRAG_RUN_DIR` the v4 corpus cache. Without
   them the endpoints answer 503 with the reason.)

2. Serve the page and open the printed URL:

   ```bash
   python demos/serve_demo.py 8765        # add --open to launch a browser
   # -> open http://localhost:8765/echo_slider.html?api=http://<lan-ip>:8000
   ```

   The `?api=` query param (or the API-base field in the page header, kept
   in `localStorage`) points the page at the backend; the launcher prints it
   prefilled. Cross-origin use from this launcher is fine: it sends
   permissive CORS headers, and the FastAPI app ships CORS middleware.

### Mode (b) — next to the FastAPI app (after #241 merges)

Copy `demos/echo_slider.html` anywhere served by the same origin as the API
(e.g. a static dir mounted by `app/backend/main.py`) or leave the API base
empty behind the same proxy. No launcher, no CORS, no `?api=` needed. This
mode is documented, not implemented — this PR does not touch `app/`.

## Files

| file | purpose |
|---|---|
| `echo_slider.html` | the demo page (inline CSS/JS, hand-rolled SVG charts) |
| `serve_demo.py` | stdlib launcher with CORS + `?api=` URL hint |
| `../tests/test_demo_page.py` | HTML static checks + launcher import/HTTP test |

Screenshots: deliberately not committed (no binaries in git) — take one with
e.g. `gnome-screenshot` when demoing and attach it to the PR description.
