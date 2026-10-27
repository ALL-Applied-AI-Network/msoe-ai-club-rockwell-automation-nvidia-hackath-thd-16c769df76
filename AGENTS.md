# AGENTS.md: Team Thd, Rockwell Automation + NVIDIA Physical AI Hackathon (Fall 2026)

Context for any AI coding agent (Codex, Claude Code, ...) working on this project.
Read this whole file before doing anything. Last updated: Tue Oct 27, 2026.

---

## 1. The event and deadlines

- Four-week physical AI hackathon at UWM's Connected Systems Institute (CSI), run by Rockwell Automation,
  NVIDIA, the MSOE AI Club and UWM-CSI.
- **Final submission: Sun Nov 1, 11:59 PM CT.** Each team submits:
  a ~5-minute video, a one-page project summary, a link to this repo, and references/credits.
- **Finals: Thu Nov 5, 5-8 PM, MSOE Diercks Hall** (live presentation).
- Judges (Rockwell, NVIDIA, MSOE, UWM) look for:
  1. the solution **running on real hardware** (the CSI line), booked as a supervised test with CSI
  2. a **specific problem** and why it matters on this line
  3. a **business case** for other manufacturers (scrap/rework/downtime/labor saved, what adoption takes)
  4. **measured results against a baseline**, and honesty about where it still fails
- Help contacts (Slack workspace for the hackathon):
  - Line, data, CSI visits, supervised tests: Shamar Webster, `#data-and-testbed-requests` (webste63@uwm.edu)
  - Dataset questions: Brett Storoe / Dr. Derek Riley / Tanner Cellio, `#dataset-2-fluid-mixing`
  - Digital twin: Dr. Riley (driley@nvidia.com)
  - Compute: Dan Schneider, `#compute-questions`

## 2. Our problem (Dataset 2: Fluid Mixing Station 3)

Four tanks feed one nozzle through a **shared mixing path**:
T-100 = clear diluent, T-200 = orange, T-300 = out of service (never use), T-400 = blue.
Fixed dispense order: **T-200 -> T-400 -> T-100** (skipping unused tanks).

When the recipe changes, residue from the previous recipe carries into the **first vial of the new batch**
("carryover"), so it comes out discolored. Vials 2 and 3 of a batch are clean.
The existing Teledyne vision check only compares color to a reference database; it does not know the
intended recipe and does not warn about carryover.

**Our solution:** a live carryover monitor that
- **PREDICTS** risk before the first vial is filled (from how different the next recipe is),
- **DETECTS** carryover during the pour (a color the recipe cannot make, e.g. orange in a no-T-200 recipe),
- **CONFIRMS** it when the batch's 2nd vial finishes (1st vial's color vs 2nd vial's color),
- and tells the operator what to do (flush / divert / scrap vial N).

Safe recipe limits (never violate; never change recipes without CSI's guidance):
max 80 mL per vial; at least 10 mL of any tank used; **at most 50 mL from T-200**; never T-300.

## 3. Environment

### DGX Spark (YAAD GPU Lab)
- Needs the **CSI VPN** (FortiClient, SSL-VPN, `vpn.csi.uwm.edu`, port 443) unless on CSI's network.
- Login: `ssh yt-thd0812@172.30.102.91` -> root in our team workspace. Each teammate uses their own SSH key.
- **Team workspace path: `/home/yt-thd0812/workspace`.** Do NOT use `~/workspace`: `~` is `/root` here,
  which is NOT the shared workspace.
- Everything in the workspace is **deleted after the event**: commit code to this GitHub repo often.
- We share one Spark's memory (up to 100 GB) between teammates: don't start several heavy jobs at once.
- Installed so far: python3, pandas, Pillow, numpy, matplotlib, tmux, unzip, pip3, `usd-core` (pxr).
  Install with `apt-get install -y ...` or `pip3 install --break-system-packages ...`.
- Use `tmux` for long-running work (`tmux new -s work`, reattach with `tmux attach -t work`).
- Anything opened on a port (Jupyter, dashboards) must have a password/token: other teams can reach open ports.
- Never commit or print secrets: CSI password, SSH private keys, YAAD `yaad_pat_` tokens.

### Workspace layout
```
/home/yt-thd0812/workspace/
  data/csi-vial-fill-dataset/       real camera dataset (do NOT commit; ~700 MB)
    vial_fill/  {images/, videos/, image_index.csv, labels/}
    tank_level/ {images/, videos/, image_index.csv}
    cap_seal/   {images/, videos/, image_index.csv}
    videos.csv, README.md
  twin/csi_digital_twin_v1/         digital twin (do NOT commit)
    stations/fill_station/fill_station.usda, tools/render_example.py, tools/render_robust.py, test_plan.csv
  code/                             our scripts (mirrored in this repo under dataset2/code/)
  out/                              generated results (CSVs, PNGs, events.jsonl, queue.csv)
  results/<job_id>/                 output of YAAD batch jobs (job.log, out/)
```

### YAAD batch jobs (website https://www.yaadops.com, or the YAAD MCP server)
- Jobs run in a fresh container; the "input folder" (relative to the workspace) is copied to `/job/in`
  (the working directory); write outputs to `/job/out`; results land in `results/<job_id>/`.
- Agents can use YAAD over MCP: URL `https://www.yaadops.com/api/mcp`, header `Authorization: Bearer $YAAD_TOKEN`.
  For Codex CLI, add to `~/.codex/config.toml`:
  ```
  [mcp_servers.yaad]
  url = "https://www.yaadops.com/api/mcp"
  bearer_token_env_var = "YAAD_TOKEN"
  ```
  Tools: `scope_job` (call first), `submit_job`, `get_job`, `list_jobs`, `fetch_results`, `report_issue`, `wait_for_answer`.

## 4. The dataset: facts and the label problem (important)

- 3 synchronized cameras, 640x480, stills every ~2 s, timestamp in each filename
  (`vial_fill_YYYYMMDD_HHMMSS_micro.jpg`). 5 videos; only **video 5** has labels.
- README rule: video 5 ran recipes 5-14, 3 vials each, vial N -> recipe `5 + (N-1)//3`.
- **The official labels are misaligned.** `vial_fill/labels/video5_frame_labels.csv` time windows drift from
  the images (about 10 s early at vial 7, 40 s by vial 9, growing). Detecting fills from the images shows
  **video 5 contains only vials 1-25** (vial 25 cut off at the end); vials 26-30 are not in the video.
  Real fills are ~65 s apart; the labels assume ~56 s.
- **Use `dataset2/results/video5_labels_corrected.csv`** (vial, recipe, tank volumes, real start/end, clock times).
  Video 5 starts at 2026-09-23 19:47:10.
- Not yet reported on Slack (draft is in Section 9).
- Organizer rule: look at real data, but **train on synthetic data from the digital twin** (high-mix, low-volume
  reasoning). Suggested evaluation: **leave-one-recipe-out**; report false alarms and misses separately.

### Measurement details
- ROI (vial bottom under the nozzle) in 640x480 pixels: `(230, 330, 380, 440)` (left, top, right, bottom).
- "Colored" pixel = HSV saturation > 90 and value > 40 (Pillow HSV, 0-255). Liquid present if colored fraction > 0.015.
- A fill = colored for at least 3 consecutive frames; it ends when color drops to ~0 (vial leaves).
- Hue on Pillow's 0-255 scale: orange ~15-20, olive/yellow ~40-55, green ~70-100, teal ~105-125, blue ~140. Hue wraps around.

## 5. Results so far

### Batch-consistency detector (`carryover_baseline.py`)
Flag a vial if its hue differs from its recipe batch's median hue by more than T.
23 scored vials (2-24), 7 carryover vials (first vial after each recipe change):

| T | caught | missed | false alarms |
|---|---|---|---|
| 15 | 7/7 | 0% | 3/16 (19%): vials 3, 12, 15 |
| 20 | 6/7 (misses vial 13) | 14% | 2/16 (12%) |

Predictive check: recipe change score (0.5 x L1 distance of [orange, blue, clear] fractions) vs observed hue
shift in the first vial gives r ~ 0.88 over 7 transitions (small sample). The smallest change (8->9, 0.21)
gave the smallest shift (15.7).

### Real-time color prediction with leave-one-recipe-out (`realtime_loro.py`): did NOT work well
Linear RGB mix of learned "tank colors" -> best around 6/7 caught with 75% false alarms. Reasons: clear only
dilutes (it was learned as olive), dye absorption is non-linear (Beer-Lambert), and only 8 recipes.
Possible improvement: model absorbance in log space with dye concentration. Use this as an honest finding.

### Live monitor (`live_monitor.py`): WORKS on a replay of video 5
- Replays frames at real or faster speed (`--speed 1`, default 20, 0 = no wait) or watches a folder
  (`--source watch --dir ...`). Reads the recipe queue from `out/queue.csv` (auto-built from corrected labels).
- Ignores color flickers shorter than 3 frames (a 4 s blip previously shifted every vial number).
- Results: PREDICT warned for all 7 recipe changes 20-44 s before the new recipe's first fill; DETECT caught
  vial 7 (orange pouring into a no-orange recipe) right as its fill started, ~26 s before it left;
  CONFIRM caught **7/7** carryover vials (vial 1 correctly passed as OK).
- Limitation: CONFIRM arrives ~50-75 s after the bad vial leaves the fill station; whether that is before
  capping depends on the line's fill-to-capping time (ask CSI). It only judges first vials.
- Events are also written to `out/events.jsonl` (for a dashboard).

## 6. Digital twin facts

- Open `stations/fill_station/fill_station.usda` (loads fast). Camera `FillCam` = real `vial_fill` camera, 640x480 pinhole.
- Animation: time codes 1-686 at 30 fps. Pour stream visible frames 12-368 (blue) and 586-661 (clear);
  liquid visible from frame 21. **Fill level = frame** (liquid mesh points animated). 686 = finished vial.
- Liquid material: `/FillStation/_materials/USD_blue_liquid_approximation/Principled_BSDF` (UsdPreviewSurface),
  `diffuseColor = (0.006, 0.2, 0.3)`, `opacity = 0.6`. Bound to mesh `.../Fill_liquid/Fill_liquid`.
- Stream material: `/FillStation/_materials/USD_jet_approximation/Principled_BSDF`,
  `diffuseColor = (0.01, 0.57, 0.67)`, `opacity = 0.5`. Bound to `.../Fill_dispensing_stream/Fill_dispensing_stream`.
- Tanks are not modeled; liquid color is an approximation (so the twin is best for robustness: fill levels,
  lighting, rare recipes, carryover look; calibrate colors from real vial RGB in `out/realtime_loro_results.csv`).

### Running Isaac Sim on the Spark (YAAD job)
- Image `nvcr.io/nvidia/isaac-sim:6.0.1` (arm64, works on the GB10), 32 GB, GPU, 60 min,
  input folder `twin/csi_digital_twin_v1`, env `ACCEPT_EULA=Y` and `PRIVACY_CONSENT=Y`. Python is `/isaac-sim/python.sh`.
- The original `tools/render_example.py` failed: `RGB capture returned shape (0,)`. Isaac Sim started fine;
  it captured ~1 s after startup, before RTX produced a frame. "OmniHub failed", "GLFW initialization failed"
  and "failed to open the default display" warnings are harmless (headless, Hub unsupported on Spark).
- `tools/render_robust.py` adds warm-up, capture retries, liquid/stream color + opacity overrides
  (session layer only), `--renderer PathTracing|RayTracedLighting`, and batch mode (`--plan CSV --out-dir`,
  writes `labels.csv`). Test command:
  ```
  /isaac-sim/python.sh tools/render_robust.py --camera FillCam --stage stations/fill_station/fill_station.usda --plan test_plan.csv --out-dir /job/out
  ```
  **Status: written but not yet confirmed working.** If captures keep failing, retry with `--renderer RayTracedLighting`.

## 7. Scripts (in `dataset2/code/`, run on the Spark from `/home/yt-thd0812/workspace/code`)

| Script | What it does |
|---|---|
| `contact_sheet.py` | last labeled frame per vial (revealed the label mismatch) |
| `check_alignment.py` | frames + labels in a time window: `python3 check_alignment.py 195230 195600` |
| `color_timeline.py` | colored fraction + hue per video-5 frame in the ROI -> `out/color_timeline.csv/.png` |
| `segment_fills.py` | finds fills from images (MAX_GAP = 0) -> `out/fills_detected.csv`, 25 fills |
| `carryover_baseline.py` | corrected labels + batch detector + predictive check |
| `realtime_loro.py` | real-time color model with leave-one-recipe-out (weak; saves `out/tank_colors.npy`) |
| `live_monitor.py` | live PREDICT / DETECT / CONFIRM monitor (replay or watch-folder) |
| `twin/.../tools/render_robust.py` + `test_plan.csv` | Isaac Sim rendering + recoloring for synthetic data |

Paths in the scripts are hard-coded to `/home/yt-thd0812/workspace/...`.

## 8. What's left (priority order)

1. **Real-line test at CSI** (required by judges). Plan: run e.g. A x3 -> B x3 -> C x3 within safe limits
   (A: T-200 40 + T-100 20; B: T-400 50 + T-100 20; C: 20/20/20). Record per vial: recipe, PREDICT score,
   DETECT, CONFIRM, human judgment (discolored yes/no + phone photo), Teledyne pass/fail and weight, timing.
   Keep the camera recordings as a new held-out test set. Run `live_monitor.py --source watch` if live images
   are reachable; otherwise process right after. Check the ROI on a fresh image first (camera may have moved).
2. **Operator web dashboard** reading `out/events.jsonl` (live alerts, current vial actual vs expected color,
   red banner on carryover). Server-Sent Events or WebSocket; protect the port with a token.
3. **Dry-run the monitor on unlabeled videos 1-4** (needs a queue for them, or run in "no recipe" mode).
4. **Synthetic data:** confirm `render_robust.py` works, then render varied colors/fill levels/lighting
   (clean recipe colors + carryover colors calibrated from real vials); use it to test the detector.
   Time-box: drop it if it blocks the deliverables.
5. **Business case:** cost of scrapped first vials, flush time and fluid, operator checks, line stops;
   what our system saves; what adoption takes (camera, recipe queue feed from the PLC via OPC UA or
   Ethernet/IP, setup time per new recipe, operator trust).
6. **One-page summary**, final `README.md` for the repo (problem, approach, results, how to run, credits).
7. **~5-minute video:** line footage + `live_monitor.py --speed 1` screen recording side by side.
8. Credits/references: dataset (CSI / tcellio@nvidia.com), digital twin, Isaac Sim, libraries, AI tools used.

## 9. Pending messages

Slack `#dataset-2-fluid-mixing` (not yet posted):
> Team Thd here. We think `video5_frame_labels.csv` is misaligned with the vial_fill images. Detecting fills
> directly from the images, video 5 seems to contain ~25 fills (vials 1-25), not 30. Fills are ~65 s apart but
> the label windows assume ~56 s, so they drift (~10 s early at vial 7, ~40 s by vial 9). Example: the green
> carryover vial (vial 7 on the dataset page, filling ~19:52:57-19:53:25) is labeled vial 8. Is there a known
> offset or corrected file? Our image-based corrected labels are in our repo.

## 10. Status from the CSI visit (Oct 27): fill in

- Line test date/time: ___
- How live images can be accessed (folder path / stream / frame rate): ___
- Recipe queue / batch history access (manual, export, PLC tags): ___
- Fill station -> capping time: ___   Can bad vials be diverted? ___
- Flush procedure, time, fluid wasted: ___
- Cost / scrap numbers for the business case: ___
- Teledyne pass/fail + weight per vial available? ___
- Answers about the label drift / held-out data: ___

## 11. How to work with this team

- Teammates are students learning as they go: **explain each command and what it does in plain language**
  before running it, and keep steps small.
- **Ask before deleting or overwriting** anything in the workspace or the repo.
- Long scripts: write them as files (pasting long heredocs into the terminal has corrupted files before).
- Don't commit large data (dataset, twin, videos, model weights); keep them out with `.gitignore`.
- `git pull` before starting work; commit and push often with clear messages.
- Don't overclaim results: small samples (23 vials, 7 carryover cases); ground truth assumes "first vial after
  a recipe change = carryover". State limitations clearly; judges asked for them.
