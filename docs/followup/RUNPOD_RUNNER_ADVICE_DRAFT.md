# DRAFT: "Advice for agents running jobs" for runpod-runner/CLAUDE.md

Not applied — for Tim to review before it goes into the runner repo. Consolidated from the
sibling repos' notes (paths relative to ~/Documents/VSCode; EC#n = emotion-concepts
docs/ISSUES_LOG.md row n) and from 2026-10-19/20 runs in grader-shifts-persona-space.

Two things to decide first:
- The existing CLAUDE.md line "`/workspace` is the network volume (persists)" is wrong for
  volume-less pods: there it is a pod volume that is sometimes a slow FUSE mount.
- Item 1 below proposes standing permission to `rp down --terminate -y` a pod the agent
  itself created that never got an ip/port. Drop it if you prefer to be asked.

## Launching & capacity
1. **Never let `rp up` block on a pod with no network.** Use `--no-wait` and poll `rp status`;
   if still no ssh after ~5 min, terminate and retry (2026-10-19: an H200 in US-NC-1 sat
   "running" with no ip for 18 min, billing). Same night, a replacement got ssh in 45-60 s.
2. **"No instances currently available" is transient**: retry every ~60 s with a GPU fallback
   list of exact ids (`--gpu "NVIDIA A100 80GB PCIe,NVIDIA A100-SXM4-80GB,NVIDIA H100 80GB HBM3"`);
   aliases can't be combined (`rp/gpus.py`). Only list GPUs the job fits on (EC#27).
3. **Price the GPU to the job.** A 27B bf16 model runs vLLM and HF extraction on one A100/H100
   80 GB ($1.79-3.99/h); an H200 ($5.29/h) is only worth it for throughput.
4. Prefer `--volume none` (a network volume pins one DC; EUR-IS-3 had no GPUs for an hour, EC#13).

## Disk & storage
5. **Check `/workspace` first**: `findmnt -n -o FSTYPE /workspace`. On volume-less pods it can
   be FUSE (pip ~70 min for 12 GB, EC#10; seen again 2026-10-20 in EUR-IS-3). Then put weights
   and venvs on the container disk — and **size `--disk` for them** (≥ 160 GB for one 27B model +
   a vLLM venv): a FUSE pod with the default-ish 80 GB disk failed preflight on 2026-10-20.
6. **Set `HF_HOME`, never default it** (`rp run` exports `/workspace/hf`, so `${HF_HOME:-x}`
   never applies; EC#16).
7. `rp bootstrap`'s venv lives on `/workspace`: pre-create it with
   `python3 -m venv --system-site-packages` so the image's torch satisfies `torch==2.8.*`.

## CUDA / driver / wheels
8. **vLLM 0.26 on a CUDA 12.x driver**: the PyPI wheel needs `libcudart.so.13`. Use the
   `vllm-0.26.0+cu129` release wheel + `torch==2.11.0+cu129` (+audio/vision), a CPU
   `torchcodec==0.14.0+cpu` + `apt-get install ffmpeg`, NVIDIA `cuda-compat-12-9` on drivers
   < 575 (LD_LIBRARY_PATH for the vLLM process only), and `ninja` with the venv `bin` on PATH.
   Driver ≥ 13: plain PyPI. (tess-antra-project/reports/generation_runtime_repair_2026-10-08.md;
   implemented in grader-shifts-persona-space/scripts/pod/install_easysteer.sh.)
9. Self-test in the job's exact env: a real GPU op, `torch.version.cuda`, `from vllm import LLM`,
   `ninja --version` — `is_available()` alone can lie.
10. One venv per CUDA line (`es_venv_cu129` / `_cu130`); never reuse one built on another host (EC#29).

## Env & secrets
11. **`RUNPOD_POD_ID` is not visible inside `rp run` jobs**: read it from `/proc/1/environ`
    (EC#11). Self-stop needs `RUNPOD_API_KEY` in the pod `.env`; preflight must FAIL if
    SHUTDOWN is set but either is missing (otherwise self-stop is a silent no-op).
12. Check every API key with one live call through the job's own client before GPU work.

## Self-stop, ceilings, uploads
13. Every unattended job: `trap on_exit EXIT` → DONE/FAILED marker, upload logs, REST stop
    (`POST /v1/pods/$ID/stop`; `DELETE` to terminate). Terminate only after a **verified**
    upload; downgrade to stop on any failure. Arm a `MAX_HOURS` ceiling at start.
    (grader-shifts-persona-space/scripts/pod/_lib.sh; worked as designed on 2026-10-20.)
14. Upload after every step, verify the remote file list, ship logs with results (EC#27, #31),
    create result datasets public (EC#18), give smoke/follow-up runs their own prefixes (EC#25).
15. Uploads only add files: a re-run's DONE/FAILED marker must replace the old one
    (`delete_patterns`), or a stale FAILED sits next to a new DONE.

## Monitoring
16. Poll the artefact store (STATE.md / DONE / FAILED on HF), not a pod that stops itself (EC#23).
17. macOS: the agent's background monitors stop if the laptop sleeps; run `caffeinate -i` for
    overnight work. Pod-side ceilings still bound cost if the laptop dies.
18. Retries must change something (AssistantAxisWithEmergentMisalignment/exp/STATE.md).

## Candidate `rp` changes
- `rp up --ip-timeout SECS --on-timeout terminate` (auto-delete pods that never get a network).
- GPU fallback across aliases with retry on "no instances available".
- `rp run` exports `RUNPOD_POD_ID` and the job name (`RP_JOB`) into the job script.
- `rp run --self-stop stop|terminate --max-hours H` built in (six repos reimplement it).
- `rp up`/`bootstrap` print `findmnt /workspace` + `df`, warn on FUSE with a small `--disk`.
- `rp bootstrap --venv-dir /venv` (container-disk venv) and a `vllm-0.26-cu129` profile.
