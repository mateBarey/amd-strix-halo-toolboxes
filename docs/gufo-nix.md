# Gufo Nix toolbox

This additive image builds Gufo and its matching tools with a single pinned Nix ROCm stack. It provides a separate `gufo-rocm-nix` Toolbx container for Qwen3.8-27B, including the F16-to-BF16 vision-projector converter, a host launcher for Hermes, live-request profiling, and the optional upstream hipBLASLt tuner.

Your existing `gufo-rocm` container and the stock ROCm 10.0 image remain available. Build and use this image under its own name; there is no need to install another ROCm into the old container.

## What is pinned and included

| Component | This image |
| --- | --- |
| Gufo | [`7c8196527ad02ac4c585a32b1131fab7e70509d0`](https://github.com/gufo-org/gufo/tree/7c8196527ad02ac4c585a32b1131fab7e70509d0) |
| Nix | `2.34.9` |
| ROCm | `7.2.3`, from Gufo's locked Nix definition, including its patched ROCr |
| Container base | Fedora Toolbx 44 |
| Tools | `gufo`, `gufo-kernel-bench`, `tune_hipblaslt`, `gufo-convert-projector`, `gufo-profile`, `gufo-tune-hipblaslt` |
| Prebuilt developer-shell profile | `/opt/gufo-shell` |
| Developer flake | `/opt/gufo-toolbox/flake`, with the same source and dependency lock |
| Official development environment | ROCm PyTorch, Transformers, download/evaluation tools, rocprofiler, debuggers and the upstream compiler environment |
| Optional benchmark packages | `ds4-reference`, `llama-cpp-reference`, `llama-cpp-mtp-reference` |

The [derived Nix package](../toolboxes/gufo-nix/flake.nix) enables Gufo's tools and wraps the native executables with their matching runtime libraries. It excludes `/opt/rocm` from CMake discovery during the build. This stack is separate from the repository's `rocm-10.0-gufo` image, which tracks a different Gufo/ROCm combination. Avoid importing that image's library paths into this one.

Nix uses a container-owned `/nix/store` and a root daemon. The runtime starts the daemon without systemd and lets the Toolbx user issue normal Nix commands. It does not install Nix into the shared host home or modify the host's shell startup files or Nix profile. Gufo is already built; entering the container and launching it does not require a new build.

## Prerequisites

Use a Linux Strix Halo host with working AMD GPU device access, Podman and Toolbx. Follow the project's [host setup guide](https://strix-halo-toolboxes.com/) for host drivers and memory allocation. This image does not change BIOS settings, kernel parameters, power profiles, or host packages.

For the Hermes launcher, `toolbox`, `curl`, `python3`, and `hermes` must be available **on the host**. Keep the model and projector under paths visible inside Toolbx; the examples use the shared home directory. Building the Nix/ROCm closure needs substantial disk space, network access, and time. A small GitHub-hosted runner may run out of disk or memory; local builds use the same recipe.

## Repair the development environment in an existing container

Use this route when the image and `gufo-rocm-nix` container already exist. Run on the host from the existing checkout:

```bash
git -C "$HOME/VMl/gufo-nix-mateBarey-github" pull --ff-only
bash "$HOME/VMl/gufo-nix-mateBarey-github/scripts/gufo-nix/develop.sh"
```

The helper uses the container's installed Nix and the updated checkout's locked flake. Before acquiring dependencies it checks that the checkout's Gufo store output is already present and equals the installed `/opt/gufo` output. A mismatch stops the helper. The production Gufo derivation is unchanged by this repair, so the completed Gufo build is reused.

The earlier image's custom minimal shell omitted upstream development tools. Restoring the official shell requires acquiring those missing dependencies, including its ROCm PyTorch environment. Nix may download or build development dependencies; this is not an image rebuild. The helper retains its development profile under `$HOME/.cache/gufo/nix-devshell/gufo-rocm-nix` and uses the existing container-owned store. It also fetches and retains the exact interactive Bash outputs, with source builds disabled for that Bash step, to avoid Nix's silent fallback after an unsuccessful shell realization.

`GUFO_TOOLBOX_NAME` selects a different existing container. `GUFO_BUILD_CORES` and `GUFO_NIX_MAX_JOBS` bound dependency build parallelism. A command can be run without opening an interactive shell:

```bash
bash scripts/gufo-nix/develop.sh --command python -c \
  'import torch, transformers; print(torch.__version__, transformers.__version__)'
```

Inside the environment, `GUFO_SOURCE_ROOT` points to the exact pinned Gufo source, including its profiling and reference scripts. The inherited ROCr library paths, `TORCH_HOME`, compiler inputs and ccache hook are preserved. This repairs the development environment; a model startup error remains a separate runtime issue.

## Route A: build locally

### Complete setup from this fork

For the `/home/esckobear/VMl` model files and Hermes project used by this fork, run this on the Fedora host as your normal user:

```bash
bash scripts/gufo-nix/setup-qwen27b.sh
```

The setup uses the source in the current repository checkout. It builds and smoke-tests the image, selects an available `gufo-rocm-nix` container name, checks or converts the exact 27B projector, chooses an available API port, saves the selected settings, and starts Gufo and Hermes. It accepts the work directory left by the earlier standalone installer; it does not reset or overwrite that installer's old source checkout.

Subsequent launches use:

```bash
bash "$HOME/VMl/gufo-nix-mateBarey/start-qwen27b.sh"
```

Use `bash scripts/gufo-nix/setup-qwen27b.sh --help` for model, projector, project, context and build-resource overrides. Nix provides the dependencies and included tools; the setup does not automatically tune Q4 or vision kernels.

### Build the image only

From the repository root:

```bash
bash scripts/gufo-nix/build-local.sh
```

The helper builds the image and runs its CPU-only image smoke checks. The default local image name is `localhost/gufo-nix:latest`. You can pass a different image name as its only positional argument. `CONTAINER_ENGINE=docker` selects Docker instead of Podman; `GUFO_BUILD_CORES` (default `4`) and `GUFO_NIX_MAX_JOBS` (default `1`) bound Nix build parallelism.

The Dockerfile is [`toolboxes/Dockerfile.gufo-nix`](../toolboxes/Dockerfile.gufo-nix); its build context is the **repository root**, since it copies the flake and helpers from several directories. To build directly and then run the same smoke checks:

```bash
podman build --tag localhost/gufo-nix:latest \
  --file toolboxes/Dockerfile.gufo-nix .
bash scripts/gufo-nix/smoke-image.sh localhost/gufo-nix:latest
```

Create a new container:

```bash
toolbox create -c gufo-rocm-nix --image localhost/gufo-nix:latest
toolbox enter -c gufo-rocm-nix
```

Keep the existing `gufo-rocm` container. Toolbx shares your home with the new container, so conversion outputs and profile reports saved there remain available from the host.

## Route B: build and publish from your fork

Push these changes to your fork and enable GitHub Actions there. The dedicated workflow is [**Gufo Nix toolbox**](../.github/workflows/gufo-nix.yml). Original upstream polling/publication workflows are guarded against running on forks; use this dedicated workflow for the Nix image.

The workflow first runs CPU tests and builds the actual Fedora/Nix bootstrap using Podman's OCI format, including an authenticated TLS connection to the Nix binary cache. It evaluates the real flake to check preservation of the official development inputs, environment, hooks and baseline outputs, then verifies a small shell remains usable offline after garbage collection. These checks avoid compiling Gufo or ROCm. On a relevant default-branch push, it also builds the full image, runs CPU-only image smoke checks, and publishes to **your fork's** GHCR package. Pull requests and feature-branch pushes run CPU and bootstrap checks without publishing. A manual run builds and smoke-tests the full image; select the `publish` input to publish that manual run.

Published tags are:

- Channel: `ghcr.io/owner/repository:gufo-nix`
- Per-build tag: `ghcr.io/owner/repository:gufo-nix-<git-sha>-<run-id>-<run-attempt>`

Use lowercase owner/repository names and substitute your fork's actual values:

```bash
podman pull ghcr.io/owner/repository:gufo-nix
toolbox create -c gufo-rocm-nix \
  --image ghcr.io/owner/repository:gufo-nix
```

If your GHCR package is private, authenticate to GHCR before pulling or change the package visibility. Check the Actions summary for the exact published tag. A successful workflow establishes build and CPU-smoke results; GitHub's ordinary hosted runner does not establish Strix Halo GPU performance or model compatibility at runtime.

## Check the installed stack and GPU access

Run these from the host after creating the container:

```bash
toolbox run -c gufo-rocm-nix gufo --version
toolbox run -c gufo-rocm-nix nix --version
toolbox run -c gufo-rocm-nix gufo-convert-projector --help
toolbox run -c gufo-rocm-nix gufo-profile --help
toolbox run -c gufo-rocm-nix gufo-tune-hipblaslt --help
toolbox run -c gufo-rocm-nix ls -l /dev/kfd /dev/dri
toolbox run -c gufo-rocm-nix gufo diagnose --fingerprint --json
```

Check that the Gufo version reports the pinned revision and that the fingerprint identifies the expected GPU/runtime. Missing GPU devices or denied device access is a host/container-access issue to resolve before trying to load the model. Do not use a successful `--help` invocation as evidence that GPU inference works.

New images also have an optional developer shell, prebuilt with Gufo's complete official development inputs and the matching tools. From inside the container:

```bash
nix develop /opt/gufo-shell
```

This is optional for inference and for the supplied profiling/tuning helpers. To evaluate the image's source flake instead, use `nix develop path:/opt/gufo-toolbox/flake`; fetching flake metadata may require network access. The shell inherits the pinned upstream environment, including ROCm PyTorch, Transformers, model-download and evaluation libraries, rocprofiler, debuggers and compiler tools. An already-created container retains its old `/opt/gufo-shell`; use the host `develop.sh` helper above to use the corrected environment in that container.

The three benchmark baselines remain separate optional packages, as upstream specifies. For example, inside the container using the updated shared checkout:

```bash
nix shell "path:$HOME/VMl/gufo-nix-mateBarey-github/toolboxes/gufo-nix#llama-cpp-reference"
```

Replace the package suffix with `ds4-reference` or `llama-cpp-mtp-reference` for the other baselines. Selecting one can acquire additional dependencies. Nix supplies the pinned build and development environments; it does not automatically benchmark a GGUF and choose the fastest settings.

The image retains the matching interactive Bash and pins its system flake registry to the included Nixpkgs source. This prevents entering the prebuilt profile from pulling a moving Nixpkgs channel to obtain a different shell. The CPU image smoke checks this profile with networking disabled.

## Prepare the Qwen3.8-27B projector

Use the main model unchanged:

```text
/home/esckobear/VMl/Qwen3.8-27B-Uncensored-Q4_K_M.gguf
```

Inspect the projector's tensor types rather than inferring them from `f16` in its filename. F16 and BF16 are different number formats despite both using 16 bits. Gufo's pinned vision loader requires BF16 matrix tensors and F32 auxiliary tensors; it does not automatically cast F16 matrices. A projector that already contains the required BF16/F32 tensors needs validation only. The reported local conversion result, `0 F16 matrices`, `110 BF16` and `224 F32` tensors, means that particular input was already in the required numeric formats.

Run the installed converter, whose arguments are positional:

```bash
toolbox run -c gufo-rocm-nix gufo-convert-projector \
  /home/esckobear/VMl/mmproj-Qwen3.8-27B-Uncensored-f16.gguf \
  /home/esckobear/VMl/mmproj-Qwen3.8-27B-Uncensored-BF16.gguf

toolbox run -c gufo-rocm-nix gufo-convert-projector \
  /home/esckobear/VMl/mmproj-Qwen3.8-27B-Uncensored-BF16.gguf \
  --check-only
```

The converter validates the exact Qwen3.8-27B vision layout: 334 tensor names/shapes, 5120 output width, BF16 matrix roles, and F32 auxiliary roles. It numerically rounds any F16 matrices to BF16, preserves existing BF16 data and the F32 tensors, verifies the completed output, and refuses to overwrite an existing destination. If the validated BF16 output already exists, use only the second, `--check-only` command above. The original projector and Q4 model remain unchanged. Conversion cannot recover precision already lost when an original model was stored as F16.

This is a converter for the supported 27B projector contract, not a general GGUF compatibility tool. It rejects a different vision architecture, tensor layout, or projection width. In particular, do not substitute the Flash Next projector or its MTP sidecar for the dense 27B model. A successful file check must still be followed by a real image request on the GPU.

## Start Gufo and Hermes

Exit the toolbox and run the launcher **on the host**, from this repository:

```bash
bash scripts/gufo-nix/launch-hermes-qwen27b.sh
```

The launcher defaults to:

| Setting | Value |
| --- | --- |
| Container | `gufo-rocm-nix` |
| Installed executable | `/usr/local/bin/gufo` inside the container |
| Model/projector | The Q4_K_M and converted BF16 paths above |
| Served API model ID | `qwen38-27b-uncensored` |
| Context | `262144` tokens |
| Sessions | `1` |
| RAM prompt-cache budget | `4294967296` bytes, or 4 GiB |
| Server binding | `0.0.0.0:8080` |
| Hermes endpoint | `http://127.0.0.1:8080/v1` |
| Speculation / thinking | `off` / `off` |
| Server log | `/tmp/gufo-qwen27b-server.log` on the host |
| Hermes project directory | `/home/esckobear/Downloads/Downloads_raw/micro_never_ends/Kimi_Agent_Quaternion Sphere 16 Centroids` |

The actual number of context tokens changes with the request. In the pinned dense Qwen implementation, the configured maximum also determines an eager KV reservation and the startup resource check. The default remains `262144`. That implementation's allocation policy and the current HIP memory budget are distinct from the machine's physical RAM total or its measured throughput. The 4 GiB RAM prompt-cache budget is a separate setting. The server binds all interfaces as requested, so host network/firewall policy determines which other machines can reach it.

The launcher verifies that `/v1/models` advertises the expected alias before opening Hermes. It configures the custom endpoint and top-level `model_overrides.custom.qwen38-27b-uncensored` entries for tool support, vision support, and context. These declarations let Hermes use the endpoint; they do not independently verify model behavior. A different model already on the port is an error, and the launcher does not kill it.

You can override paths and the project without editing the script:

```bash
GUFO_PROJECT_ROOT="$HOME/my-project" \
GUFO_MODEL_PATH="$HOME/models/Qwen3.8-27B-Uncensored-Q4_K_M.gguf" \
GUFO_VISION_PATH="$HOME/models/mmproj-Qwen3.8-27B-Uncensored-BF16.gguf" \
  bash scripts/gufo-nix/launch-hermes-qwen27b.sh
```

Run `bash scripts/gufo-nix/launch-hermes-qwen27b.sh --help` for the remaining overrides. When an existing server advertises the same alias, the launcher reuses it; it does not establish that all its paths, flags, or environment match the new request. To change context, model files, or a tuning cache, stop the specific existing server yourself before relaunching. The launcher waits up to 300 seconds by default and shows a log tail on failure. After a timeout, inspect the log and process before starting another copy; it might still be loading.

Newly generated saved launchers also honor caller-provided `GUFO_CONTEXT`, `GUFO_CACHE_RAM_BYTES`, `GUFO_STARTUP_TIMEOUT` and `GUFO_TUNE_CACHE`, applying saved defaults only when a variable is unset. Older saved launchers keep their original assignments. Use the repository launcher directly to supply overrides without rerunning setup, for example `GUFO_PORT=8081 bash scripts/gufo-nix/launch-hermes-qwen27b.sh` for the previously selected port.

Dense Qwen3.8-27B runs here with `--speculative off`. The Flash Next `--speculative mtp` setup is for another architecture and is not part of this launcher.

## Measure real text and image requests

Keep Gufo running, then use a second host terminal. For a text baseline:

```bash
toolbox run -c gufo-rocm-nix gufo-profile \
  --base-url http://127.0.0.1:8080/v1 \
  --model qwen38-27b-uncensored \
  --repetitions 3 --max-tokens 128 \
  --output "$HOME/gufo-qwen27b-text-baseline.json"
```

For text plus actual vision requests, supply your own PNG or JPEG:

```bash
toolbox run -c gufo-rocm-nix gufo-profile \
  --base-url http://127.0.0.1:8080/v1 \
  --model qwen38-27b-uncensored \
  --image "$HOME/Pictures/test-image.png" \
  --repetitions 3 --max-tokens 128 \
  --output "$HOME/gufo-qwen27b-text-image-baseline.json"
```

Choose a new output path each time: existing reports are never overwritten. Images must be nonempty PNG/JPEG files no larger than 15 MiB. The helper sends the actual image bytes as a data URL. It requires explicit `--allow-remote` for a non-loopback endpoint and does not follow HTTP redirects.

For each workload it measures three requests with `cache_prompt: false`, then an unmeasured warmup followed by three separately labeled `cache_prompt: true` repeats. The JSON records client time to the first text delta, total response time, completion hashes, server usage, model IDs, timestamps, and any response fingerprint. Image TTFT includes image processing and text prefill; it is not an isolated vision-encoder measurement. Bypassing prompt reuse does not clear OS/GPU caches or reload weights.

This is a **baseline measurement**, with `recommendation.status: baseline_only`. It changes no server settings and does not claim to find a faster configuration. Run it without competing inference if you want comparable results. HTTP errors, malformed/incomplete streams, and missing usage are reported as failures rather than successful measurements.

## Optional: synthetic BF16 hipBLASLt plans

The image includes Gufo's matching upstream `tune_hipblaslt` binary and a helper that records its version, GPU fingerprint, command, and results. These are synthetic BF16 GEMM measurements. The Q4 model's native quantized kernels bypass hipBLASLt, and this Gufo vision implementation explicitly ignores the plan-cache environment variable. A generated cache therefore does **not** mean the Q4 model or vision encoder was tuned, and no end-to-end speedup is promised.

Stop your inference workload first for meaningful tuner timings. The helper never stops processes for you. A quick run is:

```bash
toolbox run -c gufo-rocm-nix gufo-tune-hipblaslt --quick
```

For the upstream default shape set with selected batches:

```bash
toolbox run -c gufo-rocm-nix gufo-tune-hipblaslt \
  --batch 128,256 --repetitions 20 \
  --outdir "$HOME/.cache/gufo/hipblaslt"
```

Every run gets a fresh child directory with `plans.bin`, `runtime.json`, and `tuner.log`; a failure does not publish a partial cache. `--shape MxK` is repeatable and means output width by input width. The first custom shape replaces the upstream shape set. Use `--help` before selecting custom shapes; they are developer controls and do not inspect or tune a GGUF file.

To opt into a completed cache for eligible Gufo text paths, use the exact path printed by the helper, then start a fresh server:

```bash
GUFO_TUNE_CACHE="$HOME/.cache/gufo/hipblaslt/run-REPLACE_WITH_ACTUAL_RUN/plans.bin" \
  bash scripts/gufo-nix/launch-hermes-qwen27b.sh
```

The launcher sets `GUFO_HIPBLASLT_PLAN_CACHE` **inside the container** only for that explicit readable, nonempty file. Without it, Gufo uses its defaults. Compare actual baseline reports before deciding to retain a cache. Generate plans on the GPU/runtime that will use them and regenerate after relevant stack changes; runtime version and fingerprint checks can reject incompatible plans.

## Validation and troubleshooting

The earlier standalone installer's image recipe had two defects that are corrected here:

- Fedora 44 removed the compatibility CA path. All builder, runtime and Nix daemon certificate settings now use `/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem`, the bundle named by the [Fedora change documentation](https://fedoraproject.org/wiki/Changes/droppingOfCertPemFile). The recipe checks that this bundle exists and is nonempty before network use.
- Podman's OCI image format ignored the recipe's Docker `SHELL` instruction. Every multi-command build step now invokes a copied script with `/bin/bash -euo pipefail`, so a failed Nix build stops the image build instead of continuing into garbage collection or producing Gufo symlinks with missing targets. Keeping these commands in ordinary scripts also supports older Podman versions that cannot parse Dockerfile heredocs.

The repository includes regression checks for the certificate settings and executes safe failure probes through all three actual build-script commands. It also exercises Bash-output retention and failure propagation, rejected offline shell fallbacks, saved launcher overrides, and the existing-container development helper. CI additionally builds the real Fedora/Nix bootstrap in Podman OCI format and checks the flake/shell behavior with actual Nix. A successfully built bootstrap is not a full Gufo image or GPU inference result.

Repository CPU tests can be run without a GPU:

```bash
python3 -m pip install 'gguf==0.19.0'
python3 -m unittest discover -s tests/gufo_nix -v
```

These cover projector validation/conversion, launcher control flow, actual loopback HTTP/SSE profiling protocol, and the tuner wrapper with stub executables. They do not execute HIP kernels. Image smoke checks exercise the built image without a GPU, including a fresh daemon startup through the non-root Nix wrapper and a tiny offline build under a dedicated build user. Actual Strix Halo validation still requires the GPU fingerprint, model loading, a text completion, an image completion, and measured workload results on your host. Build/check success alone is not a claim of speed or 256K-context capacity.

| Symptom | Check |
| --- | --- |
| Launcher says to run on host | Exit Toolbx and run the repository's launcher on the host. |
| Port already serves a different model | Inspect that process; stop it deliberately or set `GUFO_PORT` to a free port. |
| Projector rejected | Run `gufo-convert-projector BF16_FILE --check-only`; verify the 27B sidecar and converted output path. |
| No device / permission error | Check `/dev/kfd`, `/dev/dri`, host driver health, and container device/group access. |
| Wrong libraries or `/opt/rocm` errors | Use the installed wrapped executable. Remove manually injected `LD_LIBRARY_PATH`/`LD_PRELOAD`; do not source a different ROCm stack. |
| Startup timeout | Read `/tmp/gufo-qwen27b-server.log` and check whether the process is still loading before retrying. |
| Tuner cache made no measurable difference | Only eligible hipBLASLt routes consume it; Q4 and vision paths are outside its scope. Retain Gufo defaults if the actual workload does not improve. |

## Updating and rollback

Rebuilding or pulling a newer image does not replace an existing Toolbx container's filesystem. Test updates under a new container name first and set `GUFO_TOOLBOX_NAME` when launching it. Keep a known-good per-build GHCR tag when reproducibility matters.

To return to the old setup, stop the specific Nix Gufo server and launch your existing `gufo-rocm` workflow. Nothing in this setup removes or upgrades that container. The original model and F16 projector remain available; the BF16 output and profile reports are separate files. Hermes keeps the endpoint/settings written by the new launcher, so rerun your previous Hermes configuration or launcher when switching back. Do not remove a container as a substitute for inspecting which process owns the server port.
