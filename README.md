# gaea-plugin-flux2-klein

Black Forest Labs' **FLUX.2 [klein] 4B** (a compact rectified-flow MMDiT with Qwen3-4B conditioning, Apache-2.0) as
a [Gaea](https://github.com/xocialize-code/gaea) plugin. It is built on [GaeaPluginKit](https://github.com/xocialize-code/gaea-plugin-kit)
over [flux2-klein-swift](https://github.com/xocialize/flux2-klein-swift), runs in its own process, and speaks the Gaea
Plugin Protocol v1. Add it to Gaea by its repo URL:

```bash
gaea plugins add https://github.com/xocialize-code/gaea-plugin-flux2-klein
```

## What it offers

| Model | MES operation | Quality tiers | Notes |
|---|---|---|---|
| `klein-4b` | `image.generate` | draft, standard | Distilled to 4 steps without CFG. A negative prompt does nothing and is reported `negative_prompt_ignored`. |
| `klein-4b` | `image.edit` (Gaea's default) | draft, standard | Multi-reference editing: 1–4 images, named in the instruction in order ("the fox from image 1 in the cabin from image 2"). Drawn from noise at 1024² unless a size is asked (as pixel-agency-image-engine v0.9.1 draws it), and returned at the first image's size; `strength` does nothing (`param_ignored`). |
| `klein-4b-base` | `image.generate`, `image.edit` | best | The undistilled checkpoint: 28 steps with two-pass CFG at guidance 4, and negative prompts on generate. About 12× the distilled model's time. |
| `klein-4b-rdm` (0.4.0) | `image.generate` | draft | Klein 4B's DiT distilled to **one step** by EPFL VITA ([flux2-klein-1step-rdm](https://huggingface.co/epfl-vita/flux2-klein-1step-rdm), Apache-2.0), on Klein's own encoder and VAE. 0.43× the 4-step model's time at 1024² (2.8 s against 6.5 s on an M5 Max) at the same SigLIP2 quality and keyword recall in TierBench (AB-R-1056); spells in-image text a little less reliably. Trained at 512². No negative prompts, no adapters, no edit. |
| `klein-4b-bonsai` (0.5.0) | `image.generate` (named-only) | draft | Prism ML's [Bonsai Image Ternary 4B](https://huggingface.co/prism-ml/bonsai-image-ternary-4B-mlx-2bit) (Apache-2.0): Klein 4B with the transformer retrained to **ternary weights**. A 3.9 GB download and about 8 GB of memory at 1024², against 26 GB for `klein-4b`. Best at 1024²-class sizes: 10 of 12 test prompts usable there (Klein 4B int4 10, bf16 11), 8 of 12 at 512² (AB-R-1130). It draws a different picture from `klein-4b` at the same seed. Nothing picks it by default: a request names it. No negative prompts, no adapters, no edit. |

- **Sizes (0.3.0).** 256–4096 px a side, at most 4096² in area. Up to 1024² Klein draws the image as it is.
  - **Past 1024²,** it draws a quarter of the area (up to 2048²) or a sixteenth (up to 4096²), then
    [FlowUpscaler](https://huggingface.co/mlx-community/FlowUpscaler-bf16) (TensorForger, Unlicense) doubles the
    latents once or twice before the decode. The result is the picture the smaller draw made, with finer detail, and
    effective reports it as `x-klein.upscaled` ({from, passes}).
  - **Speed:** Klein 4B makes 2048² in about 12 s and 4096² in about 32 s, against about 7 s for 1024² (M5 Max, bf16).
  - **Grid:** sides are multiples of 16 (32 with one pass, 64 with two). The drawn image is 256–2048 px a side.
    Anything else is fitted and reported `size_snapped`.
  - **Edits** are upscaled only when they ask for a size.
  - **Memory:** the package declares how its memory grows with output pixels, and Gaea 1.2.0 charges each job on that
    line, so 4096² (about 41 GB with bf16) is admitted only where it fits.
  - `aspect` with `megapixels` works too.
- **`count`** on `image.generate`: 1–8 images, each with `seed + i`. A request without a seed gets one, reported in
  `effective`. `image.edit` has one output.
- **Options**, as MES extensions: `x-klein.steps` (both models) and `x-klein.guidance` (base only).
- **Adapters (0.2.0).** Both models take up to 4 LoRAs at strength 0–2 in the bfl-fused and diffusers-peft dialects
  (`KleinAdapters.support`, base model `flux2-klein-4b`): plain ones on generate, plain and ic on edit. Both edit offers
  ship the 7 FLUX.2 Klein 4B adapters pixel-agency-image-engine v0.9.1 registers, from the same registry
  (`Sources/KleinPlugin/Resources/flux2-klein-4b.json`, verbatim from mlx-image-runtime 0.15.0 = flux2-klein-swift
  `registry/flux2-klein-4b.json` @74d4b23): ids `flux2-klein:<id>`, the same pinned revisions and sha256s, all
  Apache-2.0 and verified. `refcontrol-pose` (pose, reference) and `tryon` (person, top, bottom) are ic: their images
  arrive with the adapter and lead the edit's references in slot order, so such an edit may have no images of its own. The worker fetches each adapter into the models root (`adapters/<sha256>/`)
  on first use; Gaea applies triggers and prompt templates before the run. Requires Gaea 0.7.0.
- **Not yet offered:** the package's encoder-evict light tier for 16 GB Macs.

## Lanes and memory

Each Klein model has three lanes, one per DiT quant, with the package's declared footprints at 1024²:

| Lane | Resident | Peak while running (distilled / base) | Gaea schedules (distilled / base) |
|---|---|---|---|
| `bf16` (default) | 14.7 GB | 11.2 / 12.4 GB | 25.9 / 27.1 GB |
| `int8` | 11.5 GB | 11.2 / 12.4 GB | 22.7 / 23.9 GB |
| `int4` | 9.3 GB | 11.2 / 12.4 GB | 20.5 / 21.7 GB |

bf16 is the default, as in mlx-image-runtime's catalog. When it doesn't fit, Gaea loads a smaller lane and warns
(`lane_fallback`, G-23); `gaea models lane klein-4b int4` makes another lane the default. A request naming a lane
(`x-gaea.lane`) never falls back.

`klein-4b-bonsai` has one lane, `ternary`: 3.6 GB resident and 4.4 GB while running, so Gaea schedules 8.0 GB at up to
1024² (12.6 GB at 2048², 31.1 GB at 4096²). Through the signed bundle the worker peaked at 7.9 GB at 1024² and
12.0 GB at 2048² (M5 Max, AB-R-1138 and this release's check).

Since 0.5.0 the Klein models decode in row bands at every size (flux2-klein-swift 0.11.1), which takes about 8 GB off
their real peak at 1024². Their declared footprints above are unchanged, so Gaea reserves more than they now use.
An image differs from 0.4.0's at the same seed by at most 5 of 255 levels.

## Weights

Gaea sends its models root in `configure` (MLXEngine's store by default, `gaea models root` to change it). The
worker looks for `mlx-community/FLUX.2-klein-4B-bf16`, `mlx-community/FLUX.2-klein-base-4B-bf16` and
`mlx-community/FLUX.2-klein-4B-1step-RDM-bf16` (RDM's DiT in diffusers' layout beside Klein 4B's encoder and VAE) there
and downloads a missing snapshot on `prepare` or `materialize`. The int8 and int4 lanes quantize the bf16 DiT on load.
`klein-4b-bonsai` reads `prism-ml/bonsai-image-ternary-4B-mlx-2bit` as Prism ML publishes it (a 2-bit DiT and a 4-bit
Qwen3-4B, already packed), pinned at revision `2c24c81b`.

## Build and check locally

```bash
scripts/build-plugin.sh
```

```bash
scripts/check-plugin.sh
```

`check-plugin.sh` generates one 512² image with Klein 4B int4 against the weights in MLXEngine's store
(`CHECK_MODELS_ROOT` moves it), with every `.build` folder hidden so MLX can find its metallib only inside the
bundle. A control then shows that the bare worker, outside the bundle, can't. `scripts/release.sh` builds, checks,
notarizes with the fleet's `mlx-notary` profile, and publishes the GitHub release. See the
[demo plugin](https://github.com/xocialize-code/gaea-plugin-demo) for signing, notarization and CI secrets.

`klein-inproc` (`swift run -c release klein-inproc --help` reads its header) is not part of the plugin: it runs the
same lane in-process through MLXServeEngine, the reference that Gaea's runs are compared against byte for byte
(`--adapter <id>=<strength>` and `--adapter-input <id>:<role>=<png>` apply shipped adapters).

```bash
scripts/golden.sh <out-dir>
```

`golden.sh` first runs the sizes: 1024², 2048², 4096², 1536×1024, Base at 1792×1024, and an edit at 2048². Each is
byte for byte against `klein-inproc`, which plans sizes the same way, at the asked size, with the upscale reported.
Then it runs a seeded Klein 4B edit with one ic adapter (`flux2-klein:refcontrol-pose`, a skeleton and a subject
from `scripts/fixtures`) and one plain one (`flux2-klein:background-remove`) over GPP on the signed bundle
(`gaea-plugin-check --adapter`, which builds `run.adapters` from the manifest as Gaea does) and through `klein-inproc`
with the same `AdapterRef`s, and needs them byte for byte the same. A no-adapter control must differ.
`fixtures/pose-512.png` is a drawn OpenPose-style skeleton; `fixtures/fox-256.png` comes from gaea-plugin-forge.

## Licence

The plugin is Apache-2.0 (Gaea G-21). FLUX.2 [klein] 4B and its base are Apache-2.0 (Black Forest Labs), the 1-step
RDM DiT is Apache-2.0 (EPFL VITA, a fine-tune of Klein 4B), Bonsai Image Ternary 4B is Apache-2.0 (Prism ML, built from
Klein 4B and Qwen3-4B; Prism ML asks for the credit "Created using Bonsai Image by Prism ML", which the model's notes
carry), and flux2-klein-swift is MIT. The 9B Klein variants are under FLUX's non-commercial licence and are not in this plugin.
