# gaea-plugin-flux2-klein

Black Forest Labs' **FLUX.2 [klein] 4B** (a compact rectified-flow MMDiT with Qwen3-4B conditioning, Apache-2.0) as
a [Gaea](https://github.com/xocialize/gaea) plugin. It is built on [GaeaPluginKit](https://github.com/xocialize/gaea-plugin-kit)
over [flux2-klein-swift](https://github.com/xocialize/flux2-klein-swift), runs in its own process, and speaks the Gaea
Plugin Protocol v1. Add it to Gaea by its repo URL:

```bash
gaea plugins add https://github.com/xocialize/gaea-plugin-flux2-klein
```

## What it offers

| Model | MES operation | Quality tiers | Notes |
|---|---|---|---|
| `klein-4b` | `image.generate` | draft, standard | Distilled to 4 steps without CFG. A negative prompt does nothing and is reported `negative_prompt_ignored`. |
| `klein-4b` | `image.edit` (Gaea's default) | draft, standard | Multi-reference editing: 1–4 images, named in the instruction in order ("the fox from image 1 in the cabin from image 2"). Drawn from noise at the first image's size, so `strength` does nothing (`param_ignored`). |
| `klein-4b-base` | `image.generate`, `image.edit` | best | The undistilled checkpoint: 28 steps with two-pass CFG at guidance 4, and negative prompts on generate. About 12× the distilled model's time. |

- **Sizes.** 256–2048 px a side, at most 1024² in area (the area Klein's footprints were measured at), in multiples
  of 16. Anything else is fitted and reported `size_snapped`. `aspect` with `megapixels` works too.
- **`count`** on `image.generate`: 1–8 images, each with `seed + i`. A request without a seed gets one, reported in
  `effective`. `image.edit` has one output.
- **Options**, as MES extensions: `x-klein.steps` (both models) and `x-klein.guidance` (base only).
- **Not yet offered:** Klein's per-request LoRA adapters, and the package's encoder-evict light tier for 16 GB Macs.

## Lanes and memory

Each model has three lanes, one per DiT quant, with the package's footprints measured at 1024²:

| Lane | Resident | Peak while running (distilled / base) | Gaea schedules (distilled / base) |
|---|---|---|---|
| `bf16` (default) | 14.7 GB | 11.2 / 12.4 GB | 25.9 / 27.1 GB |
| `int8` | 11.5 GB | 11.2 / 12.4 GB | 22.7 / 23.9 GB |
| `int4` | 9.3 GB | 11.2 / 12.4 GB | 20.5 / 21.7 GB |

bf16 is the default, as in mlx-image-runtime's catalog. When it doesn't fit, Gaea loads a smaller lane and warns
(`lane_fallback`, G-23); `gaea models lane klein-4b int4` makes another lane the default. A request naming a lane
(`x-gaea.lane`) never falls back.

## Weights

Gaea sends its models root in `configure` (MLXEngine's store by default, `gaea models root` to change it). The
worker looks for `mlx-community/FLUX.2-klein-4B-bf16` and `mlx-community/FLUX.2-klein-base-4B-bf16` there and
downloads a missing snapshot on `prepare` or `materialize`. The int8 and int4 lanes quantize the bf16 DiT on load.

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
[demo plugin](https://github.com/xocialize/gaea-plugin-demo) for signing, notarization and CI secrets.

`klein-inproc` (`swift run -c release klein-inproc --help` reads its header) is not part of the plugin: it runs the
same lane in-process through MLXServeEngine, the reference that Gaea's runs are compared against byte for byte.

## Licence

The plugin is Apache-2.0 (Gaea G-21). FLUX.2 [klein] 4B and its base are Apache-2.0 (Black Forest Labs), and
flux2-klein-swift is MIT. The 9B Klein variants are under FLUX's non-commercial licence and are not in this plugin.
