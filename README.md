# export-sd-graph-ops

## Overview

`export-graph-ops` is a small utility ported from the corresponding tool in `llama.cpp` to `stable-diffusion.cpp`.

Its purpose is to export the unique GGML/GGUF computation graph operators (OPs) used by a Stable Diffusion model or related components. The exported result can then be used together with the operator test utility from `llama.cpp` (for example, `test-backend-op`) to perform backend OP validation and compatibility testing.

In practice, this tool is useful when you want to:

- inspect which operators are actually used by a model graph,
- generate operator coverage data for backend testing,
- verify whether a backend implementation supports the operators required by a specific diffusion, text encoder, or VAE stage.

## Supported model families and export stages

This tool exports unique GGML ops from stable-diffusion.cpp model graphs.

### Supported stages

The exporter currently supports the following stages:

- `diffusion`
- `text`  
  Aliases: `text-encoder`, `text_encoder`
- `vae`
- `all`

`all` runs all supported stages for the given model file.

### Diffusion stage support

The `diffusion` stage is currently supported for these model families:

- **SD1 / SD2 / SDXL** (UNet)
- **SD3** (MMDiT)
- **Flux / Flux2**
- **Qwen Image**
- **Anima**
- **Z-Image**
- **Wan**

Any model version outside the above branches is currently unsupported for diffusion export.

### Text stage support

The `text` stage is intentionally limited to model families that already have exportable GGML text-encoder graphs in stable-diffusion.cpp.

#### Supported

- **SD1 / SD2**  
  Exports the CLIP text encoder.
- **SDXL**  
  Exports `clip_l`, and `clip_g` when present.
- **SD3**  
  Exports any supported combination of:
  - `clip_l`
  - `clip_g`
  - `t5xxl`
- **Flux**
  - Standard Flux models: exports `clip_l` and/or `t5xxl` when present.
  - Chroma-style Flux variants: exports `t5xxl` and `t5xxl.masked`.
- **Wan**  
  Exports masked **UMT5** (`umt5.masked`).

#### Not supported / intentionally skipped

These model families are intentionally skipped for text export because they use a pure LLM text path, or an LLM-dominant conditioner path, rather than the GGML text-encoder graphs this tool targets:

- **Flux2**
- **Qwen Image**
- **Anima**
- **Z-Image**
- **OVIS Image** (Flux-family variant)

### VAE stage support

The `vae` stage exports VAE graphs when supported VAE tensors are present in the model file.

#### Supported VAE sources

The exporter checks both of these namespaces:

- `first_stage_model`
- `tae`

#### Exported VAE implementations

- If `first_stage_model` exists:
  - **Wan / Qwen Image / Anima** use **WanVAE**
  - other supported families use **AutoEncoderKL**
- If `tae` exists:
  - **Wan / Qwen Image / Anima** use **TinyVideoAutoEncoder**
  - other supported families use **TinyImageAutoEncoder**

Both encode and decode paths are exported.

#### Not supported

- **Chroma Radiance** VAE export is not supported because it uses **FakeVAE**, which has no GGML graph to export.
- If neither `first_stage_model` nor `tae` contains supported tensors, the VAE stage is skipped or fails, depending on the selected stage mode.

### Behavior of `--stage all`

When `--stage all` is used:

- diffusion export always runs
- text / VAE export may be skipped if unsupported for that model family
- unsupported text / VAE stages do **not** necessarily fail the whole run

When `--stage text` or `--stage vae` is used directly, unsupported models are treated as errors.

### GGML ops intentionally not exported

The exporter skips these non-compute or layout/view-style nodes:

- `GGML_OP_NONE`
- `GGML_OP_VIEW`
- `GGML_OP_RESHAPE`
- `GGML_OP_PERMUTE`
- `GGML_OP_TRANSPOSE`

So the output is **not** a dump of every graph node; it is a deduplicated set of compute-relevant ops.

## Installation

1. Clone `stable-diffusion.cpp` recursively:

   ```bash
   git clone --recursive https://github.com/leejet/stable-diffusion.cpp
   ```

2. Enter the `examples` directory:

   ```bash
   cd stable-diffusion.cpp/examples
   ```

3. Clone this repository into `export-graph-ops`:

   ```bash
   git clone https://github.com/eastmoe/export-sd-graph-ops export-graph-ops
   ```

4. Add the following line to the end of `stable-diffusion.cpp/examples/CMakeLists.txt`:

   ```cmake
   add_subdirectory(export-graph-ops)
   ```

5. Return to the root directory of `stable-diffusion.cpp` and build the target:

   ```bash
   cmake --build build --target export-graph-ops 2>&1
   ```

## Usage

```bash
stable-diffusion.cpp/build/bin/export-graph-ops -m <model_path> [-o output_file] [--stage diffusion|text|vae|all]
```

### Arguments

- `-m <model_path>`  
  Path to the model file to analyze.

- `-o output_file`  
  Optional path for the exported OP description file.

- `--stage diffusion|text|vae|all`  
  Select which stage to export:
  - `diffusion`: export operators from the diffusion model graph
  - `text`: export operators from the text encoder graph
  - `vae`: export operators from the VAE graph
  - `all`: export operators from all supported stages

## Notes

- The tool is intended for graph/operator export, not for image generation.
- The exported output is designed to support OP-level testing workflows.
- A typical use case is generating operator samples from `stable-diffusion.cpp` and then testing those operators with the backend test utility from `llama.cpp`.
