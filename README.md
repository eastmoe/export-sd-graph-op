# export-sd-graph-ops

## Overview

`export-graph-ops` is a small utility ported from the corresponding tool in `llama.cpp` to `stable-diffusion.cpp`.

Its purpose is to export the unique GGML/GGUF computation graph operators (OPs) used by a Stable Diffusion model or related components. The exported result can then be used together with the operator test utility from `llama.cpp` (for example, `test-backend-op`) to perform backend OP validation and compatibility testing.

In practice, this tool is useful when you want to:

- inspect which operators are actually used by a model graph,
- generate operator coverage data for backend testing,
- verify whether a backend implementation supports the operators required by a specific diffusion, text encoder, or VAE stage.

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

## License
