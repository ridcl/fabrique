"""Session-wide setup, run by pytest before any test module imports JAX."""

import os

# Pallas/Triton kernels need some GPU memory *outside* XLA's preallocated pool
# when they launch; with the devcontainer's 0.95 they fail with "Failed to
# launch CUDA kernel ... CUDA_ERROR_OUT_OF_MEMORY".  0.90 is measured to work.
# Must run before JAX initialises a backend.
if float(os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.75")) > 0.90:
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"
