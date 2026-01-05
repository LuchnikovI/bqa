## What is it?

This is a package for large scale tensor-networks-based simulation of quantum annealing.
It uses belief propagation based approximate inference (see https://arxiv.org/abs/2306.17837, https://arxiv.org/abs/2409.12240, https://arxiv.org/abs/2306.14887) as an engine. This implementation introduces a compilation step that classifies graph nodes by degree, groups the corresponding tensors into batched representations, groups the associated messages, enabling massively parallel belief propagation and related subroutines easelly deployable on a GPU.

## How to install?

1) Clone this repo;
2) Run `pip install .` from the clonned repo under your python environment.

To validate the computation results, some examples and tests rely on an exact quantum circuit simulator available at https://github.com/LuchnikovI/qem. To install it, follow the steps below:

1) Clone the repo https://github.com/LuchnikovI/qem;
2) Install rust (see https://rust-lang.org/tools/install/);
3) Install `maturin` by running `pip install maturin .`;
4) Run `pip install .` from the clonned repo under your python environment.

## How to use?

This package exposes a single entry point, `run_qa`, which executes the full workflow. It accepts a single argument which is a Python dictionary that fully specifies the quantum annealing task. This dictionary serves as a configuration or DSL and can be directly deserialized from JSON or other formats. For a concrete example of the configuration, see `./examples/small_ibm_heavy_hex.py`.
