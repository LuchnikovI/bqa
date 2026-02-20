## What is it?

This package provides large-scale tensor-network-based emulator of quantum annealing, powered by belief-propagation-based inference. This implementation introduces a compilation step that classifies graph nodes by degree, groups the corresponding tensors into batched representations, groups the associated messages, enabling massively parallel belief propagation and related subroutines easelly deployable on a GPU. This design enables scaling to 100,000 qubits and beyond for a range of nontrivial problem instances.

# How to use it?
There is a [wiki](https://github.com/LuchnikovI/bqa/wiki) page with the documentation.

## How to install?

1) Clone this repo;
2) Run `pip install .` from the clonned repo under your python environment.

To validate the computation results, some examples and tests rely on an exact quantum circuit simulator available at https://github.com/LuchnikovI/qem. To install it, follow the steps below:

1) Clone the repo https://github.com/LuchnikovI/qem;
2) Install rust (see https://rust-lang.org/tools/install/);
3) Install `maturin` by running `pip install maturin .`;
4) Run `pip install .` from the clonned repo under your python environment.

## How to run benchmarks against MQLib?

First, one need to install an [MQLib](https://github.com/MQLib/MQLib) wrapper awailable [here](https://github.com/LuchnikovI/mqlib_wrap), follow the instruction of README there. Now one can execute scripts in `./benchmarks_against_mqlib`, every script saves a result into a separate directory with time stamp.

## NumPy backend

To run quantum annealing emulation using `numpy` backend, one does not need any extra configuration steps. One can control the precision of the `numpy` backend by setting the environment variable `export BQA_PECISION=single` for the single precision and `export BQA_PRECISION=double` for the double precision.

## CuPy backend
To use `cupy` backend one needs to install [`cupy`](https://cupy.dev/) sepraratelly since it is not in the dependancies list. One also need [`cuTENSOR`](https://docs.nvidia.com/cuda/cutensor/latest/getting_started.html) to enable fast tensor contraction. The precision of the `cupy` backend is always `single`.

