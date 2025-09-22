# fabrique

_fabrique_ provides foundation components for ML research in LLM/VLM space, including:

* model implementations
* fine-tuning routines and examples
* multi-GPU execution
* interoperability with broader ecosystem

_fabrique_ is written in JAX/Flax NNX and follows their [philosophy](PHILOSOPHY.md).

## Installation

You can install the latest released version of _fabrique_ from PYPI:

```bash
pip install fabrique
```

Alternatively, you can mount the development version of _fabrique_ directly to your project and use existing code as reference for your own models:

```bash
cd /path/to/your/project

# clone the repository
mkdir lib
git clone https://github.com/ridcl/fabrique lib/fabrique

# or even add it as a submodule
# git submodule add git@github.com:ridlc/fabrique.git lib/fabrique

# set up PYTHONPATH to include fabrique as a package
export PYTHONPATH=${PYTHONPATH}:lib/fabrique/src
```


## Usage

TODO


## Model support

As of now, _fabrique_ focuses on Gemma as its primary LLM/VLM implementation. Previously, _fabrique_ also supported Llama 3, Phi 3/4 and Qwen 2.5. You can find these old implementations in [legacy/src/fabrique/models](legacy/src/fabrique/models).