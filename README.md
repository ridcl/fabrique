# fabrique

Fabrique is a collection of popular LLMs in JAX. It provides clean and
research-friendly code as well as easy-to-use wrappers for end users.

Fabrique uses model snapshots from Huggingface Hub and provides familiar high-level API.

## Installation

You can install the latest released version of fabrique from PYPI:

```
pip install fabrique
```

Alternatively, you can mount the development version of fabrique directly to your project and use existing code as reference for your own models:

```
cd /path/to/your/project

# clone the repository
git clone https://github.com/dfdx/fabrique

# or even add it as a submodule
# git submodule add git@github.com:dfdx/fabrique.git

# set up PYTHONPATH to include fabrique as a package
export PYTHONPATH=.:fabrique
```


## Usage

### High-level API

```python
from fabrique.models.llm import LLM


model_id = "microsoft/phi-4"

# note: use keyword arguments that fit your hardware
llm = LLM.from_pretrained(model_id, max_batch_size=1, max_seq_len=512)

out = llm.generate("""<|user|>\nHow to print a value in Python?<|end|>\n<|assistant|>""")
print(out)
```


### Working with models directly

Fabrique is built using [Flax NNX](https://flax.readthedocs.io/en/latest/nnx/index.html).
Citing NNX's home page:

> NNX is a Neural Network library for JAX that focuses on providing the best development experience, so building and experimenting with neural networks is easy and intuitive. It achieves this by embracing Python’s object-oriented model and making it compatible with JAX transforms, resulting in code that is easy to inspect, debug, and analyze.

All Fabrique models can be found in `fabrique/models/<model-name>/modeling.py` files. Feel free to
copy and modify them. If something in the code is unclear, consider it a bug.


The very first model was inspired by the [Meta's Llama 3 repo](https://github.com/meta-llama/llama3).