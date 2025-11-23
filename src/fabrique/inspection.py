import jax
from flax import nnx


def summary_size(obj):
    if isinstance(obj, nnx.Module):
        obj = nnx.split(obj)[1]
    elif isinstance(obj, nnx.graph.Static):
        obj = obj.get_value()
    return sum(x.nbytes for x in jax.tree.leaves(obj))


def summary_size_str(obj):
    nbytes = summary_size(obj)
    if nbytes > (1024**3):
        divider = 1024**3
        suffix = "Gb"
    elif nbytes > (1024**2):
        divider = 1024**2
        suffix = "Mb"
    elif nbytes > 1024:
        divider = 1024
        suffix = "Kb"
    else:
        divider = 1
        suffix = ""
    val = nbytes / divider
    if divider == 1:
        # print bytes as integer w/o suffix
        return str(nbytes)
    elif val > 100:
        # print hundreds as integer w/ suffix
        return f"{val}{suffix}"
    else:
        # otherwise print with a single decimal point
        return f"{val:.1f}{suffix}"


def print_size(params):
    def print_obj(name, obj, indent: int):
        indent_str = "  " * indent
        if isinstance(obj, jax.Array) or isinstance(obj, nnx.Variable):
            print(f"{indent_str}{name} : {summary_size_str(obj)}")
        elif isinstance(obj, nnx.graph.Static):
            print(f"{indent_str}{name} : {summary_size_str(obj.get_value())}")
        elif isinstance(obj, nnx.VariableState):
            print_obj(name, obj.value, indent)
        elif isinstance(obj, list):
            print(f"{indent_str}{name} : {summary_size_str(obj)}")
            for i, x in enumerate(obj):
                print_obj(str(i), x, indent + 1)
        elif isinstance(obj, dict) or isinstance(obj, nnx.State):
            print(f"{indent_str}{name} : {summary_size_str(obj)}")
            for key, val in obj.items():
                print_obj(key, val, indent + 1)
        elif obj is None:
            print(f"{indent_str}{name} :: None")
        else:
            print(f"{indent_str}{name} :: {type(obj)} (skipping analysis)")

    obj = params
    if isinstance(obj, nnx.Module):
        _, obj = nnx.split(obj)
    print_obj("params", obj, 0)
