from dataclasses import dataclass

import pytest

from fabrique.utils import set_nested, set_nested_attr, update_tree, get_by_path, set_by_path


def test_update_tree():
    t1 = {"params": {"a": 1, "b": 2}}
    t2 = {"params": {"b": 20, "c": 3}, "cache": {"k": 100}}
    update_tree(t1, t2)
    assert t1 == {"params": {"a": 1, "b": 20, "c": 3}, "cache": {"k": 100}}


def test_set_nested():
    nested = {}
    assert set_nested(nested, ["a", "b", "c"], 42) == {"a": {"b": {"c": 42}}}

    nested = {"a": {"d": 54}}
    assert set_nested(nested, ["a", "b", "c"], 42) == {"a": {"d": 54, "b": {"c": 42}}}

    nested = {"a": {"d": 54}}
    assert set_nested(nested, ["a", "b[3]", "c"], 42) == {
        "a": {"d": 54, "b": [None, None, None, {"c": 42}]}
    }

    nested = {"a": {"d": 54}}
    assert set_nested(nested, ["a", "b[12]", "c"], 42) == {
        "a": {"d": 54, "b": [None] * 12 + [{"c": 42}]}
    }


def test_set_nested_attr():
    @dataclass
    class Baz:
        c: int

    @dataclass
    class Bar:
        b: Baz | list[Baz]

    @dataclass
    class Foo:
        a: Bar

    obj = Foo(Bar(Baz(0)))
    assert set_nested_attr(obj, ["a", "b", "c"], 42) == Foo(Bar(Baz(42)))

    obj = Foo(Bar([Baz(0), Baz(1), Baz(2)]))
    assert set_nested_attr(obj, ["a", "b[1]", "c"], 42) == Foo(
        Bar([Baz(0), Baz(42), Baz(2)])
    )

    obj = Foo(Bar(Baz(0)))
    with pytest.raises(AttributeError):
        set_nested_attr(obj, ["a", "b", "d"], 42)

    with pytest.raises(AttributeError):
        set_nested_attr(obj, ["a", "d", "c"], 42)


## new style paths

def test_get_set_by_path():
    import pytest
    import jax.numpy as jnp
    from dataclasses import dataclass

    @dataclass
    class C:
        val: jax.Array

    @dataclass
    class B:
        cs: list[C]

    @dataclass
    class A:
        bs: dict[str, B]

    zeros = jnp.zeros((3, 4))
    ones = jnp.ones((3, 4))
    twos = 2 * ones
    another = jnp.ones((4,5))
    a = A({"key": B([C(ones)])})

    assert get_by_path(a, "bs.key") == a.bs["key"]
    assert get_by_path(a, "bs.key.cs.0") == a.bs["key"].cs[0]
    assert (get_by_path(a, "bs.key.cs.0.val") == a.bs["key"].cs[0].val).all()

    set_by_path(a, "bs.key.cs.0.val", twos)
    assert (a.bs["key"].cs[0].val == twos).all()
    with pytest.raises(ValueError):
        set_by_path(a, "bs.key.cs.0.val", another, raising=True)
    with pytest.raises(ValueError):
        set_by_path(a, "bs.key.cs.0.val", "wrong type", raising=True)

    set_by_path(a, "bs.key.cs.0", C(zeros))
    assert a.bs["key"].cs[0] == C(zeros)
    with pytest.raises(ValueError):
        set_by_path(a, "bs.key.cs.1", C(zeros), raising=True)

    set_by_path(a, "bs.key", B([]))
    assert a.bs["key"] == B([])
    with pytest.raises(ValueError):
        set_by_path(a, "bs.another_key", B([]), raising=True)
