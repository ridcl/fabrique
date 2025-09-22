import jax

from fabrique.utils import (
    get_by_path,
    set_by_path,
)


def test_get_set_by_path():
    from dataclasses import dataclass

    import jax.numpy as jnp
    import pytest

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
    another = jnp.ones((4, 5))
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
