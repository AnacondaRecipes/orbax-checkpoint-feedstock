import os
import re
import numpy as np
import pytest


def _make_state(seed: int = 0):
    rng = np.random.default_rng(seed)
    return {
        "a": np.arange(16, dtype=np.int32),
        "b": rng.normal(size=(4, 3)).astype(np.float32),
        "nested": {
            "x": np.eye(5, dtype=np.float32),
            "y": (np.arange(6) ** 2).astype(np.int64),
        },
    }


def _assert_state_equal(got, expected):
    np.testing.assert_array_equal(got["a"], expected["a"])
    np.testing.assert_allclose(got["b"], expected["b"], rtol=0, atol=0)
    np.testing.assert_allclose(got["nested"]["x"], expected["nested"]["x"], rtol=0, atol=0)
    np.testing.assert_array_equal(got["nested"]["y"], expected["nested"]["y"])


def test_import():
    import orbax.checkpoint as oc  # noqa: F401


def test_public_symbols():
    import orbax.checkpoint as oc
    assert hasattr(oc, "CheckpointManager")
    assert hasattr(oc, "CheckpointManagerOptions")
    assert hasattr(oc, "PyTreeCheckpointHandler")


def test_manager_constructs_and_closes(tmp_path):
    import orbax.checkpoint as oc
    mngr = oc.CheckpointManager(
        tmp_path,
        item_handlers={"state": oc.PyTreeCheckpointHandler()},
        options=oc.CheckpointManagerOptions(max_to_keep=2),
    )
    mngr.close()


def test_roundtrip_single_step(tmp_path):
    import orbax.checkpoint as oc

    ckpt_dir = tmp_path / "ckpts"
    state = _make_state(0)

    mngr = oc.CheckpointManager(
        ckpt_dir,
        item_handlers={"state": oc.PyTreeCheckpointHandler()},
        options=oc.CheckpointManagerOptions(max_to_keep=2),
    )

    assert mngr.save(0, {"state": state})
    mngr.wait_until_finished()

    restored = mngr.restore(0)
    assert "state" in restored
    _assert_state_equal(restored["state"], state)

    mngr.close()


def test_roundtrip_two_steps_and_latest(tmp_path):
    import orbax.checkpoint as oc

    ckpt_dir = tmp_path / "ckpts"
    s0 = _make_state(0)
    s1 = _make_state(1)

    mngr = oc.CheckpointManager(
        ckpt_dir,
        item_handlers={"state": oc.PyTreeCheckpointHandler()},
        options=oc.CheckpointManagerOptions(max_to_keep=3),
    )

    assert mngr.save(0, {"state": s0})
    assert mngr.save(1, {"state": s1})
    mngr.wait_until_finished()

    latest = mngr.latest_step()
    assert latest in (0, 1)

    restored_latest = mngr.restore(latest)["state"]
    _assert_state_equal(restored_latest, s1 if latest == 1 else s0)

    mngr.close()


def test_all_steps_contains_saved(tmp_path):
    import orbax.checkpoint as oc

    ckpt_dir = tmp_path / "ckpts"
    mngr = oc.CheckpointManager(
        ckpt_dir,
        item_handlers={"state": oc.PyTreeCheckpointHandler()},
        options=oc.CheckpointManagerOptions(max_to_keep=10),
    )

    for step in (0, 2, 5):
        assert mngr.save(step, {"state": _make_state(step)})
    mngr.wait_until_finished()

    steps = set(mngr.all_steps())
    assert {0, 2, 5}.issubset(steps)

    mngr.close()


def test_max_to_keep_enforced(tmp_path):
    import orbax.checkpoint as oc

    ckpt_dir = tmp_path / "ckpts"
    mngr = oc.CheckpointManager(
        ckpt_dir,
        item_handlers={"state": oc.PyTreeCheckpointHandler()},
        options=oc.CheckpointManagerOptions(max_to_keep=2),
    )

    for step in (0, 1, 2):
        assert mngr.save(step, {"state": _make_state(step)})
    mngr.wait_until_finished()

    steps = list(mngr.all_steps())
    assert len(steps) <= 2

    mngr.close()


def test_restore_missing_step_raises(tmp_path):
    import orbax.checkpoint as oc

    ckpt_dir = tmp_path / "ckpts"
    mngr = oc.CheckpointManager(
        ckpt_dir,
        item_handlers={"state": oc.PyTreeCheckpointHandler()},
        options=oc.CheckpointManagerOptions(max_to_keep=2),
    )

    with pytest.raises(Exception):
        mngr.restore(123)

    mngr.close()


def test_checkpoint_directory_created(tmp_path):
    import orbax.checkpoint as oc

    ckpt_dir = tmp_path / "ckpts"
    mngr = oc.CheckpointManager(
        ckpt_dir,
        item_handlers={"state": oc.PyTreeCheckpointHandler()},
        options=oc.CheckpointManagerOptions(max_to_keep=1),
    )

    assert mngr.save(0, {"state": _make_state(0)})
    mngr.wait_until_finished()

    assert any(ckpt_dir.iterdir())

    mngr.close()


def test_no_tmp_leftovers_after_save(tmp_path):
    import orbax.checkpoint as oc

    ckpt_dir = tmp_path / "ckpts"
    mngr = oc.CheckpointManager(
        ckpt_dir,
        item_handlers={"state": oc.PyTreeCheckpointHandler()},
        options=oc.CheckpointManagerOptions(max_to_keep=1),
    )

    assert mngr.save(0, {"state": _make_state(0)})
    mngr.wait_until_finished()

    names = [p.name for p in ckpt_dir.rglob("*")]
    assert not any(re.search(r"tmp", n, re.IGNORECASE) for n in names)

    mngr.close()
