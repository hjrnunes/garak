"""Capture link states: live, released, lost and stalled."""

import copy
import pickle

import pytest

pytest.importorskip("midojo_spec")

from garak.evidence.links import CaptureLink


def link(**changes):
    """A live link with caller-supplied changes."""
    values = dict(name="wire", stall_after=None, bound=None)
    values.update(changes)
    return CaptureLink(**values)


def test_live_link_beats_and_stays_live():
    """A bound, beating link is live, whatever its age without a stall limit."""
    seam = link()
    seam.beat(0.0)
    assert seam.state(100.0) == "live", "a bound link with no stall limit stalled"
    seam.beat(50.0)
    assert seam.state(51.0) == "live", "a recent beat is live"


def test_lost_link_is_lost_whatever_beats():
    """A loss wins over every later beat."""
    seam = link()
    seam.beat(0.0)
    seam.lose("stream ended")
    seam.beat(100.0)
    assert seam.state(100.0) == "lost", "a beat revived a lost link"


def test_expired_binding_is_lost():
    """A binding the peer expired is a loss, even while the socket lives."""
    seam = link(bound=lambda: False)
    seam.beat(0.0)
    assert seam.state(1.0) == "lost", "an expired binding read as live"


@pytest.mark.parametrize("failure", [ConnectionError, TimeoutError, RuntimeError])
def test_binding_that_cannot_be_read_is_lost(failure):
    """A predicate that cannot reach its peer cannot confirm the binding."""

    def unreachable():
        raise failure("peer gone")

    seam = link(bound=unreachable)
    seam.beat(0.0)
    assert seam.state(1.0) == "lost", "an unreadable binding read as live"


def test_defective_binding_predicate_is_not_hidden():
    """An error that is not a lost peer is the predicate's defect and surfaces."""

    def defective():
        raise TypeError("wrong call")

    seam = link(bound=defective)
    with pytest.raises(TypeError):
        seam.state(0.0)


def test_stalled_link_is_stalled():
    """A beat older than the stall limit is a stall, and a fresh beat clears it."""
    seam = link(stall_after=5.0)
    seam.beat(0.0)
    assert seam.state(5.0) == "live", "a beat at the limit is not yet stale"
    assert seam.state(5.5) == "stalled", "a stale beat read as live"
    seam.beat(6.0)
    assert seam.state(7.0) == "live", "a fresh beat did not clear the stall"


def test_an_older_beat_arriving_late_does_not_move_the_beat_back():
    """Frames on two threads can record their beats out of order; the newest stands."""
    seam = link(stall_after=5.0)
    seam.beat(10.0)
    seam.beat(1.0)
    assert seam.state(11.0) == "live", "a late, older beat hid the newer one"


def test_released_link_ignores_loss_and_stall():
    """An acknowledged release ends the link cleanly, before any later loss."""
    seam = link(stall_after=1.0, bound=lambda: False)
    seam.beat(0.0)
    seam.release()
    seam.lose("ended after the release")
    assert seam.state(100.0) == "released", "a loss after a release was kept"


def test_release_after_loss_keeps_the_loss():
    """The first end of a link stands: a release cannot unsay a loss."""
    seam = link()
    seam.beat(0.0)
    seam.lose("stream dropped")
    seam.release()
    assert seam.state(0.0) == "lost", "a release erased the loss"
    assert seam.summary()["released"] is False, "the summary recorded the release"


def test_loss_after_release_is_ignored():
    """A release before the loss keeps the link released."""
    seam = link()
    seam.beat(0.0)
    seam.release()
    seam.lose("late loss")
    assert seam.state(0.0) == "released", "release state lost"


def test_summary_is_diagnostics_only():
    """The summary names the link and its flags, never the digest's content."""
    seam = link(stall_after=2.0)
    seam.beat(0.0)
    seam.lose("gone")
    assert seam.summary() == {
        "name": "wire",
        "released": False,
        "lost": "gone",
        "stall_after": 2.0,
    }, "summary lost the link's state"


def test_live_link_refuses_pickling_and_copying():
    """A link holds a lock and shared timing state, so it never copies."""
    seam = link()
    with pytest.raises(TypeError):
        pickle.dumps(seam)
    with pytest.raises(TypeError):
        copy.copy(seam)
    with pytest.raises(TypeError):
        copy.deepcopy(seam)
