"""EasySteer sanity check: a late greedy flip at scale 0 passes, an early divergence fails."""

from types import SimpleNamespace

import pytest

from gshift.steer import engine


class FakeClient:
    def __init__(self, base, zero, big):
        self.outs = [base, zero, big]

    def run(self, jobs):
        return [(" ".join(map(str, g)), g, "length") for g in self.outs]


@pytest.mark.parametrize("zero,ok", [(list(range(64)), True), (list(range(50)) + [-1] * 14, True),
                                     ([-1] + list(range(1, 64)), False), (list(range(10)) + [-1] * 54, False)])
def test_zero_scale_prefix(monkeypatch, zero, ok):
    monkeypatch.setattr(engine, "prompt_token_ids", lambda *a, **k: [1, 2, 3])
    monkeypatch.setattr(engine, "steering_spec", lambda *a, **k: {})
    base, big = list(range(64)), [7] * 64
    res = engine.sanity_check(FakeClient(base, zero, big), None, SimpleNamespace(chat_template_kwargs={}), None, 36, 1.0)
    assert res["ok"] is ok and res["zero_scale_length"] == 64


def _case(best_l, l1=0.02):
    m = {"L": 0.005, "L_x0.5": 0.06, "L_x2": 0.06, "L-1": 0.04, "L+1": l1, "L_flipped_sign": 0.25, "unsteered": 0.14}
    if not best_l:
        m["L+1"] = 0.004
    return {"mae": m, "best": min(m, key=m.get)}


def test_steering_verdict():
    from gshift.validation import steering_verdict

    assert steering_verdict([_case(True)] * 6)["ok"]
    assert steering_verdict([_case(True)] * 5 + [_case(False)])["ok"]          # one adjacent-layer flip
    assert not steering_verdict([_case(True)] * 2 + [_case(False)] * 4)["ok"]  # off by one layer
    bad = _case(True)
    bad["mae"]["L_flipped_sign"], bad["best"] = 0.001, "L_flipped_sign"
    assert not steering_verdict([_case(True)] * 5 + [bad])["ok"]               # wrong sign never allowed
