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
