"""Self-check for the decision service. Run: python test_service.py

Uses a stub agent so it needs no checkpoint. The real model is exercised by the
end-to-end verification, not here.
"""

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import service


class StubAgent:
    """Stands in for laya.Agent: same cfg keys and system_one shape."""

    device = "cpu"
    cfg = {"max_len": 512, "head_max_len": 192}

    def __init__(self, prob=0.73):
        self.prob = prob
        self.calls = 0

    def system_one(self, state, questions):
        self.calls += 1
        answers = {}
        for qid in questions:
            answers[qid] = {"type": "noul", "noul": self.prob, "confidence": 0.73}
        return {"answers": answers, "usage": {"input_tokens": 94}}


def test_budget_comes_from_checkpoint():
    st = service.State()
    st.agent = StubAgent()
    assert st.budget() == 512 - 192 - 4, st.budget()
    st.agent.cfg = {"max_len": 1024, "head_max_len": 256}
    assert st.budget() == 1024 - 256 - 4, st.budget()


class StubTokenizer:
    """One token per whitespace-separated word; enough to exercise build_sequence."""

    mask_token, mask_token_id, cls_token_id, sep_token_id = "[MASK]", 1, 2, 3

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [10] * len(text.split())}


# The production v1 question, as packages/olaya/src/laya/state.ts defines it. The
# TypeScript parity test asserts the same expected hash, so either side drifting fails.
V1_QUESTIONS = {
    "auto_approve": {
        "type": "noul",
        "instructions": "Is it safe to perform this action without asking the user for confirmation first?",
    }
}
V1_HASH = "091c469afb2e827d48549d0148adb3ac828438327133d7e4efef64fd745ccb0a"


def test_question_hash_is_canonical():
    reordered = {"auto_approve": {"instructions": V1_QUESTIONS["auto_approve"]["instructions"], "type": "noul"}}
    assert service.question_hash(V1_QUESTIONS) == service.question_hash(reordered)
    assert service.question_hash(V1_QUESTIONS) == V1_HASH, service.question_hash(V1_QUESTIONS)
    changed = {"auto_approve": dict(V1_QUESTIONS["auto_approve"], instructions="Is it safe?")}
    assert service.question_hash(changed) != V1_HASH


def test_budget_is_measured_from_the_real_head():
    from laya.common import build_sequence
    from laya.agent import Agent

    st = service.State()
    st.agent = StubAgent()
    st.agent.tok = StubTokenizer()
    measured = st.budget(V1_QUESTIONS)
    ids, _ = build_sequence(st.agent.tok, "", Agent._to_internal(V1_QUESTIONS["auto_approve"]), 512, 192)
    assert measured == 512 - len(ids), measured
    # A short head leaves far more room than the head_max_len cap implies.
    assert measured > st.budget(), (measured, st.budget())
    # Fill exactly the measured budget: nothing is truncated; one more token is.
    fits, _ = build_sequence(st.agent.tok, "w " * measured, Agent._to_internal(V1_QUESTIONS["auto_approve"]), 512, 192)
    assert len(fits) == 512, len(fits)


def test_pinned_checkpoint_refuses_other_questions():
    st = service.State()
    st.agent = StubAgent()
    st.pinned_hash = service.question_hash(V1_QUESTIONS)
    code, _ = service.decide({"state": "x", "questions": V1_QUESTIONS}, st)
    assert code == 200, code
    other = {"auto_approve": {"type": "noul", "instructions": "Is it safe?"}}
    code, body = service.decide({"state": "x", "questions": other}, st)
    assert code == 409 and "probability" not in json.dumps(body), (code, body)
    assert st.agent.calls == 1  # the refused request never reached the model


def test_health_reports_the_checkpoint_gate():
    st = service.State()
    st.agent = StubAgent()
    assert st.describe()["gate"] is None  # the stock checkpoint has no certification
    st.gate = {"passed": True, "threshold": 0.93}
    assert st.describe()["gate"] == {"passed": True, "threshold": 0.93}


def test_budget_endpoint():
    st = service.State()
    st.agent = StubAgent()
    st.agent.tok = StubTokenizer()
    code, body = service.budget({"questions": V1_QUESTIONS}, st)
    assert code == 200 and body["question_hash"] == service.question_hash(V1_QUESTIONS), body
    assert body["state_budget"] == st.budget(V1_QUESTIONS)
    assert service.budget({"questions": {}}, st)[0] == 400
    assert service.budget({"questions": V1_QUESTIONS}, service.State())[0] == 503


def test_not_ready_is_503_not_an_answer():
    st = service.State()
    code, body = service.decide({"state": "x", "questions": {"q": {"type": "noul", "instructions": "i"}}}, st)
    assert code == 503, code
    assert "probability" not in json.dumps(body)


def test_malformed_requests_never_reach_the_model():
    st = service.State()
    st.agent = StubAgent()
    good_q = {"q": {"type": "noul", "instructions": "i"}}
    for payload in (
        "not an object",
        {"questions": good_q},                      # no state
        {"state": "", "questions": good_q},         # empty state
        {"state": "x"},                             # no questions
        {"state": "x", "questions": {}},            # empty questions
        {"state": "x", "questions": {"q": {"type": "noul"}}},        # no instructions
        {"state": "x", "questions": {"q": {"instructions": "i"}}},   # no type
    ):
        code, _ = service.decide(payload, st)
        assert code == 400, (payload, code)
    assert st.agent.calls == 0, "malformed request reached the model"


def test_decide_returns_probability_in_range():
    st = service.State()
    st.agent = StubAgent(prob=0.73)
    code, body = service.decide(
        {"state": {"action": "shell"}, "questions": {"auto_approve": {"type": "noul", "instructions": "i"}}}, st
    )
    assert code == 200, body
    p = body["answers"]["auto_approve"]["probability"]
    assert 0.0 <= p <= 1.0 and p == 0.73, p
    assert body["usage"]["input_tokens"] == 94
    assert "latency_ms" in body


def test_all_questions_share_one_forward_pass():
    st = service.State()
    st.agent = StubAgent()
    qs = {f"q{i}": {"type": "noul", "instructions": "i"} for i in range(4)}
    code, body = service.decide({"state": "x", "questions": qs}, st)
    assert code == 200
    assert len(body["answers"]) == 4
    assert st.agent.calls == 1, "questions were not batched into one call"


def test_http_surface():
    service.STATE = service.State()
    server = ThreadingHTTPServer(("127.0.0.1", 0), service.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = "http://127.0.0.1:%d" % server.server_address[1]
    try:
        health = json.loads(urllib.request.urlopen(base + "/health").read())
        assert health["ready"] is False, health
        assert "state_budget" not in health, "budget reported before the model loaded"

        service.STATE.agent = StubAgent()
        health = json.loads(urllib.request.urlopen(base + "/health").read())
        assert health["ready"] is True and health["state_budget"] == 316, health

        req = urllib.request.Request(
            base + "/decide",
            data=json.dumps({"state": "x", "questions": {"a": {"type": "noul", "instructions": "i"}}}).encode(),
            headers={"Content-Type": "application/json"},
        )
        body = json.loads(urllib.request.urlopen(req).read())
        assert body["answers"]["a"]["probability"] == 0.73, body

        try:
            urllib.request.urlopen(urllib.request.Request(base + "/decide", data=b"{bad"))
            raise AssertionError("malformed JSON was accepted")
        except urllib.error.HTTPError as e:
            assert e.code == 400, e.code

        try:
            urllib.request.urlopen(base + "/nope")
            raise AssertionError("unknown path was accepted")
        except urllib.error.HTTPError as e:
            assert e.code == 404, e.code
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("all checks passed")
