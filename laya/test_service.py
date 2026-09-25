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
