"""Loopback HTTP decision service wrapping a Laya checkpoint.

The checkpoint takes ~21 s to load, so this process is long-lived and pre-warmed: Olaya
starts it once and keeps it. It is never spawned per request.

Protocol
    GET  /health  -> {"ready": bool, "checkpoint": str, "device": str,
                      "max_len": int, "head_max_len": int, "state_budget": int}
    POST /decide  -> {"state": <str|obj>, "questions": {qid: {...}}}
                  -> {"answers": {qid: {...}}, "usage": {...}, "latency_ms": float}
    POST /budget  -> {"questions": {qid: {...}}}
                  -> {"state_budget": int, "question_hash": str}

A fine-tuned checkpoint is bound to the exact questions it was trained on. When the
checkpoint directory holds an `olaya_manifest.json` with a `question_hash`, /decide refuses
(409) any question set that hashes differently, and the client falls back to asking.

On startup the bound port is announced on stdout as a single line:
    OLAYA_SIDECAR_LISTENING {"port": 8731}
so the parent can use an ephemeral port and avoid collisions.
"""

import hashlib
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HOST = "127.0.0.1"
DEFAULT_MODEL = "convaiinnovations/laya"
MAX_BODY_BYTES = 1 << 20
# Decisions slower than this are logged; they are the ones that lose a judgment to a timeout.
SLOW_REQUEST_MS = 500
# A model left untouched goes cold - pages are evicted and the next decision pays for it,
# which is how a judgment gets lost to a timeout after an idle spell. A periodic no-op
# forward pass is far cheaper than the lost decision it prevents.
KEEP_WARM_SECONDS = 60

# ponytail: one model instance behind one lock. Permission requests are user-blocking and
# arrive one at a time, so a pool would just multiply a ~1.7 GB resident model. Laya already
# batches the questions within a single request into one forward pass, which is the batching
# that actually pays.
_LOCK = threading.Lock()


def question_hash(questions):
    """Canonical sha256 of a question set. Must match `questionHash` in the TypeScript client:
    keys sorted at every level, no whitespace, non-ASCII kept as UTF-8."""
    canon = json.dumps(questions, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def read_manifest(checkpoint):
    """The Olaya manifest of a local checkpoint directory, or None (hub ids have none)."""
    path = os.path.join(checkpoint, "olaya_manifest.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


class State:
    """Model and readiness, shared across handler threads."""

    def __init__(self):
        self.agent = None
        self.checkpoint = os.environ.get("OLAYA_LAYA_MODEL", DEFAULT_MODEL)
        self.error = None
        self.last_used = 0.0
        manifest = read_manifest(self.checkpoint)
        self.pinned_hash = (manifest or {}).get("question_hash")
        # Written by the certification step. Live mode acts only on a checkpoint whose gate
        # passed, and only above the threshold certified with it.
        self.gate = (manifest or {}).get("gate")

    @property
    def ready(self):
        return self.agent is not None

    def budget(self, questions=None):
        """Tokens available for state.

        build_sequence lays out [CLS] head [SEP] options [SEP] state [SEP] and gives the state
        whatever the ACTUAL head leaves; head_max_len is only the head's cap. So with the
        questions known, the budget is measured from their tokenized head: for v1's one short
        noul question that is ~480 tokens on a 512 checkpoint, not the 316 the cap implies.
        Without them (or without a tokenizer) the conservative cap-based figure is returned.
        """
        cfg = self.agent.cfg
        max_len = int(cfg.get("max_len", 512))
        head_max_len = int(cfg.get("head_max_len", 192))
        tok = getattr(self.agent, "tok", None)
        if not questions or tok is None:
            return max(0, max_len - head_max_len - 4)
        from laya.agent import Agent
        from laya.common import build_sequence

        rooms = []
        for q in questions.values():
            # With an empty state the sequence is head + [SEP], so the room is what remains.
            ids, _ = build_sequence(tok, "", Agent._to_internal(q), max_len, head_max_len)
            rooms.append(max(0, max_len - len(ids)))
        return min(rooms)

    def describe(self):
        d = {"ready": self.ready, "checkpoint": self.checkpoint}
        if self.error:
            d["error"] = self.error
        if self.ready:
            cfg = self.agent.cfg
            d.update(
                device=str(getattr(self.agent, "device", "unknown")),
                max_len=int(cfg.get("max_len", 512)),
                head_max_len=int(cfg.get("head_max_len", 192)),
                state_budget=self.budget(),
            )
        d["pinned_question_hash"] = self.pinned_hash
        d["gate"] = self.gate
        return d


STATE = State()


def load_model(state=None):
    """Load the checkpoint into `state`. Blocking; run on a background thread."""
    # Resolved at call time, not bound as a default: a default argument would capture the
    # STATE object that existed at import and silently ignore any later reassignment,
    # while the request handlers look the global up per call. Mixing the two diverges.
    state = STATE if state is None else state
    try:
        import laya

        # CPU by default: ~0.1 s per decision is ample for a permission prompt, while MPS beside a
        # resident local LLM measured 8.7 s and recompiles per input shape. "auto" lets Laya pick.
        device = os.environ.get("OLAYA_LAYA_DEVICE") or "cpu"
        device = None if device == "auto" else device
        t0 = time.time()
        agent = laya.load(state.checkpoint, device=device)

        # Warm a spread of sequence lengths, not just one.
        #
        # MPS (and CUDA) compile kernels per input shape, so the cost is paid on the first
        # request at each distinct length, not merely the first request overall. Measured on
        # M5 Pro: 200-700 ms on a new shape against a 40-90 ms steady state. Warming a single
        # tiny state therefore left every real request still paying it, which timed out
        # against the client budget and silently lost the judgment.
        #
        # ponytail: a sweep, not exhaustive coverage - lengths between the buckets still pay
        # once. Removing the variance entirely means padding every sequence to max_len, which
        # costs full-context latency on every call; revisit if spikes show up in the logs.
        warm_q = {"w": {"type": "noul", "instructions": "warm up"}}
        for filler in (0, 40, 120, 400):
            try:
                agent.system_one({"warm": "x " * filler}, warm_q)
            except Exception as warm_error:
                print("warm-up pass failed (continuing): %s" % warm_error, file=sys.stderr, flush=True)
                break

        state.agent = agent
        state.error = None
        print(
            "loaded %s on %s in %.1fs (state budget %d tokens)"
            % (state.checkpoint, agent.device, time.time() - t0, state.budget()),
            file=sys.stderr,
            flush=True,
        )
    except Exception as exc:  # surfaced via /health; the process stays up so it can be retried
        state.error = "%s: %s" % (type(exc).__name__, exc)
        print("load failed: %s" % state.error, file=sys.stderr, flush=True)


def keep_warm(state=None):
    """Keep the model hot so an idle spell cannot cost a real decision its timeout budget."""
    state = STATE if state is None else state
    while True:
        time.sleep(KEEP_WARM_SECONDS)
        if not state.ready:
            continue
        if time.time() - state.last_used < KEEP_WARM_SECONDS:
            continue  # real traffic is keeping it warm already
        try:
            with _LOCK:
                state.agent.system_one({"keepalive": True}, {"k": {"type": "noul", "instructions": "keep warm"}})
            state.last_used = time.time()
        except Exception as exc:
            print("keep-warm pass failed: %s" % exc, file=sys.stderr, flush=True)


def budget(payload, state=None):
    """State budget and hash for a question set. Returns (status_code, body)."""
    state = STATE if state is None else state
    questions = payload.get("questions") if isinstance(payload, dict) else None
    if not isinstance(questions, dict) or not questions:
        return 400, {"error": "missing or empty 'questions'"}
    if not state.ready:
        return 503, {"error": "model not ready", "detail": state.error}
    return 200, {"state_budget": state.budget(questions), "question_hash": question_hash(questions)}


def normalise(qid, answer):
    """Flatten one laya answer into a stable shape.

    `probability` is the single number a caller gates on. It only exists for `noul`, which is
    the one primitive v1 uses; choice and score pass through with their own fields.
    """
    out = {"type": answer.get("type"), "confidence": answer.get("confidence")}
    kind = answer.get("type")
    if kind == "noul":
        out["probability"] = answer.get("noul")
    elif kind == "choice":
        out["choice"] = answer.get("choice")
        out["probabilities"] = answer.get("probabilities")
    elif kind == "score":
        out["score"] = answer.get("score")
        out["probabilities"] = answer.get("probabilities")
    return out


def decide(payload, state=None):
    """Run one decision. Returns (status_code, body)."""
    state = STATE if state is None else state
    if not isinstance(payload, dict):
        return 400, {"error": "body must be a JSON object"}
    questions = payload.get("questions")
    if "state" not in payload or payload["state"] in (None, ""):
        return 400, {"error": "missing 'state'"}
    if not isinstance(questions, dict) or not questions:
        return 400, {"error": "missing or empty 'questions'"}
    for qid, q in questions.items():
        if not isinstance(q, dict) or "type" not in q or "instructions" not in q:
            return 400, {"error": "question %r needs 'type' and 'instructions'" % qid}

    if not state.ready:
        return 503, {"error": "model not ready", "detail": state.error}
    if state.pinned_hash and question_hash(questions) != state.pinned_hash:
        # The checkpoint was trained on different questions; its probabilities mean nothing here.
        return 409, {"error": "question set does not match the checkpoint", "pinned": state.pinned_hash}

    t0 = time.time()
    # All questions go through one system_one call: laya evaluates them in a single
    # batched forward pass, so splitting them would cost a full pass each.
    with _LOCK:
        result = state.agent.system_one(payload["state"], questions)
    state.last_used = time.time()
    elapsed = (time.time() - t0) * 1000

    # A slow decision is invisible to the caller once it gives up, so record it here. This is
    # the only place the real cost of a timed-out judgment can be observed.
    if elapsed > SLOW_REQUEST_MS:
        print(
            "slow decision: %.0f ms for %d token(s) across %d question(s)"
            % (elapsed, result.get("usage", {}).get("input_tokens", -1), len(questions)),
            file=sys.stderr,
            flush=True,
        )

    return 200, {
        "answers": {qid: normalise(qid, a) for qid, a in result["answers"].items()},
        "usage": result.get("usage", {}),
        "latency_ms": round(elapsed, 1),
        "checkpoint": state.checkpoint,
    }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass  # the default logger writes a line per request to stderr

    def _send(self, code, body):
        raw = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _loopback(self):
        """Defence in depth: the socket is bound to loopback, but never trust one control."""
        host = self.client_address[0]
        if host in ("127.0.0.1", "::1", "::ffff:127.0.0.1"):
            return True
        self._send(403, {"error": "loopback only"})
        return False

    def do_GET(self):
        if not self._loopback():
            return
        if self.path.split("?")[0] == "/health":
            return self._send(200, STATE.describe())
        self._send(404, {"error": "not found"})

    def do_POST(self):
        if not self._loopback():
            return
        route = self.path.split("?")[0]
        if route not in ("/decide", "/budget"):
            return self._send(404, {"error": "not found"})
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return self._send(400, {"error": "bad Content-Length"})
        if length <= 0 or length > MAX_BODY_BYTES:
            return self._send(400, {"error": "body must be 1..%d bytes" % MAX_BODY_BYTES})
        try:
            payload = json.loads(self.rfile.read(length))
        except (ValueError, UnicodeDecodeError):
            return self._send(400, {"error": "body must be valid JSON"})
        try:
            code, body = (decide if route == "/decide" else budget)(payload)
        except Exception as exc:
            code, body = 500, {"error": "%s: %s" % (type(exc).__name__, exc)}
        self._send(code, body)


def serve(port=None):
    port = int(os.environ.get("OLAYA_LAYA_PORT", "0") if port is None else port)
    server = ThreadingHTTPServer((HOST, port), Handler)
    # Announced before loading so the parent learns the port immediately and can start
    # health-polling rather than guessing when we are up.
    print(
        "OLAYA_SIDECAR_LISTENING " + json.dumps({"port": server.server_address[1]}),
        flush=True,
    )
    threading.Thread(target=load_model, daemon=True).start()
    threading.Thread(target=keep_warm, daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    serve()
