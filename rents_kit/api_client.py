"""Microburbs Hackathon API client (design.md sec. 10; T2.5, T11.8).
- key from .env (MICROBURBS_KEY), Bearer auth
- token-bucket rate limit (config.API_MAX_RPS), 429 Retry-After, 5xx/timeout backoff, stop on 401, 403 recorded
- endpoint allow/block lists checked BEFORE any network call
- every response cached on disk; X-Request-Id logged
A MockTransport serves the synthetic mock pages so tests never touch the real API."""
import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

import config as C


class BlockedEndpoint(Exception):
    pass


class AuthError(Exception):
    pass


def endpoint_allowed(path):
    """Block list wins; a block entry matches whole path segments (so /market/median-rent is blocked
    but /market/median-rent-series is allowed)."""
    import re
    p = path.split("?")[0]
    for b in C.API_BLOCK:
        if re.search("/" + re.escape(b.strip("/")) + r"(/|$)", p):
            return False
    return any(a in p for a in C.API_ALLOW)


def read_key(env_path=None):
    names = [C.API_KEY_ENV, "MICROBURBS_KEY"]
    for n in names:
        if os.environ.get(n):
            return os.environ[n]
    env_path = Path(env_path or C.ROOT / ".env")
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8-sig").splitlines():
            for n in names:
                if line.strip().startswith(n + "="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


class HttpTransport:
    def __init__(self, key):
        self.key = key

    def get(self, path, params):
        url = C.API_BASE + urllib.parse.quote(path) + ("?" + urllib.parse.urlencode(params) if params else "")
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {self.key}", "User-Agent": "rents-kit"})
        try:
            with urllib.request.urlopen(req, timeout=C.API_TIMEOUT_S) as r:
                return r.status, dict(r.headers), json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "ignore")
            try:
                body = json.loads(body)
            except ValueError:
                pass
            return e.code, dict(e.headers or {}), body


class MockTransport:
    """Serves ../data_synth/mock_api pages like the real endpoint; injects one 429 and one 503."""

    def __init__(self, mock_dir):
        self.dir = Path(mock_dir)
        meta = json.loads((self.dir / "_suburbs.json").read_text(encoding="utf-8"))
        self.covered, self.uncovered = set(meta["covered"]), set(meta["uncovered"])
        self.calls, self._tripped = 0, set()

    def get(self, path, params):
        self.calls += 1
        h = {"X-Request-Id": f"mock-{self.calls}"}
        if path == "/v1/me":
            return 200, h, {"data": {"team": "mock"}}
        if path.startswith("/v1/suburbs/list"):
            return 200, h, {"data": [{"suburb": s, "state": "New South Wales"} for s in sorted(self.covered)]}
        if "/listings/rent-history" in path:
            name = urllib.parse.unquote(path.split("/v1/suburbs/")[1].split("/listings")[0])
            if name in self.uncovered:
                return 403, h, {"detail": "outside hackathon area"}
            off = int(params.get("offset", 0))
            if off == C.API_PAGE and "429" not in self._tripped:
                self._tripped.add("429")
                return 429, {**h, "Retry-After": "0"}, {"detail": "slow down"}
            if off == 2 * C.API_PAGE and "503" not in self._tripped:
                self._tripped.add("503")
                return 503, h, {"detail": "busy"}
            df = pd.read_csv(self.dir / f"{name}.csv", dtype=str)
            df = df[(df.listing_date >= params.get("from_date", "0000")) & (df.listing_date <= params.get("to_date", "9999"))]
            lim = int(params.get("limit", 200))
            page = df.iloc[off:off + lim]
            return 200, h, {"data": {"area_name": name, "meta": {"limit": lim, "offset": off, "returned": len(page),
                                                                 "total": len(df)},
                                     "results": page.to_dict("records")}}
        if "/transport/nearby" in path:
            return 200, h, {"data": {"nearest": {"train": {"distance_m": 650}, "tram": {"distance_m": 900}}}}
        return 404, h, {"detail": "not in mock"}


class Client:
    def __init__(self, transport=None, cache_dir=None, mock_dir=None, log=print):
        if transport is None:
            if mock_dir:
                transport = MockTransport(mock_dir)
            else:
                key = read_key()
                if not key:
                    raise AuthError(f"No API key: put {C.API_KEY_ENV}=<key> in .env (never commit it)")
                transport = HttpTransport(key)
        self.t = transport
        self.cache = Path(cache_dir or C.CACHE_DIR) / "raw"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.log = log
        self._last = 0.0
        self.stats = {"calls": 0, "cached": 0, "429": 0, "403": 0, "retries": 0}

    def _wait(self):
        gap = 1.0 / C.API_MAX_RPS
        now = time.monotonic()
        if now - self._last < gap:
            time.sleep(gap - (now - self._last))
        self._last = time.monotonic()

    def get(self, path, params=None, use_cache=True):
        if not endpoint_allowed(path):
            raise BlockedEndpoint(f"Blocked endpoint (undated, live or not allowed): {path}")
        params = params or {}
        key = hashlib.sha1((path + json.dumps(params, sort_keys=True)).encode()).hexdigest()[:20]
        cf = self.cache / f"{key}.json"
        if use_cache and cf.exists():
            self.stats["cached"] += 1
            return json.loads(cf.read_text(encoding="utf-8"))
        delay = 1.0
        for attempt in range(C.API_RETRIES + 1):
            self._wait()
            self.stats["calls"] += 1
            try:
                status, headers, body = self.t.get(path, params)
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                status, headers, body = 599, {}, {"detail": str(e)}
            rid = headers.get("X-Request-Id", "-")
            if status == 200:
                out = {"status": 200, "request_id": rid, "body": body}
                cf.write_text(json.dumps(out), encoding="utf-8")
                return out
            if status == 401:
                raise AuthError(f"401 from API (request {rid}): check the key in .env")
            if status == 403:
                self.stats["403"] += 1
                return {"status": 403, "request_id": rid, "body": body}
            if status == 429:
                self.stats["429"] += 1
                wait = float(headers.get("Retry-After", 2) or 2)
                self.log(f"429 rate-limited (request {rid}); waiting {wait:.0f}s")
                time.sleep(wait)
                continue
            self.stats["retries"] += 1
            self.log(f"API {status} on {path} (request {rid}); retry {attempt + 1} in {delay:.0f}s")
            time.sleep(delay if not isinstance(self.t, MockTransport) else 0)
            delay = min(delay * 2, 30)
        return {"status": status, "request_id": rid, "body": body}
