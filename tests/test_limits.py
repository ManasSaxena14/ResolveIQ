from fastapi.testclient import TestClient

from resolveiq.api.limits import (
    AccessPolicy,
    SlidingWindowLimiter,
    key_fingerprint,
    load_api_keys,
)
from resolveiq.api.main import create_app
from resolveiq.config import RateLimitConfig
from service_doubles import make_copilot


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _config(**overrides) -> RateLimitConfig:
    values = {"enabled": True, "anonymous_per_minute": 2, "key_per_minute": 5}
    return RateLimitConfig(**{**values, **overrides})


def test_window_allows_up_to_the_limit_then_recovers():
    clock = FakeClock()
    limiter = SlidingWindowLimiter(clock=clock)

    first = limiter.hit("a", 2)
    second = limiter.hit("a", 2)
    third = limiter.hit("a", 2)

    assert (first.allowed, first.remaining) == (True, 1)
    assert (second.allowed, second.remaining) == (True, 0)
    assert third.allowed is False
    assert third.headers()["Retry-After"] == "61"
    assert limiter.hit("b", 2).allowed, "identities are counted separately"

    clock.now += 60
    assert limiter.hit("a", 2).allowed


def test_limiter_forgets_the_oldest_identities():
    limiter = SlidingWindowLimiter(clock=FakeClock(), max_identities=2)
    for identity in ("a", "b", "c"):
        limiter.hit(identity, 1)

    assert limiter.hit("a", 1).allowed, "a was evicted, so it starts fresh"


def test_keys_are_read_from_the_environment():
    keys = load_api_keys({"RESOLVEIQ_API_KEYS": " one, two ,,"})

    assert keys == {"one", "two"}
    assert load_api_keys({}) == frozenset()
    assert len(key_fingerprint("one")) == 12 and "one" not in key_fingerprint("one")


def test_policy_tiers_and_invalid_keys():
    policy = AccessPolicy(
        _config(), api_keys={"secret"}, limiter=SlidingWindowLimiter(clock=FakeClock())
    )

    assert policy.check(None, "1.2.3.4").limit == 2
    assert policy.check("secret", "1.2.3.4").limit == 5
    assert policy.check("wrong", "1.2.3.4") is None


def test_forwarded_for_is_only_trusted_when_configured():
    plain = AccessPolicy(_config())
    proxied = AccessPolicy(_config(trust_forwarded_for=True))

    assert plain.client_address("10.0.0.1", "9.9.9.9") == "10.0.0.1"
    assert proxied.client_address("10.0.0.1", "9.9.9.9, 10.0.0.1") == "9.9.9.9"
    assert proxied.client_address(None, None) == "unknown"


def test_disabled_limits_still_check_keys():
    policy = AccessPolicy(_config(enabled=False), api_keys={"secret"})

    assert all(policy.check(None, "ip").allowed for _ in range(10))
    assert policy.check("wrong", "ip") is None


def test_api_returns_429_with_retry_after(embedder, vector_store):
    access = AccessPolicy(_config(), api_keys={"secret"})
    app = create_app(make_copilot(embedder, vector_store), access=access)

    with TestClient(app) as client:
        statuses = [client.get("/stats").status_code for _ in range(3)]
        limited = client.get("/stats")
        keyed = client.get("/stats", headers={"X-API-Key": "secret"})
        bad_key = client.get("/stats", headers={"X-API-Key": "nope"})
        health = [client.get("/health").status_code for _ in range(5)]

    assert statuses == [200, 200, 429]
    assert limited.json()["detail"].startswith("Too many requests")
    assert int(limited.headers["Retry-After"]) > 0
    assert limited.headers["X-Request-ID"]
    assert keyed.status_code == 200
    assert keyed.headers["X-RateLimit-Limit"] == "5"
    assert bad_key.status_code == 401
    assert health == [200] * 5, "health checks are never limited"


def test_cors_allows_the_configured_frontend_and_wraps_429(embedder, vector_store):
    access = AccessPolicy(_config(anonymous_per_minute=1))
    app = create_app(
        make_copilot(embedder, vector_store),
        access=access,
        cors_origins=["https://resolveiq-web.onrender.com"],
    )
    origin = {"Origin": "https://resolveiq-web.onrender.com"}

    with TestClient(app) as client:
        preflight = client.options(
            "/analyze",
            headers={**origin, "Access-Control-Request-Method": "POST",
                     "Access-Control-Request-Headers": "content-type"},
        )  # fmt: skip
        first = client.get("/stats", headers=origin)
        limited = client.get("/stats", headers=origin)
        foreign = client.get("/health", headers={"Origin": "https://evil.example"})

    assert preflight.status_code == 200
    assert first.headers["access-control-allow-origin"] == origin["Origin"]
    assert limited.status_code == 429
    assert limited.headers["access-control-allow-origin"] == origin["Origin"]
    assert "retry-after" in limited.headers["access-control-expose-headers"].lower()
    assert "access-control-allow-origin" not in foreign.headers


def test_cors_origins_come_from_the_environment(monkeypatch):
    from resolveiq.api.main import cors_origins_from_env

    monkeypatch.setenv("RESOLVEIQ_CORS_ORIGINS", "https://a.example/, https://b.example")
    assert cors_origins_from_env() == ["https://a.example", "https://b.example"]
    monkeypatch.delenv("RESOLVEIQ_CORS_ORIGINS")
    assert cors_origins_from_env() == []
