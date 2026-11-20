"""Checks on the Oracle Cloud deployment files; they never touch a real server."""

import subprocess

from resolveiq.config import PROJECT_ROOT

ORACLE = PROJECT_ROOT / "deploy" / "oracle"


def test_scripts_are_valid_bash():
    for script in ("setup.sh", "update.sh", "push.sh"):
        result = subprocess.run(["bash", "-n", str(ORACLE / script)], capture_output=True)
        assert result.returncode == 0, result.stderr.decode()


def test_setup_needs_an_address_before_doing_anything(tmp_path):
    result = subprocess.run(
        ["bash", str(ORACLE / "setup.sh")],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)},
    )

    assert result.returncode == 1
    assert "PUBLIC_IP" in result.stderr


def test_service_listens_only_locally_and_is_hardened():
    unit = (ORACLE / "resolveiq.service").read_text()

    assert "--host 127.0.0.1" in unit
    assert "--forwarded-allow-ips 127.0.0.1" in unit
    assert "EnvironmentFile=/etc/resolveiq/resolveiq.env" in unit
    assert "Restart=always" in unit
    assert "ProtectSystem=strict" in unit and "NoNewPrivileges=true" in unit
    assert unit.count("@APP_DIR@") >= 3


def test_caddy_streams_drafts_and_uses_the_domain_placeholder():
    caddyfile = (ORACLE / "Caddyfile").read_text()

    site = [line for line in caddyfile.splitlines() if line and not line.startswith("#")]
    assert site[0] == "@DOMAIN@ {"
    assert "reverse_proxy 127.0.0.1:8000" in caddyfile
    assert "flush_interval -1" in caddyfile


def test_push_never_copies_duplicates_or_hidden_files():
    push = (ORACLE / "push.sh").read_text()

    assert "--exclude '* [0-9]*'" in push
    assert "--exclude '.*'" in push
    assert "models/classifier/production/" in push and "models/risk/" in push
