"""Offline tests for benchpress_hub.publishing — HF API is faked throughout."""

from __future__ import annotations

import re

import pytest

from benchpress_hub import publishing as pub


class FakeApi:
    def __init__(self, fail_whoami: bool = False):
        self.fail_whoami = fail_whoami
        self.whoami_calls = 0

    def whoami(self) -> dict:
        self.whoami_calls += 1
        if self.fail_whoami:
            raise RuntimeError("401 Unauthorized (Bearer hf_secret123)")
        return {"name": "someuser"}


def test_resolve_publisher_defaults_to_whoami_name() -> None:
    api = FakeApi()
    got_api, namespace = pub.resolve_publisher(api=api)
    assert got_api is api
    assert namespace == "someuser"
    assert api.whoami_calls == 1


def test_resolve_publisher_org_skips_whoami() -> None:
    api = FakeApi()
    got_api, namespace = pub.resolve_publisher(org="myorg", api=api)
    assert got_api is api
    assert namespace == "myorg"
    assert api.whoami_calls == 0


def test_resolve_publisher_whoami_failure_scrubs_token() -> None:
    api = FakeApi(fail_whoami=True)
    with pytest.raises(RuntimeError) as err:
        pub.resolve_publisher(token="hf_secret123", api=api)
    assert "hf_secret123" not in str(err.value)
    assert "HF 자격증명" in str(err.value)


def test_scrub_secrets_replaces_tokens_ignores_empty() -> None:
    text = "auth hf_abc failed; retry with hf_abc"
    assert pub.scrub_secrets(text, ["hf_abc"]) == "auth *** failed; retry with ***"
    assert pub.scrub_secrets(text, [None, ""]) == text
    assert pub.scrub_secrets(text, [None, "hf_abc", ""]) == "auth *** failed; retry with ***"


def test_sanitize_repo_name_slugifies() -> None:
    assert pub.sanitize_repo_name("My Mix!! 한글") == "my-mix"
    assert pub.sanitize_repo_name("--my.mix_") == "my.mix"


def test_sanitize_repo_name_rejects_unusable() -> None:
    with pytest.raises(ValueError, match="repo 이름"):
        pub.sanitize_repo_name("")
    with pytest.raises(ValueError, match="repo 이름"):
        pub.sanitize_repo_name("!!! 한글 ---")


def test_sanitize_repo_name_truncates_long_names() -> None:
    slug = pub.sanitize_repo_name("a" * 200)
    assert slug == "a" * 80
    assert len(pub.sanitize_repo_name("a" * 79 + "-b")) <= 80


def test_build_demo_repo_id_format_and_randomness() -> None:
    repo_id = pub.build_demo_repo_id("myorg", "My Mix")
    assert re.fullmatch(r"myorg/demo-[a-z0-9][a-z0-9._-]*-[0-9a-f]{6}", repo_id)
    assert pub.build_demo_repo_id("myorg", "My Mix") != repo_id


def test_build_demo_repo_id_name_part_within_96_chars() -> None:
    repo_id = pub.build_demo_repo_id("myorg", "x" * 300)
    name_part = repo_id.split("/", 1)[1]
    assert len(name_part) <= 96
    assert re.fullmatch(r"demo-[a-z0-9][a-z0-9._-]*-[0-9a-f]{6}", name_part)


class WhoamiApi:
    """Returns a caller-supplied whoami payload (for write-access checks)."""

    def __init__(self, access: dict | None, *, fail: bool = False):
        self.fail = fail
        self._info = {"name": "me", "auth": {"accessToken": access or {}}}

    def whoami(self) -> dict:
        if self.fail:
            raise RuntimeError("network down (hf_secret)")
        return self._info


def _fine(scoped_perms: list[str], *, entity: str = "me", global_perms: list[str] | None = None) -> dict:
    return {
        "role": "fineGrained",
        "fineGrained": {
            "global": global_perms or [],
            "scoped": [{"entity": {"type": "user", "name": entity}, "permissions": scoped_perms}],
        },
    }


def test_verify_write_access_classic_roles() -> None:
    ok, _ = pub.verify_write_access(WhoamiApi({"role": "write"}), namespace="me")
    assert ok
    ok, reason = pub.verify_write_access(WhoamiApi({"role": "read"}), namespace="me")
    assert not ok and "읽기 전용" in reason


def test_verify_write_access_fine_grained() -> None:
    # write scoped to the target namespace -> allowed
    ok, _ = pub.verify_write_access(WhoamiApi(_fine(["repo.content.read", "repo.write"])), namespace="me")
    assert ok
    # global write applies everywhere
    ok, _ = pub.verify_write_access(WhoamiApi(_fine([], global_perms=["repo.write"])), namespace="me")
    assert ok
    # read-only scope -> rejected with namespace-specific reason
    ok, reason = pub.verify_write_access(WhoamiApi(_fine(["repo.content.read"])), namespace="me")
    assert not ok and "me" in reason
    # write on a different namespace only -> rejected for target
    ok, _ = pub.verify_write_access(WhoamiApi(_fine(["repo.write"], entity="other")), namespace="me")
    assert not ok


def test_verify_write_access_unknown_shape_passes() -> None:
    ok, _ = pub.verify_write_access(WhoamiApi({"role": "somethingNew"}), namespace="me")
    assert ok


def test_verify_write_access_whoami_failure_returns_reason() -> None:
    ok, reason = pub.verify_write_access(WhoamiApi(None, fail=True), namespace="me")
    assert not ok and "확인 실패" in reason
