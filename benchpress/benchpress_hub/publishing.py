"""Publishing helpers: credentials, namespaces, and demo repo naming.

Write-side counterpart to ``composition`` (which stays read/manifest-focused).
The demo track publishes with a server-side token to a service org — no user
login — so every credential error message must pass through ``scrub_secrets``.
"""

from __future__ import annotations

import re
import secrets as _secrets
from typing import Any

_UNSAFE_CHARS_RE = re.compile(r"[^a-z0-9._-]+")
_REPO_NAME_MAX = 96

CREDENTIAL_HELP = (
    "HF 자격증명이 없습니다: Space에서는 HF_TOKEN secret을 설정하고, "
    "로컬에서는 `hf auth login` 후 재시도하세요"
)


def scrub_secrets(text: str, secrets: list[str | None]) -> str:
    """Replace each non-empty secret substring with ``***``."""
    for secret in secrets:
        if secret:
            text = text.replace(secret, "***")
    return text


def resolve_publisher(
    *,
    token: str | None = None,
    org: str | None = None,
    api: Any | None = None,
) -> tuple[Any, str]:
    """Resolve an ``HfApi`` client and the namespace to publish under.

    With ``token=None``, ``HfApi`` falls back to the ``HF_TOKEN`` env var or
    the local ``hf auth login`` cache. When ``org`` is given, ``whoami`` is
    skipped (lazy: org publishing fails loudly at ``create_repo`` anyway).
    """
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
    if org:
        return api, org
    try:
        namespace = api.whoami()["name"]
    except Exception as exc:
        message = scrub_secrets(f"{CREDENTIAL_HELP} ({exc!r})", [token])
        raise RuntimeError(message) from exc
    return api, str(namespace)


def _grants_write(permissions: list[str]) -> bool:
    return any(perm == "repo.write" or perm == "write" or perm.endswith(".write") for perm in permissions)


def verify_write_access(api: Any, *, namespace: str) -> tuple[bool, str]:
    """Check the resolved token can write repos under ``namespace``.

    Fails fast with a human-readable reason instead of surfacing a deep
    ``create_repo`` 403 at publish time. Classic tokens are checked by role;
    fine-grained tokens by their global/scoped permissions on ``namespace``.
    Unknown token shapes pass — never false-reject a valid token.
    """
    try:
        info = api.whoami()
    except Exception as exc:  # network / auth failure — surface, don't crash
        return False, f"자격증명 확인 실패 ({exc!r})"
    access = info.get("auth", {}).get("accessToken", {})
    role = access.get("role")
    if role == "write":
        return True, ""
    if role == "read":
        return False, "읽기 전용 토큰입니다 — write 권한이 있는 토큰이 필요합니다."
    if role == "fineGrained":
        fine = access.get("fineGrained", {})
        if _grants_write(fine.get("global", [])):
            return True, ""
        for scope in fine.get("scoped", []):
            if scope.get("entity", {}).get("name") == namespace and _grants_write(scope.get("permissions", [])):
                return True, ""
        return False, f"fine-grained 토큰에 '{namespace}' 네임스페이스 write 권한이 없습니다."
    return True, ""


def sanitize_repo_name(name: str) -> str:
    """Slugify to a safe HF repo name: lowercase ``[a-z0-9._-]``, alnum start."""
    slug = _UNSAFE_CHARS_RE.sub("-", name.lower())
    slug = slug.strip("-._")[:80].rstrip("-._")
    if not slug:
        raise ValueError("사용할 수 없는 repo 이름")
    return slug


def build_demo_repo_id(namespace: str, name: str) -> str:
    """Random-suffixed demo repo id: ``{namespace}/demo-{slug}-{hex6}``."""
    slug = sanitize_repo_name(name)
    max_slug = _REPO_NAME_MAX - len("demo-") - len("-") - 6
    slug = slug[:max_slug].rstrip("-._")
    return f"{namespace}/demo-{slug}-{_secrets.token_hex(3)}"
