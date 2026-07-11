"""Delete stale ``demo-`` composition repos from the service org.

Dry-run by default: prints candidates only. Nothing is deleted without --yes.
Fixed-name example repos (no ``demo-`` prefix) are never candidates.

Usage:
    uv run python scripts/cleanup_demo_repos.py --org my-org --days 7 [--yes]
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone


def find_stale_demo_repos(api: object, org: str, days: int) -> list[tuple[str, int]]:
    """Return ``(repo_id, age_days)`` for demo repos older than ``days``."""
    try:
        datasets = list(api.list_datasets(author=org))  # type: ignore[attr-defined]
    except Exception as exc:
        raise RuntimeError(f"'{org}' dataset 목록 조회 실패: {exc}") from exc
    now = datetime.now(timezone.utc)
    prefix = f"{org}/demo-"
    candidates: list[tuple[str, int]] = []
    for ds in datasets:
        if not ds.id.startswith(prefix):
            continue
        created_at = getattr(ds, "created_at", None)
        if created_at is None:
            continue
        age = now - created_at
        if age > timedelta(days=days):
            candidates.append((ds.id, age.days))
    return candidates


def main() -> int:
    parser = argparse.ArgumentParser(description="오래된 demo-* composition repo 정리")
    parser.add_argument("--org", default=os.environ.get("BENCHPRESS_ORG"), help="서비스 org (기본: BENCHPRESS_ORG env)")
    parser.add_argument("--days", type=int, default=7, help="이 일수보다 오래된 repo만 대상 (기본 7)")
    parser.add_argument("--yes", action="store_true", help="실제 삭제 (없으면 dry-run)")
    args = parser.parse_args()
    if not args.org:
        parser.error("--org 또는 BENCHPRESS_ORG 환경변수가 필요합니다")
    from huggingface_hub import HfApi

    api = HfApi()
    try:
        candidates = find_stale_demo_repos(api, args.org, args.days)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if not candidates:
        print(f"'{args.org}'에 {args.days}일 넘은 demo repo가 없습니다.")
        return 0
    for repo_id, age_days in candidates:
        print(f"{repo_id} (age: {age_days}d)")
    if not args.yes:
        print(f"dry-run: 후보 {len(candidates)}개. 실제 삭제하려면 --yes를 붙이세요.")
        return 0
    failures = 0
    for repo_id, _ in candidates:
        try:
            api.delete_repo(repo_id, repo_type="dataset")
            print(f"deleted: {repo_id}")
        except Exception as exc:
            print(f"삭제 실패 {repo_id}: {exc}", file=sys.stderr)
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
