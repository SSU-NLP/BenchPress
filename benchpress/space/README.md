---
title: BenchPress Composer
emoji: 🏋️
colorFrom: indigo
colorTo: pink
sdk: gradio
app_file: app.py
pinned: false
---

# BenchPress Composer

능력 기반 맞춤 평가셋 recipe(`manifest.json`)를 미리보고, 서비스 org의 HF
dataset repo에 게시하는 Space. 데이터는 저장하지 않는다 — 로드는
`benchpress_hub.load_composition`이 원본에서 streaming으로 수행한다.

The live demo requires no account creation, no personal Google or Hugging Face login, and no user-provided API token. Reviewers can inspect benchmark-composition previews directly in the Space. For reproducibility, we provide pre-published example compositions on Hugging Face that can be loaded with benchpress_hub.load_composition. Optional live publishing uses a server-side fine-grained token restricted to a service-owned Hugging Face organization; generated demo repositories are public and use a demo- prefix for cleanup.

## Space Secrets

게시 기능을 켜려면 Space Settings → Variables and secrets에 설정:

- `HF_TOKEN` (secret): 데모 org로 범위가 제한된 fine-grained 토큰
  (해당 org의 dataset repo write 권한만).
- `BENCHPRESS_ORG` (variable): 게시 대상 org 이름.

둘 중 하나라도 없으면 게시 버튼은 비활성화되고 Preview·예시 로드만 동작한다.

## Local run

```bash
HF_TOKEN=<token> BENCHPRESS_ORG=<org> python app.py
```

또는 `hf auth login` 후 `BENCHPRESS_ORG=<org> python app.py`
(`HF_TOKEN` 미설정 시 huggingface_hub가 로컬 로그인 캐시를 사용한다).

## Pre-published examples

`scripts/publish_examples.py`가 고정 이름 예시 3종을 org에 게시하고
round-trip 검증 후 `space/examples.json`을 생성한다. deploy 시 함께 올라간다.

## Deploy

레포 루트에서:

```bash
./space/deploy.sh <username>/<space-name>
```

스크립트가 `space/*` + `benchpress_hub/` + `part2_experiment/hf_sampling.py` +
필요한 `data/*.json`을 staging해 Space repo로 업로드한다.
로컬에 write 권한 HF 토큰이 있어야 한다 (`hf auth login`).
