# M4-2 · LlamaIndex 재구현 비교

같은 RAG를 LlamaIndex로 한 번 더 만들어 직접 구현한 6단계와 비교한다.

## 이 폴더는 제품이 아니다

`MEMORY.md` D-003은 LlamaIndex를 도입하지 않기로 했다. `docs/PACKAGES.md`가 **"비교 목적으로만"** 이라는 예외를 뒀고, 이 폴더가 그 예외다.

격리 방법:

| 항목 | 어떻게 |
|---|---|
| 가상환경 | `.venv-llamaindex` (메인 `.venv`와 분리) |
| 의존성 | `requirements-llamaindex.txt` (메인 `requirements.txt` 무변경) |
| 코드 | `experiments/` 아래에만. `src/`는 읽기만 한다 |
| 레이어 검사 | `scripts/check_layers.py`는 `src/`만 본다. 이 폴더는 대상 밖 |

`docs/PACKAGES.md`는 "별도 브랜치"라고 적었지만 이 프로젝트는 git 저장소가 아니다. 폴더와 가상환경으로 대신했다.

## 실행

```bash
python -m venv .venv-llamaindex
.venv-llamaindex\Scripts\python.exe -m pip install -r requirements-llamaindex.txt

.venv-llamaindex\Scripts\python.exe experiments\m4-2-llamaindex\build_index.py   # 1~4단계
.venv-llamaindex\Scripts\python.exe experiments\m4-2-llamaindex\run_eval.py      # 5~6단계 + 채점
```

Ollama가 떠 있어야 한다. `bge-m3`와 `qwen2.5:3b`를 쓴다.

## 비교가 성립하도록 맞춘 것

한 번에 하나만 다르게 한다는 원칙(`eval/README.md`)을 여기에도 적용했다. **다른 것은 프레임워크 하나뿐이다.**

| 항목 | 값 | 근거 |
|---|---|---|
| 색인 재료 | QA쌍 (질문 + 답변) | D-032 |
| 청킹 | 쪼개지 않는다 | D-032, E-004 |
| 임베딩 | `bge-m3` | D-005 |
| 생성 | `qwen2.5:3b`, temperature 0 | D-033 |
| 검색 | 상위 3개, 유사도 0.35 이상 | D-030 |
| 컨텍스트 | 8,192 토큰 | 잘림 방지 |
| 프롬프트 | `src.application.answering.INSTRUCTION` 그대로 | — |
| 평가 | 200건, seed 42, 색인에서 제외 | 누출 방지 |
| 채점 | 음절 2-gram F1 | D-026 |

## 프레임워크에 맡기지 않은 것

세 가지는 일부러 우리 코드를 그대로 썼다.

1. **1단계 적재** — AI Hub JSON은 구조가 고유해서 어느 프레임워크를 쓰든 파싱 코드를 직접 짠다. 프레임워크가 절약해 주는 것이 없어 비교 대상에서 뺐다.
2. **안전 게이트** — 유사도 임계값(생성 전)과 `SafetyPolicy`(생성 후)는 도메인 규칙이다. 프레임워크 기능이 아니다.
3. **채점** — 지표까지 바꾸면 점수 차이가 파이프라인 때문인지 채점 때문인지 알 수 없다.

즉 LlamaIndex가 맡은 것은 **2~6단계의 배선**이다.

## 상대편도 같은 날 코드로 잰다

`eval/m3-6-qaindex.json`(2026-09-09)을 기준으로 쓰면 안 된다. 그 뒤 생성기에 `num_ctx`가 붙었고 안전 규칙이 하나 늘었다. 차이가 프레임워크 때문인지 그동안의 변경 때문인지 구분되지 않는다.

그래서 우리 쪽도 다시 만든다.

```bash
.venv\Scripts\python.exe experiments\m4-2-llamaindex\build_ours.py   # 누출 없는 npz 색인
.venv\Scripts\python.exe -m src.cli eval --aihub ^
  --index index\aihub-qa-holdout.npz --sample 200 --seed 42 --material qa ^
  --out eval\m4-2-ours.json --change "M4-2 비교 상대편"
```

CLI `ingest`에는 평가 케이스를 빼는 기능이 없다. 누출 없는 색인이 필요한 것은 실험뿐이라 제품 코드에 넣지 않았다. `build_ours.py`가 그 조립을 한다.

## 결과 (2026-09-10)

| 항목 | 직접 구현 | LlamaIndex 0.14.24 |
|---|---|---|
| F1 (공통 171건) | 0.3210 | 0.3182 (**-0.0028**) |
| F1 (각자 채점) | 0.321 (174건) | 0.3169 (194건) |
| 색인 빌드 (8,831건) | 6.3분 | 6.7분 |
| 색인 크기 | **37.5 MB** | 167 MB |
| 색인 파일 수 | 1개 | 5개 |
| 2~6단계 코드 | 568줄 | 약 135줄 |
| 런타임 의존성 | **2개** | 70개 |

검색 점수는 소수점 4자리까지 일치했다. 같은 질문에 양쪽 다 0.8457 / 0.8424 / 0.8421이었다.

케이스별 비교:

```bash
.venv\Scripts\python.exe scripts\compare_eval.py ^
  eval\raw\m4-2-ours.cases.json eval\raw\m4-2-llamaindex.cases.json
```

판정과 근거는 `MEMORY.md` D-036에 있다.

## 한계 — 프롬프트를 맞추지 못했다

`run_eval.py`의 `build_prompt`를 다시 짜면서 우리 것(`src/application/answering.py:145`)과 다르게 만들었다.

| | 직접 구현 | 여기 |
|---|---|---|
| 머리말 | `제공된 문단:` 있음 | 없음 |
| 문단 표시 | `[문단 N] <청크ID, 피부유형, 부위>` | `[근거 N]` |
| 끝 | `답변:` 유도 있음 | 없음 |

지시문이 "제공된 문단에 없는 내용은…"이라고 말하는데 여기에는 그 머리말이 없다. **변수를 하나만 두겠다고 위에 적어 놓고 둘을 바꿨다.**

그래서 **거부 26건 대 5건 차이는 프레임워크 차이로 읽으면 안 된다.** F1은 양쪽 다 0.32라 품질 결론은 흔들리지 않는다. 거부율 비교는 이 실험에서 얻지 못했다.

다시 하려면 LlamaIndex 색인에 메타데이터(피부 유형, 부위)를 넣어 빌드하고 `build_prompt`를 그대로 불러 써야 한다. 색인 재빌드 약 7분, 평가 약 20분이다.
