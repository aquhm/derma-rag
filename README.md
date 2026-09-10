# derma-rag

AI Hub `문제성 피부 메이크업 추천 데이터`를 근거로 질문에 답하는 RAG 시스템입니다. **학습용 프로젝트**이며 상용 서비스가 아닙니다.

RAG 6단계를 프레임워크 없이 직접 구현했습니다. 의존성은 `requests`와 `numpy` 둘뿐입니다.

## 무엇인가

문제성 피부(민감성·염증성·색소·조직변화)와 관련된 질문을 받아, 데이터셋 안에서 근거를 찾고 그 근거만으로 답합니다. 근거를 찾지 못하면 답하지 않습니다.

모델을 새로 훈련하지 않습니다. 검색과 프롬프트 조립으로 해결합니다.

## 왜 이렇게 만들었나

- **학습이 목적입니다.** 6단계가 코드에 그대로 보이는 것이 완성 속도보다 중요합니다. 그래서 LangChain·LlamaIndex를 쓰지 않습니다 (`MEMORY.md` D-003). 이 판단은 M4-2에서 실제로 측정해 확인했습니다 (D-036).
- **데이터에 정답지가 있습니다.** 질의응답 쌍이 함께 제공되므로 정확도를 감이 아니라 숫자로 잽니다.
- **주제가 의료에 인접합니다.** 지어내기를 막는 장치를 처음부터 넣었습니다.

## 6단계

| 단계 | 하는 일 | 주기 | 구현 |
|---|---|---|---|
| 1 | 원본을 읽어 문서로 | 데이터 변경 시 | `infrastructure/aihub_loader.py` |
| 2 | 청크 분할 | 데이터 변경 시 | `infrastructure/chunking.py` |
| 3 | 임베딩 변환 | 데이터 변경 시 | `infrastructure/embedding.py` |
| 4 | 벡터 색인 저장 | 데이터 변경 시 | `infrastructure/index.py` |
| 5 | 질문과 가까운 근거 검색 | 질문마다 | `infrastructure/index.py` |
| 6 | 찾은 근거만으로 답 생성 | 질문마다 | `infrastructure/generation.py` |

조립은 `src/composition.py` 한 곳에서만 합니다. 유스케이스는 `src/application/`에, 판정 규칙은 `src/domain/`에 있습니다.

## 현재 구성

| 항목 | 값 | 근거 |
|---|---|---|
| 색인 재료 | QA쌍 (질문 + 답변) 9,031건 | D-032 |
| 청킹 | QA는 쪼개지 않는다 | D-032, `ERRORS.md` E-004 |
| 임베딩 | `bge-m3` (1024차원) | D-005 |
| 생성 | `qwen2.5:3b`, temperature 0, `num_ctx` 8192 | D-033 |
| 검색 | 상위 3개, 유사도 0.35 이상 | D-030 |
| 안전 게이트 | 생성 전 1개 + 생성 후 5개 규칙 | `docs/process/SAFETY.md` |

벡터 DB를 쓰지 않습니다. 9,031건 × 1024차원 float32가 약 37MB이므로 numpy 전수 검색으로 충분합니다 (D-004).

## 측정 결과

답변 정확도는 음절 2-gram F1입니다 (D-026). 표본 200건, seed 42, **평가 케이스는 색인에서 제외**해 측정했습니다.

| 시점 | F1 | 비고 |
|---|---|---|
| 기준선 (M2-8) | 0.237 | 근거 발췌 색인 |
| 최종 | **0.321** | QA쌍 색인 |

사람이 직접 질문해 채점한 22건은 모두 쓸 만한 답변이었습니다. **F1은 실제 품질보다 비관적입니다** — 같은 뜻을 다른 말로 답하면 벌점을 줍니다 (D-034).

### 실험 이력

한 번에 하나만 바꿔 측정했습니다.

| 실험 | 바꾼 것 | F1 변화 | 판정 |
|---|---|---|---|
| M3-2 | 메타데이터 필터링 | +0.0001 | 기각 (D-031) |
| M3-3 | 근거 개수 5 → 3 | +0.0053 | 채택 (D-030) |
| M3-4 | 프롬프트에 길이 제약 | -0.0031 | F1 기준 기각, 생성 실패 감소로 유지 |
| M3-5 | 생성 모델 3B → 14B | -0.0361 | 기각 (D-033) |
| M3-6 | 색인 재료를 QA쌍으로 | **+0.0766** | 채택 (D-032) |
| M3-7 | 논문 원문 색인 | -0.0809 | 기각 (D-035) |
| M4-2 | LlamaIndex로 재구현 | -0.0028 | 차이 없음. 직접 구현 유지 (D-036) |

**검색 파라미터를 아무리 만져도 ±0.005 안에서 움직였습니다.** 색인에 무엇을 넣느냐를 바꾸자 한 번에 올랐습니다.

`Recall@k`는 이 데이터셋에서 쓸 수 없다고 판정했습니다. 주석자가 답변을 뒷받침하려고 근거를 붙여서, 질문에서 근거까지가 두 홉입니다 (D-029).

## 기술 스택

- Python 3.12
- `requests` — Ollama HTTP API
- `numpy` — 코사인 유사도
- Ollama 로컬 — `bge-m3`(임베딩), `qwen2.5:3b`(생성)

외부 유료 API를 쓰지 않습니다. 전부 로컬에서 돕니다.

## 시작하기

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Ollama 모델 확인
ollama list

# 1~4단계 (샘플 데이터)
python -m src.cli ingest

# 1~4단계 (AI Hub 실데이터). 경로는 조립부에 있다
python -m src.cli ingest --aihub

# 중단된 적재를 이어서
python -m src.cli ingest --aihub --resume

# 5~6단계
python -m src.cli ask "민감성 피부인데 볼 부위 커버는 어떻게 하나요" --aihub

# 채점 (표본 200건, seed 42). --skip-answers면 검색만
python -m src.cli eval --aihub --out eval/<이름>.json --change "바꾼 것 하나"

# 로컬 웹 UI (127.0.0.1:8765). 사람이 직접 질문하고 채점한다
python -m src.cli serve
```

`--aihub`는 원본 폴더와 기본 색인을 고릅니다. 명령줄에 원본 경로를 적으면 `scripts/guard_data.py` 훅이 막습니다.

### 검사

```bash
pytest -q                       # 전체 (490건)
pytest -q -m "not integration"  # Ollama 없이
python scripts/check_layers.py  # 레이어 의존 규칙
```

## 디렉터리

```
derma-rag/
├── CLAUDE.md              세션마다 읽히는 기본 규칙
├── MEMORY.md              결정 기록 (D-001 ~ D-036)
├── ERRORS.md              실패 기록 (E-001 ~ E-004)
├── .claude/
│   ├── rules/             경로별 규칙
│   ├── skills/            반복 절차 매뉴얼
│   ├── agents/            역할별 subagent
│   └── commands/          자주 쓰는 slash command
├── docs/
│   ├── PRD.md             무엇을 왜 만드는가
│   ├── ARCHITECTURE.md    레이어와 의존 규칙
│   ├── HARNESS.md         하네스 구성 설명
│   ├── ROADMAP.md         마일스톤과 실험 기록
│   ├── DATASETS.md        데이터 출처와 라이선스
│   ├── PACKAGES.md        의존성 대장
│   ├── guidelines/        레이어별 코딩 지침
│   └── process/           SRS · RISK · SAFETY · TRACEABILITY
├── src/
│   ├── domain/            값 객체와 판정 규칙 (표준 라이브러리만)
│   ├── application/       유스케이스와 포트
│   ├── infrastructure/    어댑터 (Ollama, numpy, 파일 판독)
│   ├── interface/         CLI와 로컬 웹 UI
│   └── composition.py     조립. 전 레이어를 아는 유일한 파일
├── experiments/           비교 실험 (제품 아님)
├── scripts/               검사·조사 스크립트
└── eval/                  평가 결과 (집계만 커밋)
```

원본 폴더, 색인 산출물, 케이스별 평가 로그, 생성된 검사 보고서는 커밋하지 않습니다.

## 눈여겨볼 만한 것

- **`src/infrastructure/doc_reader.py`** — 구형 워드(`.doc`)를 표준 라이브러리만으로 읽습니다. 복합 파일(OLE2) 판독과 Word 조각표 파싱을 직접 구현했습니다. 실측 6,237건 전량 추출에 성공했고 새 의존성은 0개입니다.
- **`src/application/ports.py`** — 포트를 `typing.Protocol`로 두어 의존을 역전시켰습니다. `@runtime_checkable`은 붙이지 않습니다.
- **`src/domain/policy.py`** — 안전 규칙이 값 객체입니다. 임계값 조정이 코드 수정이 아니라 값 변경이 됩니다.
- **`scripts/check_layers.py`** — 레이어 의존 위반을 훅으로 잡습니다. `domain`은 `numpy`와 `requests`조차 import 할 수 없습니다.
- **`experiments/m4-2-llamaindex/`** — 같은 기능을 LlamaIndex로 다시 만들어 비교한 기록입니다. 별도 가상환경에서만 돕니다.

## 데이터 라이선스

AI Hub 데이터는 **원본 재배포가 금지**됩니다. 가공본 공유도 금지입니다. 학습 결과물(모델·서비스)은 영리·비영리 자유이며, 이때 데이터셋 정식 명칭과 `aihub.or.kr`을 출처로 표기해야 합니다.

원본 폴더는 `.gitignore` 대상입니다. 절대 커밋하지 않습니다.

자세한 내용은 `docs/DATASETS.md`에 있습니다.

## 안전

의료에 인접한 주제입니다. 다음이 강제 규칙입니다.

1. 근거가 부족하면 **생성 모델을 호출하지 않습니다.** 호출하면 지어냅니다.
2. 생성 결과에 진단·치료 단정이 섞였는지 검사합니다. 프롬프트 지시만으로는 막히지 않습니다.
3. 모든 답변에 면책 문구와 출처를 붙입니다. 조건부로 붙이지 않습니다.
4. 사용자 질문과 답변 원문을 저장하지 않습니다. 건강 정보일 수 있습니다.

전문은 `docs/process/SAFETY.md`에 있습니다.

## 출처

- 데이터: 문제성 피부 메이크업 추천 데이터 (AI Hub, `aihub.or.kr`, 저작권자 ㈜데이터쿡)
- 하네스 설계: 『하네스 엔지니어링 백과사전』 제7장 「Claude에서 하네스 구축하기」 (wikidocs.net/346799)
