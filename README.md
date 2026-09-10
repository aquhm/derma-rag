# derma-rag

AI Hub `문제성 피부 메이크업 추천 데이터`를 근거로 질문에 답하는 RAG 시스템. **학습용 프로젝트**이며 상용 서비스가 아니다.

## 무엇인가

문제성 피부(민감성·염증성·색소·조직변화)와 관련된 질문을 받아, 데이터셋 안에서 근거 문단을 찾고 그 근거만으로 답한다. 근거를 찾지 못하면 답하지 않는다.

모델을 새로 훈련하지 않는다. 검색과 프롬프트 조립으로 해결한다.

## 왜 이렇게 만드는가

- **학습이 목적이다.** 6단계가 코드에 그대로 보이는 것이 완성 속도보다 중요하다. 그래서 LangChain·LlamaIndex를 쓰지 않는다.
- **데이터에 정답지가 있다.** 질의응답 10,043쌍이 함께 제공되므로 정확도를 감이 아니라 숫자로 잰다.
- **주제가 의료에 인접하다.** 지어내기를 막는 장치를 처음부터 넣는다.

## 6단계

| 단계 | 하는 일 | 주기 |
|---|---|---|
| 1 | AI Hub 데이터 확보 | 1회 |
| 2 | 문서 파싱 · 청크 분할 | 데이터 변경 시 |
| 3 | 임베딩 변환 | 데이터 변경 시 |
| 4 | 벡터 색인 저장 | 데이터 변경 시 |
| 5 | 질문과 가까운 문단 검색 | 질문마다 |
| 6 | 찾은 근거만으로 답 생성 | 질문마다 |

## 기술 스택

- Python 3.12
- `requests` — Ollama HTTP API
- `numpy` — 코사인 유사도
- Ollama 로컬 — `nomic-embed-text`(임베딩, 768차원), `qwen2.5:3b`(생성)

벡터 DB를 쓰지 않는다. 9,377건 × 768차원 float32 = 약 28MB이므로 numpy 배열 전수 검색으로 충분하다.

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
```

`--aihub`는 `data/raw`와 `index/aihub.npz`를 고른다. 명령줄에 원본 경로를 적으면 `scripts/guard_data.py` 훅이 막기 때문이다.

## 디렉터리

```
derma-rag/
├── CLAUDE.md              세션마다 읽히는 기본 규칙
├── MEMORY.md              결정 기록 (D-###)
├── ERRORS.md              실패 기록
├── .claude/
│   ├── rules/             경로별 규칙
│   ├── skills/            반복 절차 매뉴얼
│   ├── agents/            역할별 subagent
│   └── commands/          자주 쓰는 slash command
├── docs/
│   ├── PRD.md             무엇을 왜 만드는가
│   ├── ARCHITECTURE.md    레이어와 의존 규칙
│   ├── HARNESS.md         하네스 구성 설명
│   ├── ROADMAP.md         마일스톤
│   ├── DATASETS.md        데이터 출처와 라이선스
│   ├── PACKAGES.md        의존성 대장
│   ├── guidelines/        레이어별 코딩 지침
│   └── process/           SRS · RISK · SAFETY · TRACEABILITY
├── src/                   구현 (아직 없음)
├── scripts/               검사 스크립트
├── data/                  원본 데이터 (git 제외)
└── eval/                  평가 결과
```

## 데이터 라이선스

AI Hub 데이터는 **원본 재배포가 금지**된다. 가공본 공유도 금지다. 학습 결과물(모델·서비스)은 영리·비영리 자유이며, 이때 데이터셋 정식 명칭과 `aihub.or.kr`을 출처로 표기해야 한다.

`data/`는 `.gitignore` 대상이다. 절대 커밋하지 않는다.

자세한 내용은 `docs/DATASETS.md`.

## 안전

의료에 인접한 주제다. 진단 표현 생성 금지, 근거 없는 답변 금지, 면책·병원 안내 고정 부착이 강제 규칙이다. 전문은 `docs/process/SAFETY.md`.

## 출처

- 데이터: 문제성 피부 메이크업 추천 데이터 (AI Hub, `aihub.or.kr`, 저작권자 ㈜데이터쿡)
- 하네스 설계: 『하네스 엔지니어링 백과사전』 제7장 「Claude에서 하네스 구축하기」 (wikidocs.net/346799)
