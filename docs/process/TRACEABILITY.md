# TRACEABILITY

요구사항이 어디에 구현되고 어디서 검증되는지 잇는다. 구현이 진행되면 채운다.

## 기능 요구사항 추적

| ID | 요구사항 | 레이어 | 구현 위치 | 테스트 | 상태 |
|---|---|---|---|---|---|
| FR-1 | 데이터 적재 | infrastructure | `loaders.py` + `ingest.py` | `tests/infrastructure/test_loaders.py` | 완료 (M1-3, 샘플) |
| FR-2 | 청크 분할 | infrastructure | `chunking.py` | `tests/infrastructure/test_chunking.py` | 완료 (M1-3) |
| FR-3 | 임베딩 | infrastructure | `embedding.py` + `indexing.py` | `tests/infrastructure/test_embedding.py` | 완료 (M1-4) |
| FR-4 | 색인 저장·로드 | infrastructure | `index.py` | `tests/infrastructure/test_index.py` | 완료 (M1-5) |
| FR-5 | 검색 | infrastructure | `index.py` | `tests/infrastructure/test_index.py` | 완료 (M1-5) |
| FR-6 | 근거 충분성 판정 | domain | `policy.py` `RetrievalPolicy` | `tests/domain/test_policy.py` | 완료 (M1-6) |
| FR-7 | 답변 생성 | application | `answering.py` + `generation.py` | `tests/application/test_answering.py` | 완료 (M1-6) |
| FR-8 | 출력 안전 검사 | domain + application | `policy.py`, `answering.py` | 필수 테스트 2 | 완료 (M1-7) |
| FR-9 | 면책 문구 부착 | domain | `policy.py` `DISCLAIMER` + `Answer.text` | 필수 테스트 3 | 완료 (M1-7) |
| FR-10 | 근거 제시 | interface | `cli.py` `format_answer` | `tests/interface/test_cli.py` | 완료 (M1-8) |
| FR-11 | 거부 사유 표시 | interface | `cli.py` `format_refusal` | `tests/interface/test_cli.py` | 완료 (M1-8) |
| FR-12 | 평가 | application | `evaluation.py` + `domain/scoring.py` | `tests/application/test_evaluation.py` | 완료 (Recall@k, 거부율, 답변 정확도) |
| FR-13 | CLI | interface | `cli.py` + `composition.py` | `tests/interface/` | 완료 (M1-8, eval 제외) |

## 비기능 요구사항 추적

| ID | 요구사항 | 강제 수단 | 상태 |
|---|---|---|---|
| NFR-1 | 오프라인 동작 | `docs/PACKAGES.md` 의존성 통제 | 규칙 수립 |
| NFR-2 | 데이터 격리 | `.gitignore` + PreToolUse hook (`scripts/guard_data.py`) | 완료 |
| NFR-3 | 레이어 규칙 | `scripts/check_layers.py` + PostToolUse hook | 완료 |
| NFR-4 | 재현성 | 평가 반복 실행 | 미착수 |
| NFR-5 | 중단 복원 | FR-3 체크포인트 | 미착수 |
| NFR-6 | 인코딩 | `docs/guidelines/00-common.md` | 규칙 수립 |

## 안전 규칙 추적

| ID | 규칙 | 강제 수단 | 상태 |
|---|---|---|---|
| S-1 | 진단하지 않는다 | `SafetyPolicy` 금지 표현 목록 | 미구현 |
| S-2 | 근거 없으면 답하지 않는다 | `answering.py` 분기 + 필수 테스트 1 | 미구현 |
| S-3 | 근거 밖 내용 생성 금지 | 프롬프트 조립 규칙 | 미구현 |
| S-4 | 출력 검사 후 반환 | 두 번째 게이트 + 필수 테스트 2 | 미구현 |
| S-5 | 면책 문구 항상 부착 | `SafetyPolicy` + 필수 테스트 3 | 미구현 |
| S-6 | 원본 반출 금지 | `.gitignore`, PreToolUse hook, permissions deny | 완료 |
| S-7 | 질문 저장 안 함 | 기능 미구현으로 달성 | 달성 |
| S-8 | 이미지 기능 없음 | 기능 미구현으로 달성 | 달성 |

## 결정 추적

| 결정 | 영향받는 문서 |
|---|---|
| D-001 학습 목적 | PRD, ROADMAP, CLAUDE.md |
| D-002 RAG 선택 | PRD, ARCHITECTURE |
| D-003 프레임워크 금지 | PACKAGES, CLAUDE.md, RISK R-8 |
| D-004 벡터 DB 미사용 | ARCHITECTURE, PACKAGES |
| D-005 Ollama 로컬 | PACKAGES, RISK R-6 |
| D-006 데이터 반출 금지 | DATASETS, SAFETY S-6, RISK R-2 |
| D-007 근거 없으면 생성 안 함 | SAFETY S-2, ARCHITECTURE §7 |
| D-008 QA쌍 평가 | PRD, ROADMAP M2-8 |
| D-009 레이어 분리 | ARCHITECTURE, guidelines |
| D-010 문서 우선 | ROADMAP M0 |

## 갱신 규칙

구현이 하나 끝날 때마다 해당 행의 상태를 갱신한다. `/handoff` 명령이 이 문서 갱신을 포함한다.
