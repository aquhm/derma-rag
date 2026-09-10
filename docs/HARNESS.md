# HARNESS

이 프로젝트의 Claude Code 하네스 구성. 『하네스 엔지니어링 백과사전』 제7장의 5계층 모델을 따른다.

## 왜 하네스를 만드는가

모델에게 무엇을 시킬지보다, 모델이 일하는 환경을 어떻게 설계할지가 결과를 좌우한다. 규칙을 매 세션 다시 설명하지 않고, 반드시 실행되어야 하는 검사를 사람의 기억에 맡기지 않기 위한 구성이다.

핵심 원칙 하나: **말로 부탁하는 규칙과 실제로 집행되는 규칙을 구분한다.** CLAUDE.md에 "레이어 규칙을 지켜라"라고 쓰는 것은 요청이다. hook으로 `check_layers.py`를 돌리는 것은 집행이다. 반드시 지켜야 하는 것은 집행 쪽으로 옮긴다.

## 5계층 구성

| 층 | 구성 요소 | 이 프로젝트에서 |
|---|---|---|
| 1층 · 기본 규칙 | `CLAUDE.md`, `.claude/rules/` | 6영역 규칙 + 경로별 지침 |
| 2층 · 업무 매뉴얼 | `.claude/skills/` | 반복 절차 4개 |
| 3층 · 자동 집행 | `.claude/settings.json` | permissions + hooks |
| 4층 · 역할 분담 | `.claude/agents/` | subagent 3개 |
| 5층 · 배포 | Plugin | **쓰지 않는다.** 단일 프로젝트라 불필요 |

MCP는 다섯 층 옆에 붙는 외부 연결 통로다. 이 프로젝트는 **MCP를 연결하지 않는다.** 로컬 파일과 로컬 Ollama만 쓰므로 외부 연결이 필요 없고, 연결하지 않는 것이 가장 안전하다.

### 검토했으나 도입하지 않은 MCP

`github.com/aihub-git/AIHub-MCP` — AI Hub 데이터셋 메타데이터를 자연어로 탐색하는 SpringAI 기반 MCP 서버.

| 항목 | 확인 결과 |
|---|---|
| 제공 도구 | `getDataSet`, `describeDataSet`, `searchDataSets`, `countDataSets`, `getDataSetsWithGuide` |
| 기능 범위 | **메타데이터 조회·검색만.** 실제 데이터 다운로드 불가 |
| 인증 | `AIHUB_API_KEY` 환경변수 필요 |
| 실행 | Java 17 + JAR |
| 라이선스 | **미기재** |
| 성숙도 | 커밋 4개, 스타 2개, 포크 1개 |

도입하지 않는 이유는 `MEMORY.md` D-011에 있다. 요약하면 **대상 데이터셋이 이미 확정돼 탐색이 필요 없고, 정작 필요한 다운로드는 이 MCP가 못 한다.** 제7장의 경고("third-party MCP의 correctness와 security는 직접 검증해야 하며, untrusted content를 가져오는 서버는 prompt injection 위험이 있다")도 함께 고려했다.

## 1층 · 항상 읽히는 규칙

### CLAUDE.md

여섯 영역으로 구성한다. 길게 쓰지 않는다. 긴 절차는 Skill로 뺀다.

| 영역 | 이 프로젝트의 핵심 |
|---|---|
| 대화 방식 | 결론부터. 불확실하면 "확인 필요" 표시 |
| 변경 통제 | 요청 범위만. 레이어 경계 넘는 변경은 ARCHITECTURE 확인 후 |
| 사용자·프로젝트 맥락 | Python·RAG 초심자. 학습이 목적 |
| 기억과 연속성 | 결정은 `MEMORY.md` D-###, 실패는 `ERRORS.md` E-### |
| 개발 작업 안전 | 스택 고정. LangChain·LlamaIndex 금지 |
| 고위험 행동 차단 | 원본 데이터 반출 금지. 진단 표현 생성 금지 |

### .claude/rules/

경로별로 다른 안내문을 둔다. 모든 규칙을 CLAUDE.md 한 장에 몰아넣으면 아무도 읽지 않는다.

| 파일 | 적용 경로 | 담는 것 |
|---|---|---|
| `domain.md` | `src/domain/**` | 외부 의존 금지, 값 객체 원칙 |
| `application.md` | `src/application/**` | 포트 정의 규칙, 유스케이스 단일 책임 |
| `infrastructure.md` | `src/infrastructure/**` | 어댑터 규칙, 외부 호출 처리 |
| `interface.md` | `src/interface/**` | CLI 출력 형식, 면책 문구 부착 |
| `tests.md` | `tests/**` | 목 대신 가짜 구현, 정상·실패 사례 병행 |
| `docs.md` | `docs/**` | 문서 톤, 근거 표기 |

## 2층 · 필요할 때 열리는 매뉴얼

Skill은 세션마다 항상 로드되지 않는다. 관련성이 있을 때만 열린다. 그래서 긴 절차를 넣기 좋다.

| Skill | 언제 열리는가 | 하는 일 |
|---|---|---|
| `ingest-stage` | 파이프라인 단계를 새로 만들거나 고칠 때 | 포트 정의 → 어댑터 구현 → 조립 → 테스트 순서 강제 |
| `eval-run` | 평가를 돌리거나 결과를 해석할 때 | Recall@k와 답변 정확도 분리 측정, 리포트 형식 고정 |
| `safety-audit` | 프롬프트나 생성 경로를 건드릴 때 | 두 게이트 존재 확인, 금지 표현 검사 |
| `process-docs` | 결정·실패·진행을 기록할 때 | MEMORY/ERRORS/ROADMAP 갱신 형식 |

각 Skill의 `description`은 소개문이 아니라 **트리거**로 쓴다. 사용자가 실제로 할 법한 표현을 넣고, "이럴 때는 쓰지 말 것"이라는 부정 경계도 함께 적는다.

## 3층 · 자동으로 집행되는 것

### permissions

| 도구 | 설정 | 이유 |
|---|---|---|
| `Read`, `Grep`, `Glob` | allow | 읽기는 안전하다 |
| `Edit`, `Write` | ask | 파일이 바뀐다 |
| 읽기 전용 셸 명령 | allow | `ls`, `find`, `tree`, `cat`, `head`, `tail`, `wc`, `git branch`. 파일을 바꾸지 않는다 |
| 그 밖의 `Bash` | ask | 강력하다 |
| 프로젝트 명령 | allow | `pytest`, `python -m src.cli`, `check_layers.py`, `guard_data.py`, `ollama list` |
| `pip install` | ask | 스택이 고정돼 있다 (`docs/PACKAGES.md`). `.venv/Scripts/python.exe -m pip` 경로도 함께 막는다 |
| 삭제·외부 전송·push | deny 또는 명시 승인 | 되돌리기 어렵다 |

읽기 전용 명령을 allow로 연 대신, `data/` 보호를 훅으로 옮겼다. `Read(data/**)`만 ask로 두면 `cat data/...`로 우회되기 때문이다.

### hooks

| 시점 | 실행 | 목적 |
|---|---|---|
| `PostToolUse` (Edit/Write) | `python scripts/check_layers.py --hook` | 레이어 의존 규칙 위반을 즉시 잡는다 |
| `PreToolUse` (Write/Edit/Bash) | `python scripts/guard_data.py` | 원본 데이터 디렉터리 접근 차단 |

`guard_data.py`는 두 가지를 본다. `Write`/`Edit`는 `file_path`를, `Bash`는 `command` 문자열을 검사한다. 명령은 `;` `|` `&&` `||` 로 쪼개 조각마다 판정한다.

| 대상 | 판정 |
|---|---|
| `data/` 아래 파일 쓰기 | 차단 |
| `cat`, `head`, `grep`, `cp` 등으로 `data/` 아래 접근 | 차단 |
| `ls`, `find`, `tree`, `du`, `stat`으로 목록 확인 | 통과. 내용을 읽지 않는다 |
| `find ... -exec`, `-delete` | 차단. 다른 명령을 대신 실행한다 |
| `data/` 디렉터리 자체, `data/README.md` | 통과 |

완전 차단은 아니다. `python -c "open('data/raw/x.json').read()"`처럼 경로가 따옴표 안에 들어가면 토큰 분리로 잡지 못한다. 실수 방지 수준으로 본다.

**hook을 많이 붙이지 않는다.** 모든 문에 경보를 달면 아무도 경보를 믿지 않게 된다. 위 두 개는 사람이 자주 빼먹고, 어겼을 때 비용이 큰 항목이라 자동화한다.

## 4층 · 역할을 나누는 담당자

Subagent의 핵심은 똑똑함이 아니라 **격리**다. 긴 로그와 여러 파일을 읽어야 하지만 최종 요약만 필요한 작업을 맡긴다.

| Agent | 도구 권한 | 맡는 일 |
|---|---|---|
| `reviewer` | Read, Grep, Glob | 변경 diff를 독립 관점으로 검토. 수정 권한 없음 |
| `safety-auditor` | Read, Grep, Glob | 생성 경로에 두 게이트가 살아 있는지 확인 |
| `researcher` | Read, Grep, Glob, WebSearch | 외부 자료 조사 후 요약만 보고 |

셋 다 **읽기 전용**이다. 조사와 검토는 수정 권한이 필요 없다.

**Agent Teams는 쓰지 않는다.** 이 프로젝트는 서로의 결정이 계속 영향을 주는 규모가 아니다. 독립 위임으로 충분하다.

## 운영 버튼 (slash command)

반복 요청을 짧은 명령으로 바꾼다.

| 명령 | 하는 일 |
|---|---|
| `/review` | 변경사항을 `reviewer` agent에 넘겨 독립 검토 |
| `/eval` | 평가 실행 후 기준선 대비 비교 |
| `/layer-check` | 레이어 규칙 수동 검사 |
| `/handoff` | 세션 종료 요약을 MEMORY.md와 ROADMAP.md에 반영 |

## 검토는 독립 세션에서

같은 세션에서 자기가 만든 것을 검토하면 앞서 세운 가정에 끌린다. 중요한 리뷰는 새 세션에서 하거나 `reviewer` agent에 맡긴다.

```
이 변경사항을 독립 리뷰어 관점으로 검토해줘.
구현 의도는 추측하지 말고, 실제 diff와 테스트 근거만 보고 위험을 찾아줘.
```

## Plan Mode를 쓰는 시점

여러 파일을 동시에 바꾸거나 구조를 변경할 때는 먼저 계획만 받는다.

```
먼저 계획만 세워줘. 어떤 파일을 읽고, 무엇을 바꿀지 설명한 뒤 내가 승인하면 진행해.
```

## 무엇을 어디에 둘지 판단하는 기준

| 질문 | 위치 |
|---|---|
| 항상 알아야 하는 규칙인가? | `CLAUDE.md` 또는 `.claude/rules/` |
| 가끔 필요한 긴 절차인가? | Skill |
| 반드시 자동 실행되어야 하는 검사인가? | Hook |
| 별도 담당자에게 맡겨도 되는 독립 작업인가? | Subagent |
| 여러 프로젝트에 배포할 것인가? | Plugin (이 프로젝트는 해당 없음) |
| 외부 도구에 연결해야 하는가? | MCP (이 프로젝트는 해당 없음) |

## 구축 순서

한 번에 전부 켜지 않는다. 반복되는 문제가 생길 때마다 맞는 층을 하나씩 추가한다.

1. `CLAUDE.md` — 완료
2. `.claude/rules/` — 완료
3. `.claude/skills/` — 완료
4. `.claude/agents/`, `.claude/commands/` — 완료
5. `.claude/settings.json` + `scripts/check_layers.py` + `scripts/guard_data.py` — 완료

다섯 층이 모두 채워졌다. 이제 규칙이 부탁이 아니라 집행이다.

## 출처

『하네스 엔지니어링 백과사전』 제7장 「Claude에서 하네스 구축하기」 — https://wikidocs.net/346799
