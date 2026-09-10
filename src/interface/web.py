"""로컬 웹 UI. 질문을 치고 답변·근거를 보고 O/△/X를 남긴다.

표준 라이브러리 `http.server`만 쓴다. 새 의존성을 넣지 않는 이유는 기술 스택이
고정돼 있기 때문이다 (CLAUDE.md 5절). Gradio나 FastAPI를 쓰면 코드는 줄지만
HTTP 계층이 감춰진다. 학습 목적에는 얇게 드러나는 편이 낫다.

**127.0.0.1에만 바인딩한다.** 0.0.0.0으로 열면 같은 네트워크의 다른 기기가
AI Hub 원문이 섞인 답변을 받아 갈 수 있다 (SAFETY.md S-6).

**판정 기록에 질문과 답변을 남기지 않는다.** 판정, 시각, 근거 청크 ID, 유사도,
답변 길이만 적는다. 질문은 개인 건강 정보일 수 있다 (S-7).
"""

from __future__ import annotations

import json
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

from src.domain.models import Query, Refusal, Result
from src.domain.policy import DISCLAIMER

# 바깥에서 접속할 수 없게 루프백에만 연다.
HOST = "127.0.0.1"
DEFAULT_PORT = 8765

VERDICTS = ("O", "△", "X")

# 질문 유형. 판정만으로는 "정상 질문인데 거부했다"와 "무관한 질문이라 거부했다"가
# 구분되지 않는다. 실사용 확인 30건에서 이 구분이 없어 결론을 못 냈다 (D-034).
# 유형은 분류일 뿐 질문 원문이 아니므로 기록해도 S-7에 걸리지 않는다.
QUESTION_KINDS = ("정상", "무관", "진단요구", "미분류")

# 요청 본문 상한. 질문 하나에 이보다 클 이유가 없다.
MAX_BODY_BYTES = 64 * 1024


def answer_payload(result: Result) -> dict[str, Any]:
    """Answer 또는 Refusal을 화면에 그릴 형태로 바꾼다."""
    if isinstance(result, Refusal):
        return {
            "kind": "refusal",
            "reason": result.reason.value,
            "detail": result.detail,
            "evidence": [],
        }

    return {
        "kind": "answer",
        "body": result.body,
        "disclaimer": DISCLAIMER,
        "evidence": [
            {
                "id": item.chunk.id,
                "score": round(item.score, 4),
                "text": item.chunk.text,
                "skin_type": (
                    None if item.chunk.skin_type is None else item.chunk.skin_type.value
                ),
                "skin_detail": item.chunk.skin_detail,
                "area": item.chunk.area,
            }
            for item in result.evidence
        ],
    }


def judgment_row(
    verdict: str, payload: dict[str, Any], question_kind: str = "미분류"
) -> dict[str, Any]:
    """사람 판정 한 줄. 질문도 답변도 담지 않는다 (SAFETY.md S-7).

    근거 청크 ID와 유사도는 남긴다. 나중에 "낮은 유사도에서 X가 몰리는가" 같은
    것을 셀 수 있어야 한다. ID는 원문이 아니다.
    """
    if verdict not in VERDICTS:
        raise ValueError(f"판정은 {VERDICTS} 중 하나여야 한다: {verdict}")
    if question_kind not in QUESTION_KINDS:
        raise ValueError(f"질문 유형은 {QUESTION_KINDS} 중 하나여야 한다: {question_kind}")

    row: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "verdict": verdict,
        "question_kind": question_kind,
        "kind": payload.get("kind", ""),
        "chunk_ids": [item["id"] for item in payload.get("evidence", [])],
        "scores": [item["score"] for item in payload.get("evidence", [])],
        "body_length": len(payload.get("body", "")),
    }
    if payload.get("kind") == "refusal":
        row["reason"] = payload.get("reason", "")
    return row


def append_judgment(path: Path, row: dict[str, Any]) -> None:
    """한 줄에 한 건씩 덧붙인다. 중간에 끊겨도 앞부분은 온전하다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


PAGE = """<!doctype html>
<html lang="ko">
<meta charset="utf-8">
<title>derma-rag 로컬 확인</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 46rem; margin: 2rem auto;
         padding: 0 1rem; line-height: 1.7; color: #1a1a1a; }
  h1 { font-size: 1.25rem; }
  .warn { background: #fff4e5; border-left: 4px solid #e08600; padding: .75rem 1rem;
          font-size: .9rem; }
  textarea { width: 100%; height: 5rem; font: inherit; padding: .5rem; }
  button { font: inherit; padding: .4rem .9rem; cursor: pointer; }
  .row { display: flex; gap: .5rem; align-items: center; margin: .75rem 0; }
  .card { border: 1px solid #ddd; border-radius: 6px; padding: 1rem; margin: 1rem 0; }
  .refusal { border-color: #d33; background: #fff5f5; }
  .ev { background: #fafafa; border-left: 3px solid #bbb; padding: .5rem .75rem;
        margin: .5rem 0; font-size: .88rem; white-space: pre-wrap; }
  .meta { color: #666; font-size: .8rem; }
  .disc { color: #666; font-size: .82rem; white-space: pre-wrap; margin-top: 1rem;
          border-top: 1px dashed #ccc; padding-top: .6rem; }
  #tally { color: #444; font-size: .85rem; }
</style>

<h1>derma-rag 로컬 확인</h1>
<p class="warn">이 답변은 데이터셋을 검색한 결과이며 의학적 진단이나 처방이 아닙니다.
판정 기록에는 질문과 답변 원문을 남기지 않습니다. 판정·시각·근거 ID·유사도만 저장합니다.</p>

<div class="row">
  <label>색인
    <select id="index">
      <option value="qa">QA쌍 (기본)</option>
      <option value="excerpt">근거 발췌</option>
    </select>
  </label>
  <label>질문 유형
    <select id="kind">
      <option value="정상">정상 (답이 나와야 함)</option>
      <option value="무관">무관 (거부해야 함)</option>
      <option value="진단요구">진단요구 (거부해야 함)</option>
      <option value="미분류">미분류</option>
    </select>
  </label>
  <span class="meta" id="status"></span>
</div>

<textarea id="q" placeholder="질문을 입력하십시오. Ctrl+Enter로 전송"></textarea>
<div class="row">
  <button id="ask">질문</button>
  <span id="tally"></span>
</div>

<div id="out"></div>

<script>
let last = null;
const tally = { "O": 0, "\\u25b3": 0, "X": 0 };

function drawTally() {
  document.getElementById("tally").textContent =
    "판정 O " + tally["O"] + " / \\u25b3 " + tally["\\u25b3"] + " / X " + tally["X"];
}

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
}

function renderError(message) {
  const box = el("div", "card refusal");
  box.append(el("strong", null, "오류"), el("div", null, message));
  document.getElementById("out").append(box);
}

function render(data) {
  const out = document.getElementById("out");
  const box = el("div", data.kind === "refusal" ? "card refusal" : "card");

  if (data.kind === "refusal") {
    box.append(el("strong", null, "답할 수 없습니다"));
    box.append(el("div", null, "사유: " + data.reason + " — " + data.detail));
  } else {
    box.append(el("div", null, data.body));
    box.append(el("div", "meta", "근거 " + data.evidence.length + "건"));
    data.evidence.forEach(function (ev, i) {
      const labels = [ev.skin_type, ev.skin_detail, ev.area].filter(Boolean).join(" · ");
      const node = el("div", "ev");
      node.append(el("div", "meta",
        "[" + (i + 1) + "] " + ev.id + " · 유사도 " + ev.score + (labels ? " · " + labels : "")));
      node.append(el("div", null, ev.text));
      box.append(node);
    });
    box.append(el("div", "disc", data.disclaimer));
  }

  const row = el("div", "row");
  row.append(el("span", "meta", "판정:"));
  ["O", "\\u25b3", "X"].forEach(function (v) {
    const button = el("button", null, v);
    button.onclick = function () { judge(v, row); };
    row.append(button);
  });
  box.append(row);
  out.append(box);
}

async function ask() {
  const question = document.getElementById("q").value.trim();
  if (!question) return;
  const status = document.getElementById("status");
  status.textContent = "생성 중...";
  document.getElementById("out").innerHTML = "";
  try {
    const res = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: question, index: document.getElementById("index").value })
    });
    const data = await res.json();
    status.textContent = "";
    if (data.error) { renderError(data.error); return; }
    last = data;
    render(data);
  } catch (err) {
    status.textContent = "";
    renderError(String(err));
  }
}

async function judge(verdict, row) {
  if (!last) return;
  await fetch("/judge", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      verdict: verdict,
      payload: last,
      question_kind: document.getElementById("kind").value
    })
  });
  tally[verdict] += 1;
  drawTally();
  row.replaceWith(el("div", "meta", "기록함: " + verdict));
}

document.getElementById("ask").onclick = ask;
document.getElementById("q").addEventListener("keydown", function (e) {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) ask();
});
drawTally();
</script>
</html>
"""


def build_handler(
    answerers: dict[str, Any], log_path: Path
) -> type[BaseHTTPRequestHandler]:
    """요청 처리기를 만든다. 유스케이스는 미리 조립해 넘긴다.

    interface는 어댑터를 직접 만들지 않는다 (docs/guidelines/04-interface.md).
    """

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - http.server 규약
            if self.path not in ("/", "/index.html"):
                self._send(404, {"error": "없는 경로다"})
                return
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:  # noqa: N802 - http.server 규약
            payload = self._read_json()
            if payload is None:
                return

            if self.path == "/ask":
                self._handle_ask(payload)
            elif self.path == "/judge":
                self._handle_judge(payload)
            else:
                self._send(404, {"error": "없는 경로다"})

        def _handle_ask(self, payload: dict[str, Any]) -> None:
            question = str(payload.get("question", "")).strip()
            if not question:
                self._send(400, {"error": "질문이 비어 있다"})
                return

            answerer = answerers.get(str(payload.get("index", "qa")))
            if answerer is None:
                self._send(400, {"error": "그 색인은 준비돼 있지 않다"})
                return

            try:
                result = answerer.execute(Query(question))
            except Exception as exc:  # noqa: BLE001 - 화면에 알리고 서버는 계속 띄운다
                # 메시지에 질문을 넣지 않는다 (S-7).
                self._send(200, {"error": f"생성 실패: {type(exc).__name__}"})
                return

            self._send(200, answer_payload(result))

        def _handle_judge(self, payload: dict[str, Any]) -> None:
            try:
                row = judgment_row(
                    str(payload.get("verdict", "")),
                    payload.get("payload") or {},
                    question_kind=str(payload.get("question_kind", "미분류")),
                )
            except ValueError as exc:
                self._send(400, {"error": str(exc)})
                return

            append_judgment(log_path, row)
            self._send(200, {"ok": True})

        def _read_json(self) -> dict[str, Any] | None:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > MAX_BODY_BYTES:
                self._send(400, {"error": "요청 본문 크기가 잘못됐다"})
                return None
            try:
                return json.loads(self.rfile.read(length).decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                self._send(400, {"error": "JSON이 아니다"})
                return None

        def _send(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            # 기본 접근 로그를 끈다. 경로만 찍히지만 조용한 편이 낫다.
            return

    return Handler


def serve(answerers: dict[str, Any], log_path: Path, port: int = DEFAULT_PORT) -> None:
    """서버를 띄운다. Ctrl+C로 멈춘다."""
    server = HTTPServer((HOST, port), build_handler(answerers, log_path))
    print(f"http://{HOST}:{port} 에서 열렸다. 판정 기록: {log_path}")
    print("멈추려면 Ctrl+C.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n종료한다.")
    finally:
        server.server_close()
