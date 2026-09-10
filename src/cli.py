"""python -m src.cli 진입점.

본체는 src/interface/cli.py에 있다. 이 파일을 따로 두는 이유는 두 문서가 서로
다른 것을 요구하기 때문이다. CLAUDE.md와 README.md는 `python -m src.cli`를
명령으로 적었고, docs/ARCHITECTURE.md 4절과 레이어 규칙은 진입점이
src/interface/ 아래 있어야 한다고 정한다. 얇은 진입점 하나로 둘 다 지킨다
(MEMORY.md D-023).
"""

from __future__ import annotations

from src.interface.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
