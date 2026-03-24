from pathlib import Path


UI_PATH = Path(__file__).parent / "templates" / "index.html"


def ui_html() -> str:
    return UI_PATH.read_text(encoding="utf-8")
