from pathlib import Path

UI_PATH = Path(__file__).parent / "templates" / "ui.html"


def ui_html() -> str:
    return UI_PATH.read_text(encoding="utf-8")
