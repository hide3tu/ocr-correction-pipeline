"""E2E test: run the WebUI, enter text, execute correction, verify download files."""

from __future__ import annotations

import csv
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest
from playwright.sync_api import sync_playwright

WEBUI_PORT = 7865
WEBUI_URL = f"http://127.0.0.1:{WEBUI_PORT}"
TEST_INPUT = "遺伝字の研究は重要です。"


def _start_server():
    """Start the Gradio WebUI server and wait until ready."""
    proc = subprocess.Popen(
        [sys.executable, "-m", "ocr_corrector", "--webui", "--port", str(WEBUI_PORT)],
        cwd=str(Path(__file__).resolve().parent.parent),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    deadline = time.time() + 120
    while time.time() < deadline:
        try:
            import urllib.request
            urllib.request.urlopen(WEBUI_URL, timeout=2)
            return proc
        except Exception:
            time.sleep(2)

    out = proc.stdout.read().decode(errors="replace") if proc.stdout else ""
    proc.kill()
    raise RuntimeError(f"WebUI server did not start within 120s.\nOutput:\n{out}")


def _stop_server(proc):
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.fixture(scope="class")
def server_and_browser():
    """Per-class server + browser to avoid cross-class state leaking."""
    proc = _start_server()
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        yield proc, browser
        browser.close()
    _stop_server(proc)


RESULT_FILE_NAMES = {"ocr_raw.txt", "corrections.csv",
                     "corrected_bert.txt", "corrected_llm.txt", "corrected_all.txt",
                     "searchable.pdf"}


def _fresh_page(browser):
    """Create a fresh browser context + page to avoid state leaking."""
    ctx = browser.new_context(accept_downloads=True)
    page = ctx.new_page()
    page.goto(WEBUI_URL, wait_until="networkidle")
    return ctx, page


def _uncheck_llm(page):
    cb = page.locator("input[type='checkbox']").first
    if cb.is_checked():
        cb.click()
        time.sleep(0.5)
    # Verify it's actually unchecked; retry once if needed
    if cb.is_checked():
        cb.click()
        time.sleep(0.5)
    assert not cb.is_checked(), "Failed to uncheck LLM checkbox"


def _wait_for_status(page, marker: str, timeout_s: int = 180):
    """Poll status textarea until it contains `marker`."""
    status_ta = page.get_by_label("ステータス")
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        val = status_ta.input_value()
        if marker in val:
            return val
        time.sleep(1)
    raise TimeoutError(f"Status did not contain '{marker}' within {timeout_s}s. Last: {status_ta.input_value()!r}")


def _get_download_links(page):
    """Return download link elements for pipeline result files only."""
    all_links = page.locator("a[download]").all()
    return [a for a in all_links
            if (a.get_attribute("download") or "") in RESULT_FILE_NAMES]


# ---------- Text input tests ----------

class TestTextInput:

    def test_bert_only_produces_downloads(self, server_and_browser):
        ctx, page = _fresh_page(server_and_browser[1])
        try:
            _uncheck_llm(page)
            page.get_by_label("OCRテキスト").fill(TEST_INPUT)
            page.get_by_role("button", name="校正実行").click()

            status = _wait_for_status(page, "完了 |")
            assert "完了" in status

            dl_links = _get_download_links(page)
            dl_names = [a.get_attribute("download") for a in dl_links]
            print(f"Download links: {dl_names}")

            assert any("corrections.csv" in (n or "") for n in dl_names)
            assert any("corrected_bert.txt" in (n or "") for n in dl_names)
            assert not any("corrected_llm.txt" in (n or "") for n in dl_names)
        finally:
            ctx.close()

    def test_download_csv_content(self, server_and_browser):
        ctx, page = _fresh_page(server_and_browser[1])
        try:
            _uncheck_llm(page)
            page.get_by_label("OCRテキスト").fill(TEST_INPUT)
            page.get_by_role("button", name="校正実行").click()
            _wait_for_status(page, "完了 |")

            csv_link = None
            for a in _get_download_links(page):
                if (a.get_attribute("download") or "") == "corrections.csv":
                    csv_link = a
                    break
            assert csv_link is not None

            with page.expect_download() as dl_info:
                csv_link.click()
            path = dl_info.value.path()
            content = Path(path).read_text(encoding="utf-8-sig")
            rows = list(csv.reader(content.splitlines()))
            assert len(rows) >= 1
            assert "BERT確率" in rows[0][3]
            print(f"CSV rows: {len(rows)}")
        finally:
            ctx.close()

    def test_corrected_text_preserves_newlines(self, server_and_browser):
        ctx, page = _fresh_page(server_and_browser[1])
        try:
            _uncheck_llm(page)
            page.get_by_label("OCRテキスト").fill(TEST_INPUT)
            page.get_by_role("button", name="校正実行").click()
            _wait_for_status(page, "完了 |")

            bert_link = None
            for a in _get_download_links(page):
                if (a.get_attribute("download") or "") == "corrected_bert.txt":
                    bert_link = a
                    break
            assert bert_link is not None

            with page.expect_download() as dl_info:
                bert_link.click()
            content = Path(dl_info.value.path()).read_text(encoding="utf-8")
            assert content.strip()
            assert content.count("\n") == TEST_INPUT.count("\n")
            print(f"Corrected text: {content.strip()!r}")
        finally:
            ctx.close()


# ---------- Multi-image tests ----------

def _create_test_images(n: int = 2) -> list[str]:
    from PIL import Image, ImageDraw
    tmpdir = tempfile.mkdtemp(prefix="ocr_test_")
    paths = []
    for i in range(n):
        img = Image.new("RGB", (400, 100), "white")
        draw = ImageDraw.Draw(img)
        draw.text((10, 30), f"TEST PAGE {i + 1}", fill="black")
        p = os.path.join(tmpdir, f"page_{i + 1}.png")
        img.save(p)
        paths.append(p)
    return paths


class TestMultiImage:

    def test_multi_image_produces_downloads(self, server_and_browser):
        ctx, page = _fresh_page(server_and_browser[1])
        try:
            _uncheck_llm(page)

            # Switch to multi-image tab
            tabs = page.locator("button[role='tab']").all()
            tabs[2].click()
            time.sleep(0.5)

            # Upload test images
            test_images = _create_test_images(2)
            page.locator("input[type='file'][multiple]").set_input_files(test_images)
            time.sleep(1)

            # Click multi-image run button
            page.locator("#run-btn-multi").click()

            # Wait for completion
            status = _wait_for_status(page, "完了 (2画像)", timeout_s=300)
            assert "2画像" in status

            # Verify downloads
            dl_links = _get_download_links(page)
            dl_names = [a.get_attribute("download") for a in dl_links]
            print(f"Multi-image download links: {dl_names}")

            assert any("ocr_raw.txt" in (n or "") for n in dl_names)
            assert any("corrections.csv" in (n or "") for n in dl_names)
            assert any("corrected_bert.txt" in (n or "") for n in dl_names)
        finally:
            ctx.close()
