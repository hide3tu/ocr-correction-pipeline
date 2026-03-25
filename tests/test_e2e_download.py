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

WEBUI_PORT = 7865  # Use non-default port to avoid conflicts
WEBUI_URL = f"http://127.0.0.1:{WEBUI_PORT}"
# A short Japanese sentence with a deliberate OCR-like error for BERT to detect.
# "遺伝子" (gene) — we use "遺伝字" where 字 (character) is wrong, should be 子 (child).
TEST_INPUT = "遺伝字の研究は重要です。"


@pytest.fixture(scope="module")
def webui_server():
    """Start the Gradio WebUI server as a subprocess."""
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "ocr_corrector",
            "--webui", "--port", str(WEBUI_PORT),
        ],
        cwd=str(Path(__file__).resolve().parent.parent),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    # Wait for the server to become ready
    deadline = time.time() + 120
    ready = False
    while time.time() < deadline:
        try:
            import urllib.request
            urllib.request.urlopen(WEBUI_URL, timeout=2)
            ready = True
            break
        except Exception:
            time.sleep(2)

    if not ready:
        out = proc.stdout.read().decode(errors="replace") if proc.stdout else ""
        proc.kill()
        pytest.fail(f"WebUI server did not start within 120s.\nOutput:\n{out}")

    yield proc

    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.fixture(scope="module")
def browser_page(webui_server):
    """Launch headless Chromium and navigate to the WebUI."""
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context(accept_downloads=True)
        page = context.new_page()
        page.goto(WEBUI_URL, wait_until="networkidle")
        yield page
        browser.close()


# ---------- helpers ----------

def _uncheck_llm(page):
    """Uncheck the LLM checkbox so we run BERT-only (no LLM server needed)."""
    cb = page.locator("input[type='checkbox']").first
    if cb.is_checked():
        cb.click()


def _enter_text(page, text: str):
    """Type text into the OCRテキスト textarea (index 1; index 0 is LLM API URL)."""
    textarea = page.locator("textarea[data-testid='textbox']").nth(1)
    textarea.fill(text)


def _click_run(page):
    """Click the 校正実行 button."""
    page.get_by_role("button", name="校正実行").click()


def _wait_for_completion(page, timeout_ms: int = 180_000):
    """Wait until the status textarea (index 2) contains the final '完了 |' marker."""
    status_ta = page.locator("textarea[data-testid='textbox']").nth(2)
    # Match '完了 |' or '完了 (' to distinguish final status from intermediate 'OCR完了:'
    deadline = time.monotonic() + timeout_ms / 1000
    while time.monotonic() < deadline:
        val = status_ta.input_value()
        if "完了 |" in val or "完了 (" in val:
            return val
        time.sleep(1)
    raise TimeoutError(
        f"Status did not reach final state within {timeout_ms}ms. "
        f"Last value: {status_ta.input_value()!r}"
    )


RESULT_FILE_NAMES = {"ocr_raw.txt", "corrections.csv",
                     "corrected_bert.txt", "corrected_llm.txt", "corrected_all.txt"}


def _get_download_links(page):
    """Return download link elements for pipeline result files only."""
    all_links = page.locator("a[download]").all()
    return [a for a in all_links
            if (a.get_attribute("download") or "") in RESULT_FILE_NAMES]


# ---------- tests ----------

class TestDownloadFiles:
    """Test the download functionality end-to-end."""

    def test_bert_only_produces_downloads(self, browser_page):
        """BERT-only mode: enter text -> run -> verify download files appear."""
        page = browser_page

        # 1. Disable LLM (avoids needing a running LLM server)
        _uncheck_llm(page)

        # 2. Enter test text
        _enter_text(page, TEST_INPUT)

        # 3. Run correction
        _click_run(page)

        # 4. Wait for completion
        status = _wait_for_completion(page)
        assert "完了" in status

        # 5. Download files should be present
        #    In BERT-only (LLM disabled) mode, we expect 2 files:
        #    corrections.csv, corrected_bert.txt
        dl_links = _get_download_links(page)
        dl_names = [a.get_attribute("download") for a in dl_links]
        print(f"Download links found: {dl_names}")

        assert len(dl_links) >= 2, f"Expected >= 2 download files, got {len(dl_links)}: {dl_names}"

        # Verify expected file names
        assert any("corrections.csv" in (n or "") for n in dl_names), \
            f"corrections.csv not found in {dl_names}"
        assert any("corrected_bert.txt" in (n or "") for n in dl_names), \
            f"corrected_bert.txt not found in {dl_names}"

        # LLM disabled -> no LLM/all files
        assert not any("corrected_llm.txt" in (n or "") for n in dl_names), \
            "corrected_llm.txt should not appear when LLM is disabled"

    def test_download_csv_content(self, browser_page):
        """Verify the CSV file can be downloaded and has valid content."""
        page = browser_page

        dl_links = _get_download_links(page)
        csv_link = None
        for a in dl_links:
            name = a.get_attribute("download") or ""
            if "corrections.csv" in name:
                csv_link = a
                break

        assert csv_link is not None, "CSV download link not found"

        # Download the file
        with page.expect_download() as dl_info:
            csv_link.click()
        download = dl_info.value
        path = download.path()
        assert path is not None, "Download failed"

        content = Path(path).read_text(encoding="utf-8-sig")
        reader = csv.reader(content.splitlines())
        rows = list(reader)
        assert len(rows) >= 1, "CSV should have at least a header row"
        header = rows[0]
        assert "行" in header[0], f"Unexpected CSV header: {header}"
        assert "BERT確率" in header[3], f"Unexpected CSV header: {header}"
        print(f"CSV rows (incl header): {len(rows)}")

    def test_download_bert_txt_preserves_newlines(self, browser_page):
        """Verify corrected_bert.txt preserves original text structure."""
        page = browser_page

        dl_links = _get_download_links(page)
        bert_link = None
        for a in dl_links:
            name = a.get_attribute("download") or ""
            if "corrected_bert.txt" in name:
                bert_link = a
                break

        assert bert_link is not None, "BERT txt download link not found"

        with page.expect_download() as dl_info:
            bert_link.click()
        download = dl_info.value
        path = download.path()
        assert path is not None, "Download failed"

        content = Path(path).read_text(encoding="utf-8")
        # The input was a single line, so corrected should also be a single line
        # (no spurious newlines added)
        assert content.strip(), "Downloaded text file is empty"
        assert content.count("\n") == TEST_INPUT.count("\n"), \
            f"Newline count mismatch: input has {TEST_INPUT.count(chr(10))}, output has {content.count(chr(10))}"
        print(f"BERT corrected text: {content.strip()!r}")


# ---------- multi-image helpers ----------

def _create_test_images(n: int = 2) -> list[str]:
    """Create simple test images with text using Pillow."""
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
    """Test the multi-image tab end-to-end."""

    def test_multi_image_produces_downloads(self, browser_page):
        """Upload multiple images -> OCR -> correction -> verify downloads."""
        page = browser_page

        # 1. Switch to the multi-image tab (3rd tab, index 2)
        tabs = page.locator("button[role='tab']").all()
        tabs[2].click()
        time.sleep(0.5)

        # 2. Disable LLM
        _uncheck_llm(page)

        # 3. Create test images and upload via the multiple file input
        test_images = _create_test_images(2)
        file_input = page.locator("input[type='file'][multiple]")
        file_input.set_input_files(test_images)
        time.sleep(1)

        # 4. Click the multi-image run button (first "校正実行" in DOM — inside tab)
        run_btns = page.get_by_role("button", name="校正実行").all()
        run_btns[0].click()

        # 5. Wait for multi-image completion — final status contains "完了 (N画像)"
        status_ta = page.locator("textarea[data-testid='textbox']").nth(2)
        deadline = time.monotonic() + 300
        status = ""
        while time.monotonic() < deadline:
            status = status_ta.input_value()
            if "完了 (2画像)" in status:
                break
            time.sleep(1)
        assert "完了 (2画像)" in status, f"Expected '完了 (2画像)' in status, got: {status}"

        # 6. Download files should be present (at least CSV + BERT txt + OCR raw)
        dl_links = _get_download_links(page)
        dl_names = [a.get_attribute("download") for a in dl_links]
        print(f"Multi-image download links: {dl_names}")

        assert any("ocr_raw.txt" in (n or "") for n in dl_names), \
            f"ocr_raw.txt not found in {dl_names}"
        assert any("corrections.csv" in (n or "") for n in dl_names), \
            f"corrections.csv not found in {dl_names}"
        assert any("corrected_bert.txt" in (n or "") for n in dl_names), \
            f"corrected_bert.txt not found in {dl_names}"
