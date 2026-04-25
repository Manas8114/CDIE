"""
Render all .mmd Mermaid files to high-quality PNG.
Uses a local HTTP server so ES modules load correctly (file:// blocks them).
High resolution: 3840x2160 viewport + 2x deviceScaleFactor for crisp output.
"""

import http.server
import pathlib
import socketserver
import threading

from playwright.sync_api import sync_playwright

FIGS = pathlib.Path(__file__).parent
MMD_FILES = sorted(FIGS.glob('*.mmd'))
if not MMD_FILES:
    print('No .mmd files found.')
    exit(0)

PORT = 18734
server_ready = threading.Event()


# ─── Minimal HTTP server for local files ───
class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, directory=None, **kw):
        super().__init__(*a, directory=directory, **kw)

    def log_message(self, fmt, *a):
        pass  # Silent


def serve():
    with socketserver.TCPServer(('', PORT), lambda *a: Handler(*a, directory=str(FIGS))) as httpd:
        server_ready.set()
        httpd.serve_forever()


t = threading.Thread(target=serve, daemon=True)
t.start()
server_ready.wait()

MIME_MAP = {'.mmd': 'text/plain'}
_orig_gue = Handler.guess_type


def _guess(self, url):
    ext = pathlib.Path(url).suffix
    return MIME_MAP.get(ext, _orig_gue(self, url))


Handler.guess_type = _guess

# ─── HTML template — uses ES module from CDN ───
HTML = """\
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body { margin: 0; padding: 0; background: #ffffff; }
</style>
</head>
<body>
<div class="mermaid">
{{content}}
</div>
<script type="module">
import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
await mermaid.initialize({
  startOnLoad: true,
  theme: 'neutral',
  securityLevel: 'loose',
  fontFamily: '"Segoe UI", Arial, sans-serif',
  fontSize: 14,
  flowchart: { useMaxWidth: false, htmlLabels: true, curve: 'basis', padding: 20 },
  sequence: { useMaxWidth: false, wrap: true },
  er: { useMaxWidth: false },
  gantt: { useMaxWidth: false },
  classDiagram: { useMaxWidth: false },
  stateDiagram: { useMaxWidth: false }
});
mermaid.init(undefined, document.querySelectorAll('.mermaid'));
</script>
</body>
</html>
"""


def render(mmd: pathlib.Path, page) -> None:
    content = mmd.read_text(encoding='utf-8')
    html = HTML.replace('{{content}}', content)
    tmp = FIGS / f'_render_{mmd.name}.html'
    tmp.write_text(html, encoding='utf-8')
    out_png = mmd.with_suffix('.png')

    try:
        # navigate via HTTP so ES module works
        page.goto(f'http://127.0.0.1:{PORT}/{tmp.name}', wait_until='networkidle', timeout=45000)
        page.wait_for_timeout(5000)  # let mermaid render

        svg = page.query_selector('svg')
        if not svg:
            page.wait_for_timeout(5000)
            svg = page.query_selector('svg')

        if not svg:
            print(f'  SKIP {mmd.stem}.png  (no SVG after render)')
            tmp.unlink()
            return False

        bbox = svg.bounding_box()
        if not bbox:
            print(f'  SKIP {mmd.stem}.png  (no bbox)')
            tmp.unlink()
            return False

        svg_w = int(bbox['width']) + 40
        svg_h = int(bbox['height']) + 40

        # Set a large viewport for crisp capture
        vw = max(svg_w, 1920)
        vh = max(svg_h, 1080)
        page.set_viewport_size({'width': vw, 'height': vh})
        page.wait_for_timeout(500)

        # Screenshot entire page → full-resolution PNG
        # deviceScaleFactor=2 is set below → 2x crisp
        page.screenshot(path=str(out_png), full_page=True, omit_background=False)
        print(f'  OK  {out_png.name}  ({svg_w} x {svg_h} viewport {vw}x{vh} @2x)')
        tmp.unlink()
        return True

    except Exception as e:
        print(f'  FAIL {mmd.stem}.png  {e}')
        tmp.unlink()
        return False


# ─── Main ───
print(f'\n{"=" * 60}')
print(f' Rendering {len(MMD_FILES)} diagrams at high quality')
print(f'{"=" * 60}\n')

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    ctx = browser.new_context(
        viewport={'width': 3840, 'height': 2160},
        device_scale_factor=2,  # 2x DPR → crisp text and edges
        is_mobile=False,
    )
    page = ctx.new_page()

    ok = 0
    for mmd in MMD_FILES:
        print(f'[{ok + 1}/{len(MMD_FILES)}] {mmd.name}')
        if render(mmd, page):
            ok += 1

    ctx.close()
    browser.close()

print(f'\n{"=" * 60}')
print(f' Done: {ok}/{len(MMD_FILES)} rendered')
print(f'{"=" * 60}\n')
