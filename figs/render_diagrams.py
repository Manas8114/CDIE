"""
Convert all .mmd Mermaid files in the figs/ folder to PNG images.
Uses Playwright + Mermaid JS via CDN.
"""

import pathlib

from playwright.sync_api import sync_playwright

FIGS_DIR = pathlib.Path(__file__).parent  # figs/ dir same level as this script

MERMAID_FILES = sorted(f for f in FIGS_DIR.glob('*.mmd'))

if not MERMAID_FILES:
    print('No .mmd files found.')
    exit(0)

print(f'Found {len(MERMAID_FILES)} diagram files.')

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {{ margin: 40px; background: #ffffff; }}
  .mermaid {{ font-family: 'Segoe UI', sans-serif; }}
</style>
</head>
<body>
<pre class="mermaid">
{content}
</pre>
<script type="module">
import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
await mermaid.initialize({{
  startOnLoad: true,
  theme: 'default',
  securityLevel: 'loose',
  fontFamily: 'Segoe UI, sans-serif',
  fontSize: 14,
  flowchart: {{ useMaxWidth: false, htmlLabels: true, curve: 'basis' }},
  sequence: {{ useMaxWidth: false, wrap: true }},
  er: {{ useMaxWidth: false }}
}});
await mermaid.run();
// Tell parent we're ready
window.addEventListener('message', (e) => {{
  if (e.data === 'rendered') {{
    window.postMessage('done', '*');
  }}
}});
</script>
</body>
</html>
"""

with sync_playwright() as p:
    browser = p.chromium.launch()

    for mmd_file in MERMAID_FILES:
        print(f'\nRendering {mmd_file.name}...')
        page = browser.new_page()

        html_content = HTML_TEMPLATE.format(content=mmd_file.read_text(encoding='utf-8'))

        # Serve via data URI to avoid file:// issues
        data_uri = 'data:text/html;charset=utf-8,' + html_content
        page.goto(data_uri, wait_until='networkidle', timeout=30000)

        # Wait for mermaid to render
        page.wait_for_timeout(3000)

        # Check if mermaid svg is present
        svg = page.query_selector('svg')
        if svg is None:
            print('  ⚠ No SVG found, waiting more...')
            page.wait_for_timeout(5000)
            svg = page.query_selector('svg')

        if svg is None:
            print(f'  ❌ Failed to render {mmd_file.name}')
            page.close()
            continue

        # Get bounding box for tight crop
        bbox = svg.bounding_box()
        if bbox:
            # Set viewport to SVG size + padding
            viewport_w = int(bbox['width'] + 120)
            viewport_h = int(bbox['height'] + 120)
            page.set_viewport_size({'width': viewport_w, 'height': viewport_h})
            page.wait_for_timeout(500)

        # Screenshot the SVG element
        out_path = mmd_file.with_suffix('.png')
        svg.screenshot(path=str(out_path), type='png')
        print(f'  ✅ {out_path}')
        page.close()

    browser.close()

print('\n🎉 All diagrams rendered!')
