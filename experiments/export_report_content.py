"""Export the report's source flowables for native editable Google Docs authoring."""
from pathlib import Path
import json
from reportlab.platypus import Paragraph, Table, Image, Preformatted, PageBreak

path = Path(__file__).with_name('create_final_report.py')
scope = {'__file__': str(path)}
exec(compile(path.read_text().split('\ndef footer(')[0], str(path), 'exec'), scope)
blocks = []
for obj in scope['story']:
    if isinstance(obj, Paragraph):
        spans = []
        text = ''
        for frag in obj.frags:
            value = '\n' if getattr(frag, 'lineBreak', False) else getattr(frag, 'text', '')
            span = {'start': len(text), 'end': len(text) + len(value)}
            if getattr(frag, 'bold', False): span['bold'] = True
            if getattr(frag, 'italic', False): span['italic'] = True
            links = getattr(frag, 'link', [])
            if links: span['url'] = links[0][1]
            if len(span) > 2: spans.append(span)
            text += value
        blocks.append(dict(type='text', text=text, style=obj.style.name, spans=spans))
    elif isinstance(obj, Table):
        blocks.append(dict(type='table', rows=[[c.getPlainText() for c in row]
                                              for row in obj._cellvalues], widths=obj._argW))
    elif isinstance(obj, Image):
        blocks.append(dict(type='image', path=str(Path(obj.filename).resolve()),
                           width=obj.drawWidth, height=obj.drawHeight))
    elif isinstance(obj, Preformatted):
        blocks.append(dict(type='code', text='\n'.join(obj.lines)))
    elif isinstance(obj, PageBreak):
        blocks.append(dict(type='page'))
dest = path.parents[2] / 'output/report_content.json'
dest.write_text(json.dumps(blocks, indent=2), encoding='utf-8')
print(f'{len(blocks)} content blocks exported to {dest}')
