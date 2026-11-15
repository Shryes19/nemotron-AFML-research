from copy import deepcopy
from pathlib import Path
import shutil

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

source = Path('/home/admin/Downloads/Team_Nemotron_IEEE_Paper_v2.docx')
output = Path('docs/paper/Team_Nemotron_IEEE_Paper_final.docx')
figure = Path('results/figures/fig_g_recall_at5_vs_n.png')
shutil.copy2(source, output)
doc = Document(output)

limitation = next(p for p in doc.paragraphs if p.text.startswith('Model size and method are confounded:'))
text = (' The planned G-versus-action-lookup comparison was not run because no scored '
        'action-lookup artifact with a shared utility field is available. Raw G prompts and '
        'predictions were not retained in this checkout; the scored G outputs cannot reconstruct '
        'the generated text or support a post-hoc raw-prediction audit.')
run = limitation.add_run(text)
if limitation.runs and limitation.runs[0]._r.rPr is not None:
    run._r.get_or_add_rPr().append(deepcopy(limitation.runs[0]._r.rPr))

section = doc.add_section(WD_SECTION.NEW_PAGE)
cols = section._sectPr.xpath('./w:cols')[0]
cols.set(qn('w:num'), '1')
cols.attrib.pop(qn('w:space'), None)

heading = doc.add_paragraph()
heading.alignment = WD_ALIGN_PARAGRAPH.CENTER
heading.paragraph_format.space_after = Pt(5)
run = heading.add_run('APPENDIX A\nG RETRIEVAL COVERAGE')
run.bold = True
run.font.name = 'Times New Roman'
run.font.size = Pt(10)

image_para = doc.add_paragraph()
image_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
image_para.paragraph_format.space_after = Pt(4)
image_para.add_run().add_picture(str(figure), width=Inches(6.65))

caption = doc.add_paragraph()
caption.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
caption.paragraph_format.space_after = Pt(3)
cap_run = caption.add_run(
    'Fig. A1. Recall@5 of G\'s hierarchical retriever on branched evaluation states by '
    'training-transition budget. Each point is a state-weighted mean across per-game or '
    'per-task groups, rather than an unweighted mean of groups.'
)
cap_run.font.name = 'Times New Roman'
cap_run.font.size = Pt(9)

note = doc.add_paragraph()
note.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
note.paragraph_format.space_after = Pt(0)
note_run = note.add_run(
    'This is descriptive retrieval coverage evaluated with the gold action. It is distinct '
    'from the leave-one-out, training-only Recall@5 feature used for RQ3.'
)
note_run.italic = True
note_run.font.name = 'Times New Roman'
note_run.font.size = Pt(9)

doc.save(output)
print(output)
