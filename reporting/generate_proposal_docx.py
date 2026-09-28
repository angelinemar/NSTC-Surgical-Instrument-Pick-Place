"""
Generates proposal_dual_track.docx
Run from the project root: python reporting/generate_proposal_docx.py
Requires: pip install python-docx
"""

from docx import Document
from docx.shared import Pt, RGBColor, Inches, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import copy

# ── Palette ──────────────────────────────────────────────────────────────────
NAVY        = RGBColor(0x0D, 0x1B, 0x2A)
STEEL       = RGBColor(0x4A, 0x6F, 0xA5)
GOLD        = RGBColor(0xB8, 0x92, 0x2A)
ANGEL_HUE   = RGBColor(0xC0, 0x5E, 0x52)
JORDAN_HUE  = RGBColor(0x3A, 0x8E, 0x80)
SHARED_HUE  = RGBColor(0x5C, 0x6B, 0xC0)
TEXT_MUTED  = RGBColor(0x6B, 0x7A, 0x8D)
WHITE       = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_GRAY  = RGBColor(0xF0, 0xF2, 0xF8)
ANGEL_BG    = RGBColor(0xFD, 0xF1, 0xEF)
JORDAN_BG   = RGBColor(0xED, 0xF7, 0xF5)
SHARED_BG   = RGBColor(0xF0, 0xF1, 0xFA)
WARNING_BG  = RGBColor(0xFF, 0xF8, 0xE8)
BORDER_COLOR= RGBColor(0xD8, 0xDC, 0xE8)

# ── Helpers ───────────────────────────────────────────────────────────────────

def set_cell_bg(cell, rgb: RGBColor):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), f'{rgb.red:02X}{rgb.green:02X}{rgb.blue:02X}')
    tcPr.append(shd)

def set_cell_border(cell, **kwargs):
    """kwargs: top, bottom, left, right — each a dict with color and sz keys."""
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcBorders = OxmlElement('w:tcBorders')
    for side, cfg in kwargs.items():
        border = OxmlElement(f'w:{side}')
        border.set(qn('w:val'), cfg.get('val', 'single'))
        border.set(qn('w:sz'), str(cfg.get('sz', 4)))
        border.set(qn('w:space'), '0')
        border.set(qn('w:color'), cfg.get('color', 'D8DCE8'))
        tcBorders.append(border)
    tcPr.append(tcBorders)

def para_space(para, before=0, after=100):
    pPr = para._p.get_or_add_pPr()
    spacing = OxmlElement('w:spacing')
    spacing.set(qn('w:before'), str(before))
    spacing.set(qn('w:after'), str(after))
    pPr.append(spacing)

def add_horizontal_rule(doc, color='D8DCE8'):
    para = doc.add_paragraph()
    pPr = para._p.get_or_add_pPr()
    pBdr = OxmlElement('w:pBdr')
    bottom = OxmlElement('w:bottom')
    bottom.set(qn('w:val'), 'single')
    bottom.set(qn('w:sz'), '4')
    bottom.set(qn('w:space'), '1')
    bottom.set(qn('w:color'), color)
    pBdr.append(bottom)
    pPr.append(pBdr)
    para_space(para, before=160, after=160)
    return para

def add_section_label(doc, text, color=STEEL):
    p = doc.add_paragraph()
    run = p.add_run(text.upper())
    run.font.size = Pt(7.5)
    run.font.color.rgb = color
    run.font.bold = True
    run.font.name = 'Courier New'
    para_space(p, before=120, after=40)
    return p

def add_h2(doc, text):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.size = Pt(14)
    run.font.bold = True
    run.font.color.rgb = NAVY
    run.font.name = 'Georgia'
    pPr = p._p.get_or_add_pPr()
    pBdr = OxmlElement('w:pBdr')
    bottom = OxmlElement('w:bottom')
    bottom.set(qn('w:val'), 'single')
    bottom.set(qn('w:sz'), '4')
    bottom.set(qn('w:space'), '1')
    bottom.set(qn('w:color'), 'D8DCE8')
    pBdr.append(bottom)
    pPr.append(pBdr)
    para_space(p, before=80, after=100)
    return p

def add_h3(doc, text, color=NAVY):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.size = Pt(11)
    run.font.bold = True
    run.font.color.rgb = color
    run.font.name = 'Georgia'
    para_space(p, before=120, after=40)
    return p

def add_body(doc, text, indent=False):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.size = Pt(10)
    run.font.color.rgb = RGBColor(0x2C, 0x3A, 0x4E)
    run.font.name = 'Calibri'
    if indent:
        p.paragraph_format.left_indent = Inches(0.25)
    para_space(p, before=0, after=80)
    return p

def add_bullet(doc, label, desc, label_color=NAVY):
    p = doc.add_paragraph(style='List Bullet')
    r1 = p.add_run(label + '  ')
    r1.font.name = 'Courier New'
    r1.font.size = Pt(9)
    r1.font.bold = True
    r1.font.color.rgb = label_color
    r2 = p.add_run(desc)
    r2.font.name = 'Calibri'
    r2.font.size = Pt(9.5)
    r2.font.color.rgb = TEXT_MUTED
    para_space(p, before=20, after=40)
    return p

def add_info_box(doc, title, body_text, bg=LIGHT_GRAY, border_color=STEEL):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.LEFT
    cell = tbl.cell(0, 0)
    set_cell_bg(cell, bg)
    # left accent border
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcBorders = OxmlElement('w:tcBorders')
    for side in ['top', 'bottom', 'right']:
        b = OxmlElement(f'w:{side}')
        b.set(qn('w:val'), 'none')
        b.set(qn('w:sz'), '0')
        b.set(qn('w:space'), '0')
        b.set(qn('w:color'), 'auto')
        tcBorders.append(b)
    left = OxmlElement('w:left')
    left.set(qn('w:val'), 'single')
    left.set(qn('w:sz'), '18')
    left.set(qn('w:space'), '0')
    left.set(qn('w:color'), f'{border_color.red:02X}{border_color.green:02X}{border_color.blue:02X}')
    tcBorders.append(left)
    tcPr.append(tcBorders)

    cell.width = Inches(5.8)
    p_title = cell.paragraphs[0]
    r = p_title.add_run(title.upper())
    r.font.name = 'Courier New'
    r.font.size = Pt(8)
    r.font.bold = True
    r.font.color.rgb = border_color
    para_space(p_title, before=60, after=40)

    p_body = cell.add_paragraph()
    r2 = p_body.add_run(body_text)
    r2.font.name = 'Calibri'
    r2.font.size = Pt(9.5)
    r2.font.color.rgb = RGBColor(0x2C, 0x3A, 0x4E)
    para_space(p_body, before=0, after=60)
    return tbl

def add_owner_box(doc, badge, name, body_text, color):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.LEFT
    cell = tbl.cell(0, 0)
    bg = ANGEL_BG if color == ANGEL_HUE else JORDAN_BG
    set_cell_bg(cell, bg)
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcBorders = OxmlElement('w:tcBorders')
    for side in ['top', 'bottom', 'left', 'right']:
        b = OxmlElement(f'w:{side}')
        b.set(qn('w:val'), 'single')
        b.set(qn('w:sz'), '6')
        b.set(qn('w:space'), '0')
        b.set(qn('w:color'), f'{color.red:02X}{color.green:02X}{color.blue:02X}')
        tcBorders.append(b)
    tcPr.append(tcBorders)

    p_badge = cell.paragraphs[0]
    rb = p_badge.add_run(f'◆ {badge.upper()}')
    rb.font.name = 'Courier New'
    rb.font.size = Pt(8)
    rb.font.bold = True
    rb.font.color.rgb = color
    para_space(p_badge, before=60, after=30)

    p_name = cell.add_paragraph()
    rn = p_name.add_run(name)
    rn.font.name = 'Georgia'
    rn.font.size = Pt(12)
    rn.font.bold = True
    rn.font.color.rgb = color
    para_space(p_name, before=0, after=30)

    p_body = cell.add_paragraph()
    rb2 = p_body.add_run(body_text)
    rb2.font.name = 'Calibri'
    rb2.font.size = Pt(9.5)
    rb2.font.color.rgb = RGBColor(0x2C, 0x3A, 0x4E)
    para_space(p_body, before=0, after=60)
    return tbl

def add_attrib_table(doc, headers, rows, col_widths=None):
    tbl = doc.add_table(rows=1 + len(rows), cols=len(headers))
    tbl.style = 'Table Grid'
    tbl.alignment = WD_TABLE_ALIGNMENT.LEFT

    # Header row
    hdr_row = tbl.rows[0]
    for i, h in enumerate(headers):
        cell = hdr_row.cells[i]
        set_cell_bg(cell, LIGHT_GRAY)
        p = cell.paragraphs[0]
        r = p.add_run(h.upper())
        r.font.name = 'Courier New'
        r.font.size = Pt(7.5)
        r.font.bold = True
        r.font.color.rgb = TEXT_MUTED
        para_space(p, before=40, after=40)

    # Data rows
    for ri, row_data in enumerate(rows):
        tr = tbl.rows[ri + 1]
        for ci, val in enumerate(row_data):
            cell = tr.cells[ci]
            p = cell.paragraphs[0]
            if isinstance(val, tuple):
                text, color = val
                r = p.add_run(text)
                r.font.name = 'Courier New'
                r.font.size = Pt(8.5)
                r.font.bold = True
                r.font.color.rgb = color
            else:
                r = p.add_run(str(val))
                r.font.name = 'Calibri'
                r.font.size = Pt(9.5)
                r.font.color.rgb = RGBColor(0x2C, 0x3A, 0x4E)
            para_space(p, before=40, after=40)

    if col_widths:
        for ri in range(len(tbl.rows)):
            for ci, w in enumerate(col_widths):
                tbl.rows[ri].cells[ci].width = Inches(w)
    return tbl

def spacer(doc, n=1):
    for _ in range(n):
        p = doc.add_paragraph()
        para_space(p, before=0, after=60)

# ── Build Document ────────────────────────────────────────────────────────────

doc = Document()

# Page margins
for section in doc.sections:
    section.top_margin    = Cm(2.2)
    section.bottom_margin = Cm(2.2)
    section.left_margin   = Cm(2.8)
    section.right_margin  = Cm(2.8)

# ── Cover block ──────────────────────────────────────────────────────────────
cover_tbl = doc.add_table(rows=1, cols=1)
cover_tbl.alignment = WD_TABLE_ALIGNMENT.LEFT
cc = cover_tbl.cell(0, 0)
set_cell_bg(cc, NAVY)

p_ey = cc.paragraphs[0]
r_ey = p_ey.add_run('Special Implementation Topic — Research Proposal')
r_ey.font.name = 'Courier New'
r_ey.font.size = Pt(8)
r_ey.font.color.rgb = RGBColor(0xD4, 0xA8, 0x43)
r_ey.font.bold = True
para_space(p_ey, before=100, after=80)

p_title = cc.add_paragraph()
r_title = p_title.add_run('Sim-to-Sim Imitation Learning for Thin Surgical Instrument Manipulation')
r_title.font.name = 'Georgia'
r_title.font.size = Pt(20)
r_title.font.bold = True
r_title.font.color.rgb = WHITE
para_space(p_title, before=0, after=80)

p_sub = cc.add_paragraph()
r_sub = p_sub.add_run('Dual-Track Investigation: Diffusion Policy & OpenVLA on a Shared Simulation Platform')
r_sub.font.name = 'Calibri'
r_sub.font.size = Pt(10.5)
r_sub.font.color.rgb = RGBColor(0xCC, 0xD4, 0xE0)
para_space(p_sub, before=0, after=80)

for chip_text, chip_color in [
    ('Angel  ·  Track A: Diffusion Policy', ANGEL_HUE),
    ('Jordan  ·  Track B: OpenVLA', JORDAN_HUE),
    ('Shared Infrastructure', SHARED_HUE),
    ('Academic Year 2025–2026', RGBColor(0xCC, 0xD4, 0xE0)),
]:
    pc = cc.add_paragraph()
    rc = pc.add_run(f'  {chip_text}  ')
    rc.font.name = 'Calibri'
    rc.font.size = Pt(9)
    rc.font.color.rgb = chip_color
    rc.font.bold = True
    para_space(pc, before=20, after=20)

spacer_p = cc.add_paragraph()
para_space(spacer_p, before=60, after=60)

spacer(doc)

# ── § 01 Abstract ─────────────────────────────────────────────────────────────
add_section_label(doc, '§ 01')
add_h2(doc, 'Abstract')
add_body(doc, 'This document presents a coordinated dual-track research proposal investigating imitation learning strategies for autonomous manipulation of thin surgical instruments within a physics-based simulation environment. The central challenge addressed is sim-to-sim transfer: training policies on a source simulation and evaluating generalization to a distinct target simulation, without physical robot deployment. This constraint isolates the policy learning algorithm as the primary variable, enabling rigorous cross-method comparison.')
add_body(doc, 'Track A (Angel) applies Diffusion Policy to the task, leveraging a custom Isaac Lab environment, Finite State Machine (FSM) task logic, and a specialized demonstration recorder developed independently by Angel. Track B (Jordan) adapts OpenVLA to the same task by extending a subset of Angel\'s simulation infrastructure, contributing new model integration and language-conditioned training protocols. Each track constitutes a distinct, assessable research contribution; the shared platform is a defined infrastructure layer, not a joint authorship claim.')
add_body(doc, 'The application domain — minimally invasive surgical instrumentation — carries direct clinical relevance: thin, high-aspect-ratio instruments demand sub-millimeter precision, and improved autonomous or semi-autonomous control can reduce unintended tissue trauma, lower infection risk, and simplify sterilization through reduced human contact cycles.')

add_horizontal_rule(doc)

# ── § 02 Motivation ───────────────────────────────────────────────────────────
add_section_label(doc, '§ 02')
add_h2(doc, 'Problem Statement & Clinical Motivation')
add_h3(doc, 'The Surgical Instrumentation Challenge')
add_body(doc, 'Modern minimally invasive surgery (MIS) relies on instruments with shaft diameters between 3 mm and 12 mm and working lengths of 25–45 cm. At these scales, tactile feedback is attenuated, kinematic constraints from trocar ports induce counter-intuitive motion mapping, and any lateral force exceeding tissue tolerances (typically 0.1–2 N for delicate structures) risks perforation or bruising. Manual teleoperation under such constraints is cognitively demanding and fatigue-sensitive.')
add_body(doc, 'Automation or decision-support at the subtask level — grasping, tissue retraction, clip placement — could reduce operator cognitive load, decrease procedure time, and improve reproducibility. However, real-to-sim gaps and data scarcity make direct learning from physical demonstrations difficult for early research. Sim-to-sim methodology allows algorithm development and benchmarking at low cost and zero biological risk.')

add_h3(doc, 'Why Sim-to-Sim Before Sim-to-Real')
add_body(doc, 'The staged research approach adopted here — establishing robust sim-to-sim performance before pursuing sim-to-real transfer — follows established practice in contact-rich manipulation research. A policy that cannot generalize across simulation engines with different physics parameters (stiffness, friction, contact resolution) is unlikely to survive the additional gap to real hardware. Sim-to-sim results thus constitute a meaningful scientific contribution independent of eventual physical deployment.')

add_info_box(doc,
    'Clinical Relevance Metrics',
    'Instrument-related complications in laparoscopic procedures account for approximately 0.3–0.5% of adverse events. Autonomous or AI-assisted subtask execution targeting <0.5 mm positional error and controlled grasping forces below 1.5 N is the operative target for this research track. Every algorithm improvement that reduces commanded position variance or unintended contact force directly maps to reduced patient risk.',
    bg=WARNING_BG, border_color=GOLD)
spacer(doc)

add_horizontal_rule(doc)

# ── § 03 Shared Infrastructure ────────────────────────────────────────────────
add_section_label(doc, '§ 03')
add_h2(doc, 'Shared Infrastructure Platform')

add_info_box(doc,
    'Authorship & Ownership Notice',
    'The simulation environment, FSM task logic, and demonstration recording pipeline described in this section were designed and implemented in full by Angel as part of Track A development. Jordan is granted a defined, read-only interface to these components for Track B use. Use of these components by Jordan does not constitute joint ownership, co-authorship of the infrastructure, or credit for their design. Jordan\'s independent contribution lies solely in OpenVLA model adaptation, language-conditioned training, and Track B evaluation (see § 05).',
    bg=WARNING_BG, border_color=GOLD)
spacer(doc)

add_h3(doc, 'Isaac Lab Simulation Environment')
add_body(doc, 'The primary simulation platform is NVIDIA Isaac Lab (built on Isaac Sim / PhysX), selected for its deformable-body support, GPU-parallel environment instantiation, and active robotics research community. The environment encapsulates:')
add_bullet(doc, 'ScissorGraspEnv', 'Task-level environment class exposing observation and action spaces compatible with both Gym and Isaac Lab APIs.', ANGEL_HUE)
add_bullet(doc, 'SurgicalScissorAsset', 'Articulated scissor model with calibrated joint stiffness, damping, and collision geometry matching physical scissor kinematics.', ANGEL_HUE)
add_bullet(doc, 'TissuePhantomAsset', 'Deformable mesh tissue phantom with configurable Young\'s modulus (5–25 kPa) representing soft-tissue compliance range in MIS.', ANGEL_HUE)
add_bullet(doc, 'WorkspaceObserver', 'Wrist-mounted and global RGB-D observation streams, with configurable resolution and noise injection for domain randomization.', ANGEL_HUE)
add_bullet(doc, 'ContactRewardShaper', 'Reward function components including tissue contact force penalties, target grasp-point proximity, and instrument orientation alignment terms.', ANGEL_HUE)

add_h3(doc, 'Finite State Machine (FSM) Task Logic')
add_body(doc, 'All structured task decomposition is governed by a multi-stage FSM authored entirely by Angel. The FSM drives expert demonstrations and evaluation episode structure:')
for stage, desc in [
    ('APPROACH', 'Pre-grasp approach to target tissue site with velocity ramping and collision avoidance margins.'),
    ('ORIENT',   'Instrument tip alignment to target cut or grasp axis within ±3° tolerance before contact.'),
    ('ENGAGE',   'Controlled jaw or blade engagement with force-regulated descent; abort on force threshold breach.'),
    ('EXECUTE',  'Task-specific action (cut, grasp, retract) with live success criterion monitoring.'),
    ('WITHDRAW', 'Safe retraction to clear pose; episode success/fail logging and reset trigger.'),
]:
    add_bullet(doc, stage, desc, ANGEL_HUE)

add_h3(doc, 'Demonstration Recorder')
add_body(doc, 'The demonstration recording pipeline outputs HDF5 files compatible with the Robomimic dataset schema, enabling direct ingestion by both Diffusion Policy (Track A) and, via conversion utilities, OpenVLA\'s training pipeline (Track B). Fields recorded per timestep include:')
for field, desc in [
    ('obs/images/wrist_rgb',  '480×640 uint8 RGB from wrist-mounted camera.'),
    ('obs/images/global_rgb', '480×640 uint8 RGB from fixed overhead camera.'),
    ('obs/robot_state',       '7-DOF joint positions and velocities; TCP pose as 6D vector.'),
    ('obs/contact_forces',    'Instrument-tissue contact wrench (6-axis); tip force scalar.'),
    ('actions',               'Commanded joint deltas in absolute joint-space representation.'),
    ('fsm/stage_id',          'Current FSM stage integer; enables stage-conditioned filtering and curriculum slicing.'),
]:
    add_bullet(doc, field, desc, ANGEL_HUE)

spacer(doc)
add_h3(doc, 'Shared Interface Contract — What Jordan May Use from Angel\'s Platform', color=SHARED_HUE)
add_body(doc, 'Jordan\'s Track B work may import and use the following, unchanged, as infrastructure inputs. Modifications to these components require explicit acknowledgement of the original author and must not be presented as Jordan\'s independent work.')

add_attrib_table(doc,
    ['Component', 'Usage by Jordan', 'Owner'],
    [
        ('ScissorGraspEnv',                       'Environment instantiation and observation collection for OpenVLA rollouts',    ('Angel — sole', ANGEL_HUE)),
        ('FSM Expert Demonstrations (HDF5)',       'Source dataset for OpenVLA fine-tuning; format conversion permitted',          ('Angel — sole', ANGEL_HUE)),
        ('ContactRewardShaper',                    'Evaluation metric computation in Track B closed-loop assessment',              ('Angel — sole', ANGEL_HUE)),
        ('WorkspaceObserver config',               'Camera stream configuration reused as-is for visual observation',              ('Angel — sole', ANGEL_HUE)),
    ],
    col_widths=[2.0, 2.6, 1.2])

add_horizontal_rule(doc)

# ── § 04 Track A ──────────────────────────────────────────────────────────────
add_section_label(doc, '§ 04')
add_h2(doc, 'Track A — Diffusion Policy for Surgical Manipulation')
add_owner_box(doc, 'Sole Author', 'Angel',
    'All work in this track — environment design, FSM logic, recorder implementation, dataset curation, training pipeline, evaluation protocol, and written analysis — is the independent contribution of Angel. No portion of Track A is co-authored with Jordan.',
    ANGEL_HUE)
spacer(doc)

add_h3(doc, 'Research Objective')
add_body(doc, 'Track A investigates whether Diffusion Policy — a score-based generative model that learns action distributions through denoising diffusion — can acquire robust manipulation skills for thin surgical instruments from a limited number of expert demonstrations (<200 episodes), with sufficient precision to meet clinical-analog accuracy thresholds in simulation.')

add_h3(doc, 'Technical Approach')
add_body(doc, 'Policy Architecture. A DDPM-based (Denoising Diffusion Probabilistic Model) action prediction network conditioned on a 2-frame visual history (wrist + global RGB) and proprioceptive state. The action head predicts a trajectory chunk of length T=16 at each inference step, with re-planning triggered by FSM stage transitions.')
add_body(doc, 'Dataset Curation. Expert demonstrations are collected through FSM-driven scripted rollouts with injected Gaussian noise (σ = 0.3 mm, 0.5°) to increase behavioral diversity. Demonstrations failing contact-force thresholds during the ENGAGE state are filtered automatically. Approximately 150–300 successful episodes constitute the training corpus.')
add_body(doc, 'Training Protocol. Vision encoder: ResNet-18 pretrained on ImageNet, fine-tuned jointly. Diffusion steps: 100 (training), 20 (inference with DDIM). Optimizer: AdamW, lr = 1×10⁻⁴, cosine schedule. Batch size: 64. Hardware: single NVIDIA RTX 4090.')

add_h3(doc, 'Evaluation Metrics')
add_body(doc, 'Success rate across 50 evaluation episodes per configuration; mean positional error at ENGAGE entry (target: <1.5 mm); mean tissue contact force during EXECUTE (target: <1.0 N peak); sim-to-sim transfer score from training domain to held-out physics-parameter variant (±30% stiffness, friction offset).')

add_h3(doc, 'Expected Contributions')
add_body(doc, 'Quantitative characterization of Diffusion Policy sample efficiency on contact-rich thin-instrument tasks; ablation of observation modality (vision-only vs. vision + proprioception + contact); open dataset of FSM-curated surgical manipulation demonstrations in Isaac Lab format.')

add_horizontal_rule(doc)

# ── § 05 Track B ──────────────────────────────────────────────────────────────
add_section_label(doc, '§ 05')
add_h2(doc, 'Track B — OpenVLA Adaptation for Surgical Manipulation')
add_owner_box(doc, 'Sole Author', 'Jordan',
    'Jordan\'s independent contributions are: OpenVLA model integration and fine-tuning pipeline, language prompt engineering for task conditioning, Track B evaluation protocol, and cross-track comparative analysis. Jordan does not own the simulation environment, FSM logic, or recorder.',
    JORDAN_HUE)
spacer(doc)

add_h3(doc, 'Research Objective')
add_body(doc, 'Track B investigates whether OpenVLA — a vision-language-action model pretrained on broad robot manipulation data — can be efficiently fine-tuned for surgical instrument manipulation using the demonstration dataset produced by Angel\'s recorder, and whether language conditioning enables zero-shot sub-task specialization (e.g., "approach and orient scissor tip to mark A").')

add_h3(doc, 'Technical Approach')
add_body(doc, 'Model. OpenVLA-7B (or OFT-variant), loaded with LoRA fine-tuning adapters targeting attention projection layers (rank 32, α = 64) to reduce VRAM requirements while preserving pretrained visual representations.')
add_body(doc, 'Dataset Conversion. Angel\'s HDF5 demonstrations are converted to the LeRobot/RLDS format required by OpenVLA\'s fine-tuning scripts. Jordan is responsible for the format conversion utilities and data augmentation pipeline (color jitter, random crop) applied at training time.')
add_body(doc, 'Language Conditioning. Each FSM stage maps to a natural-language instruction template authored by Jordan (e.g., "Grasp the tissue phantom at the marked site with controlled force"). Prompt ablations test the sensitivity of policy quality to instruction specificity.')
add_body(doc, 'Training Protocol. LoRA fine-tuning for 20,000 steps; batch size 16; gradient accumulation 4; mixed precision (bfloat16). Evaluation checkpoints every 2,000 steps.')

add_h3(doc, 'Evaluation Metrics')
add_body(doc, 'Success rate on the same 50-episode evaluation suite used by Track A (enabling direct comparison); language ablation: task success versus instruction type (stage-specific, generic, null); compute efficiency: fine-tuning time and GPU memory versus Track A training cost.')

add_h3(doc, 'Expected Contributions')
add_body(doc, 'First characterization of OpenVLA fine-tuning efficiency on thin surgical instrument tasks in simulation; language conditioning sensitivity analysis; direct benchmark comparison against Diffusion Policy on identical evaluation conditions.')

add_horizontal_rule(doc)

# ── § 06 Attribution Table ────────────────────────────────────────────────────
add_section_label(doc, '§ 06')
add_h2(doc, 'Complete Contribution Attribution')
add_body(doc, 'The table below provides an unambiguous mapping of every project component to its author. This record constitutes the official attribution for assessment and publication purposes.')

add_attrib_table(doc,
    ['Component / Artifact', 'Category', 'Author', 'Track B Usage'],
    [
        ('Isaac Lab Environment (ScissorGraspEnv)',      'Simulation',         ('Angel — SOLE', ANGEL_HUE),  'Read-only use permitted'),
        ('SurgicalScissorAsset & TissuePhantomAsset',   'Asset Design',       ('Angel — SOLE', ANGEL_HUE),  'Used as-is, no modification'),
        ('FSM Task Logic (5-stage)',                     'Task Engineering',   ('Angel — SOLE', ANGEL_HUE),  'Episode structure reference only'),
        ('Demonstration Recorder (HDF5 pipeline)',      'Data Infrastructure',('Angel — SOLE', ANGEL_HUE),  'Output dataset used by Jordan'),
        ('ContactRewardShaper',                          'Reward Engineering', ('Angel — SOLE', ANGEL_HUE),  'Evaluation metric reuse'),
        ('Diffusion Policy training pipeline',           'ML Engineering',     ('Angel',         ANGEL_HUE),  'Not used'),
        ('Track A dataset curation & filtering',        'Data Science',       ('Angel',         ANGEL_HUE),  'Not used'),
        ('Track A evaluation protocol & results',       'Research',           ('Angel',         ANGEL_HUE),  'Comparison benchmark only'),
        ('HDF5 → LeRobot/RLDS conversion utility',      'Data Engineering',   ('Jordan',        JORDAN_HUE), 'Jordan — sole'),
        ('OpenVLA fine-tuning pipeline (LoRA)',          'ML Engineering',     ('Jordan',        JORDAN_HUE), 'Jordan — sole'),
        ('Language instruction templates & ablations',  'NLP / Prompt Eng.',  ('Jordan',        JORDAN_HUE), 'Jordan — sole'),
        ('Track B evaluation protocol & results',       'Research',           ('Jordan',        JORDAN_HUE), 'Jordan — sole'),
        ('Cross-track comparative analysis',             'Research',           ('Joint (both)',  SHARED_HUE), 'Jointly authored section'),
        ('Isaac Lab base framework',                     'Third-Party Dep.',   ('NVIDIA / OSS',  TEXT_MUTED), 'Dependency, not a contribution'),
    ],
    col_widths=[2.2, 1.4, 1.0, 1.2])

add_horizontal_rule(doc)

# ── § 07 Timeline ─────────────────────────────────────────────────────────────
add_section_label(doc, '§ 07')
add_h2(doc, 'Project Timeline')

timeline_items = [
    ('Phase 1 · Weeks 1–3',   'Environment & Infrastructure Finalization',
     'Angel finalizes ScissorGraspEnv, FSM stages, and recorder. Jordan reviews interface documentation and confirms dataset schema compatibility with OpenVLA ingestion requirements.',
     'Angel', ANGEL_HUE),
    ('Phase 2 · Weeks 4–6',   'Track A Dataset Collection & Track B Conversion',
     'Angel collects 200+ demonstrations via FSM scripting. Jordan develops and validates HDF5→RLDS conversion pipeline on a pilot batch of 20 episodes.',
     'Angel + Jordan', STEEL),
    ('Phase 3 · Weeks 7–10',  'Parallel Training Runs',
     'Angel trains Diffusion Policy on curated dataset. Jordan fine-tunes OpenVLA-7B with LoRA on converted dataset. Both tracks operate independently on their own hardware.',
     'Angel + Jordan', STEEL),
    ('Milestone · Week 10',   'Mid-Review Checkpoint',
     'Preliminary evaluation results shared between tracks. Evaluation suite standardized on Angel\'s 50-episode benchmark protocol.',
     'Joint', SHARED_HUE),
    ('Phase 4 · Weeks 11–13', 'Ablations & Sim-to-Sim Transfer Evaluation',
     'Both tracks run physics-variant transfer evaluation (±30% stiffness, friction sweep). Jordan conducts language prompt ablations. Angel conducts modality ablations.',
     'Angel + Jordan', STEEL),
    ('Phase 5 · Weeks 14–15', 'Cross-Track Analysis & Report Writing',
     'Joint cross-track comparison section authored. Individual track reports written separately. Final submission prepared with separate Track A and Track B sections clearly attributed.',
     'Joint (comparison) + individual', SHARED_HUE),
    ('Deliverable · Week 16', 'Final Submission',
     'Complete dual-track report, code repositories with clearly labelled ownership per module, demonstration videos, and evaluation datasets submitted.',
     'Joint Submission', SHARED_HUE),
]

for phase, title, desc, owner, color in timeline_items:
    p_phase = doc.add_paragraph()
    rp = p_phase.add_run(phase)
    rp.font.name = 'Courier New'
    rp.font.size = Pt(7.5)
    rp.font.color.rgb = TEXT_MUTED
    rp.font.bold = True
    para_space(p_phase, before=120, after=20)

    p_title = doc.add_paragraph()
    rt = p_title.add_run(title)
    rt.font.name = 'Calibri'
    rt.font.size = Pt(10.5)
    rt.font.bold = True
    rt.font.color.rgb = NAVY
    para_space(p_title, before=0, after=20)

    p_desc = doc.add_paragraph()
    rd = p_desc.add_run(desc)
    rd.font.name = 'Calibri'
    rd.font.size = Pt(9.5)
    rd.font.color.rgb = TEXT_MUTED
    p_desc.paragraph_format.left_indent = Inches(0.2)
    para_space(p_desc, before=0, after=20)

    p_owner = doc.add_paragraph()
    ro = p_owner.add_run(f'Owner: {owner}')
    ro.font.name = 'Courier New'
    ro.font.size = Pt(8)
    ro.font.color.rgb = color
    ro.font.bold = True
    p_owner.paragraph_format.left_indent = Inches(0.2)
    para_space(p_owner, before=0, after=60)

add_horizontal_rule(doc)

# ── § 08 Usage Terms ──────────────────────────────────────────────────────────
add_section_label(doc, '§ 08')
add_h2(doc, 'Infrastructure Usage Terms & IP Agreement')

add_info_box(doc,
    'Binding Conditions',
    'The following terms govern Jordan\'s use of Angel\'s simulation infrastructure. These are not informal expectations — they constitute the agreed research collaboration terms and should be referenced in any joint submission, course documentation, or publication.',
    bg=WARNING_BG, border_color=GOLD)
spacer(doc)

add_h3(doc, 'Permitted Use')
add_body(doc, 'Jordan may use Angel\'s simulation environment, FSM demonstrations, and evaluation infrastructure solely for the purpose of developing and evaluating Track B (OpenVLA) as described in this proposal. Use is limited to the duration of this Special Implementation Topic project.')

add_h3(doc, 'Attribution Requirements')
add_body(doc, 'Any paper, report, slide deck, or repository README that uses or describes Angel\'s infrastructure components must include the statement: "The simulation environment, FSM task logic, and demonstration recording pipeline used in this work were developed by Angel [surname] as an independent contribution to Track A of this project." This statement must appear in a Methods section, Acknowledgements section, or equivalent, not solely in a footnote.')

add_h3(doc, 'Prohibited Actions')
add_body(doc, 'Jordan may not: (a) modify Angel\'s core environment files and submit modified versions as Jordan\'s own work; (b) represent the shared infrastructure as a jointly designed system in any graded submission without Angel\'s written agreement; (c) use Angel\'s demonstration dataset for purposes beyond Track B training and evaluation within this project without prior written consent.')

add_h3(doc, 'Repository Structure & Permissions')
add_attrib_table(doc,
    ['Path', 'Owner', 'Jordan May Modify?'],
    [
        ('envs/scissor_grasp/',       ('Angel', ANGEL_HUE),   'No — read-only import'),
        ('fsm/surgical_fsm.py',       ('Angel', ANGEL_HUE),   'No — read-only import'),
        ('recorder/',                  ('Angel', ANGEL_HUE),   'No'),
        ('training/diffusion_policy/',('Angel', ANGEL_HUE),   'No'),
        ('training/openvla/',          ('Jordan', JORDAN_HUE), 'Yes — Jordan\'s work'),
        ('data_conversion/',           ('Jordan', JORDAN_HUE), 'Yes — Jordan\'s work'),
        ('eval/cross_track/',          ('Joint',  SHARED_HUE), 'Both, by agreement'),
    ],
    col_widths=[2.0, 1.2, 2.6])

add_horizontal_rule(doc)

# ── § 09 Significance ─────────────────────────────────────────────────────────
add_section_label(doc, '§ 09')
add_h2(doc, 'Research Significance')

add_owner_box(doc, 'Track A Impact', 'Diffusion Policy Contributions',
    'Establishes a reproducible baseline for contact-rich thin-instrument manipulation using a state-of-the-art behavior cloning approach. The custom Isaac Lab environment and FSM-curated dataset are independently publishable artifacts that can serve as benchmarks for future work.',
    ANGEL_HUE)
spacer(doc)
add_owner_box(doc, 'Track B Impact', 'OpenVLA Contributions',
    'Demonstrates the feasibility of adapting large pretrained vision-language-action models to specialized surgical manipulation domains via LoRA fine-tuning. Characterizes the trade-off between pretraining generality and task-specific precision on a clinically motivated benchmark.',
    JORDAN_HUE)
spacer(doc)

add_h3(doc, 'Joint Significance — Cross-Track Comparative Value', color=SHARED_HUE)
add_body(doc, 'The parallel experimental design — identical task, identical evaluation suite, shared demonstration data, independent policy architectures — enables a controlled algorithm comparison that neither project could achieve alone. This comparison directly addresses an open question in imitation learning: whether the inductive biases of a generative diffusion model versus a pretrained multimodal transformer lead to meaningfully different failure modes on precision manipulation tasks.')
add_body(doc, 'Results from both tracks may inform future work on hybrid approaches, e.g., using OpenVLA for high-level task decomposition and Diffusion Policy for low-level motor control, a promising direction for surgical autonomy systems.')

add_h3(doc, 'Customization & Maintenance Advantages')
add_body(doc, 'Both approaches are designed with field customization in mind. The modular FSM architecture allows new surgical subtasks to be added by defining additional stages without rewriting the core environment. Fine-tuned models can be updated with new demonstration data without full retraining. This matters in clinical deployment contexts where instrument designs evolve and sterilization-compatible (i.e., low-physical-contact) autonomy is a maintenance advantage over purely manual operation.')

add_horizontal_rule(doc)

# ── § 10 Agreement & Signatures ───────────────────────────────────────────────
add_section_label(doc, '§ 10')
add_h2(doc, 'Agreement & Sign-Off')
add_body(doc, 'By signing below, both researchers confirm that they have read and agree to the contribution attribution, usage terms, and project scope defined in this proposal. Both parties acknowledge that the simulation environment, FSM logic, and demonstration recorder are the sole intellectual property and academic contribution of Angel, and that Jordan\'s independent contribution is limited to Track B as defined in § 05.')

sig_tbl = doc.add_table(rows=3, cols=2)
sig_tbl.alignment = WD_TABLE_ALIGNMENT.LEFT

for ci, (name, role, color) in enumerate([
    ('Angel', 'Track A Principal\nEnvironment · FSM · Recorder · Diffusion Policy', ANGEL_HUE),
    ('Jordan', 'Track B Principal\nOpenVLA Adaptation & Language Conditioning', JORDAN_HUE),
]):
    cell = sig_tbl.cell(0, ci)
    pn = cell.paragraphs[0]
    rn = pn.add_run(name)
    rn.font.name = 'Georgia'
    rn.font.size = Pt(13)
    rn.font.bold = True
    rn.font.color.rgb = color
    para_space(pn, before=120, after=20)

    cell2 = sig_tbl.cell(1, ci)
    pr = cell2.paragraphs[0]
    rr = pr.add_run(role)
    rr.font.name = 'Calibri'
    rr.font.size = Pt(9)
    rr.font.color.rgb = TEXT_MUTED
    para_space(pr, before=0, after=80)

    cell3 = sig_tbl.cell(2, ci)
    ps = cell3.paragraphs[0]
    rs = ps.add_run('Signature: _______________________________    Date: ___________')
    rs.font.name = 'Calibri'
    rs.font.size = Pt(9)
    rs.font.color.rgb = TEXT_MUTED
    para_space(ps, before=40, after=80)

for row in sig_tbl.rows:
    for cell in row.cells:
        tc = cell._tc
        tcPr = tc.get_or_add_tcPr()
        tcBorders = OxmlElement('w:tcBorders')
        for side in ['top', 'bottom', 'left', 'right']:
            b = OxmlElement(f'w:{side}')
            b.set(qn('w:val'), 'none')
            b.set(qn('w:sz'), '0')
            b.set(qn('w:space'), '0')
            b.set(qn('w:color'), 'auto')
            tcBorders.append(b)
        tcPr.append(tcBorders)

spacer(doc)
add_body(doc, 'This proposal should be submitted to the supervising professor with both signatures prior to commencing Track B work. A copy should be retained by each researcher and attached as an appendix to the final joint report.')

# Footer note
p_footer = doc.add_paragraph()
para_space(p_footer, before=200, after=0)
pPr = p_footer._p.get_or_add_pPr()
pBdr = OxmlElement('w:pBdr')
top = OxmlElement('w:top')
top.set(qn('w:val'), 'single')
top.set(qn('w:sz'), '4')
top.set(qn('w:space'), '1')
top.set(qn('w:color'), 'D8DCE8')
pBdr.append(top)
pPr.append(pBdr)
rf = p_footer.add_run('Special Implementation Topic · Sim-to-Sim Surgical Imitation Learning  ·  Prepared: September 2026')
rf.font.name = 'Courier New'
rf.font.size = Pt(7.5)
rf.font.color.rgb = TEXT_MUTED

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = r'C:\IsaacLab\scripts\custom\i4h_project\p4\reporting\artifacts\proposal_dual_track.docx'
doc.save(out_path)
print(f'Saved: {out_path}')
