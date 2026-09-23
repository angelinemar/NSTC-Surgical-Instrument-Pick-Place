"""Build an English, read-aloud explanation of the recorded H5 dataset."""
from pathlib import Path
from xml.sax.saxutils import escape

import fitz
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'debug/output/pdf'
QA = ROOT / 'debug/tmp/pdfs/dataset_speaking_script'
INK = colors.HexColor('#173c43')
TEAL = colors.HexColor('#087f8c')
MUTED = colors.HexColor('#52616b')
STYLES = {
    'title': ParagraphStyle('title', fontName='Helvetica-Bold', fontSize=25,
                            leading=30, textColor=INK, spaceAfter=13),
    'body': ParagraphStyle('body', fontName='Helvetica', fontSize=12.5,
                           leading=18.5, textColor=INK, spaceAfter=12),
    'cue': ParagraphStyle('cue', fontName='Helvetica', fontSize=10,
                          leading=14, textColor=MUTED, spaceAfter=18),
    'label': ParagraphStyle('label', fontName='Helvetica-Bold', fontSize=10,
                            leading=14, textColor=TEAL, spaceAfter=12),
    'cell': ParagraphStyle('cell', fontName='Helvetica', fontSize=10.5,
                           leading=15, textColor=INK),
    'small': ParagraphStyle('small', fontName='Helvetica', fontSize=9,
                            leading=13, textColor=MUTED, spaceAfter=9),
}
story = []


def para(text, style='body'):
    story.append(Paragraph(text, STYLES[style]))


def chapter(number, title, cue):
    if story:
        story.append(PageBreak())
    para(f'PRESENTATION SPEAKING SCRIPT  /  {number:02d}', 'label')
    para(title, 'title')
    para(cue, 'cue')


def box(text):
    t = Table([[Paragraph(text, STYLES['body'])]], colWidths=[479])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#eff7f8')),
        ('BOX', (0, 0), (-1, -1), 0.7, colors.HexColor('#bfdbdf')),
        ('LEFTPADDING', (0, 0), (-1, -1), 14),
        ('RIGHTPADDING', (0, 0), (-1, -1), 14),
        ('TOPPADDING', (0, 0), (-1, -1), 12),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
    ]))
    story.extend([t, Spacer(1, 16)])


def table(rows, widths):
    data = [[Paragraph(escape(str(v)), STYLES['cell']) for v in row] for row in rows]
    t = Table(data, colWidths=widths, hAlign='LEFT')
    t.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#eaf4f5')),
        ('LINEBELOW', (0, 0), (-1, -1), 0.4, colors.HexColor('#d5dfe2')),
        ('LEFTPADDING', (0, 0), (-1, -1), 9),
        ('RIGHTPADDING', (0, 0), (-1, -1), 9),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.extend([t, Spacer(1, 16)])


chapter(1, 'What does one H5 file contain?',
        'Opening / six-camera slide. Read the main paragraphs aloud; the small notes are presenter cues.')
para('Our dataset records three main things: what the robot sees, the condition of the robot, and the movement command given to the robot. We record these together at each time step, and save the sequence in an H5 file.')
para('H5, or HDF5, is a file format that can hold many arrays and their descriptions in one file. An array is an organized collection of values, such as image pixels or robot joint positions. Groups inside the file work like folders.')
box('<b>One recorded step:</b><br/>Camera observations + robot condition + movement command')
para('The letter <b>T</b> means the number of recorded time steps in that file. In our example, T is 152. Each camera has 152 RGB frames, and the robot data and actions also have 152 rows.')
para('We use six cameras: front, wrist, top, left, right, and tray. These are six views of the same recorded step. We do not finish recording the front camera before starting the wrist camera.')
para('For example, row 27 contains the camera views and robot information associated with the same step. The recorded observation comes before applying that row\'s action to the simulation.')
para('In this dataset, pick and place are stored as separate H5 segments. The example discussed here is a scalpel pick segment, not the entire pick-and-place sequence.')
para('Presenter cue: say "H-five" or "H-D-F-five". Say "time step" for one recorded step.', 'small')

chapter(2, 'How to read shapes and data types',
        'Six-camera slide / the Shape and Dtype columns in the dataset table.')
para('The word <b>shape</b> describes how the data is arranged. For an RGB camera, the shape is T by 224 by 224 by 3. This means T images, each 224 pixels high and 224 pixels wide, with three color channels: red, green, and blue.')
table([
    ('Shape', 'How to read it'),
    ('(T, 224, 224, 3)', 'T color images; 224 by 224 pixels; 3 RGB channels.'),
    ('(T, 224, 224)', 'T pixel maps; one value per pixel, such as depth or a class ID.'),
    ('(T, 16), (T, 18), (T, 8)', 'T rows, with 16, 18, or 8 values in each row.'),
    ('(T, 3, 3)', 'One 3-by-3 matrix at each recorded step.'),
], [151, 328])
para('The word <b>dtype</b> means the type of number used for storage. RGB uses uint8: whole numbers from zero to 255. Semantic labels use uint16: nonnegative whole numbers. Float16 and float32 can store decimal values, and differ in precision and storage size. A Boolean value stores true or false.')
para('The label <b>six streams</b> means six image sequences, one per camera. The RGB shape shown in the slide applies to each camera separately.')
para('Our recording interval is 20 milliseconds, or 0.02 seconds. That is 50 steps per second of simulation time. A segment with 152 steps therefore represents approximately three seconds of simulated activity. Processing it on the computer can take longer.')
para('The presentation GIF uses sampled frames and slower playback to make the explanation easier to follow. Its playback duration is not the simulation duration. T can also differ between episodes.')

chapter(3, 'What are observations?',
        'Dataset overview / RGB, depth, and semantic explanation.')
para('An <b>observation</b> is information recorded about the scene or the robot at a particular step. In our H5 file, observations is the name of a group that contains several kinds of data.')
para('<b>RGB</b> describes appearance. It answers: what does the scene look like? <b>Depth</b> stores a depth measurement at each pixel, in meters. <b>Semantic segmentation</b> assigns an object class to each pixel. It answers: what kind of object does this pixel belong to?')
box('RGB: appearance.<br/>Depth: depth at each pixel.<br/>Semantic: class identity at each pixel.')
para('The observations group also contains robot_proprio, state, object_type_id, skill_id, and stage_id. These describe the robot condition, the selected target, the skill, and the current movement stage.')
para('The six cameras all have RGB, depth, and semantic data in the checked example. The wrist camera has a naming detail: its RGB is called wrist_rgb, but its semantic array is called grip_b_semantic. Here, grip_b refers to the wrist or gripper camera.')
para('Semantic labels are stored as numbers. The mapping is: zero for background, one for robot, two for tray, three for scalpel, four for scissor, five for love retractor, six for Kelly, and seven for scalpel type two.')
para('The colors in the preview only help us see these classes. Objects other than the target can also have labels. However, an object outside the camera view or hidden behind another object does not have to appear in every frame.')
para('A dataset being inside the observations group does not mean every value is automatically fed into the model. The training pipeline explicitly selects its inputs.')

chapter(4, 'What does robot_proprio mean?',
        'Robot state and commands slide / point to the left-hand robot diagram.')
para('<b>Proprio</b> is short for <b>proprioception</b>. It means information about the body\'s own condition. For example, we can sense that our elbow is bent without looking at it. For a robot, similar information includes joint positions and gripper positions.')
para('In our dataset, robot_proprio contains 16 values at each time step. These are divided into four groups.')
table([
    ('Values', 'Meaning'),
    ('7 joint positions', 'The angular positions of the seven arm joints.'),
    ('2 finger positions', 'The positions of the two gripper fingers.'),
    ('3 position values', 'The x, y, and z position of the end effector.'),
    ('4 orientation values', 'The end-effector orientation, stored as a quaternion.'),
], [151, 328])
para('The <b>end effector</b>, abbreviated EE, is the working end of the robot, where the gripper or tool is attached. Position tells us where it is. Orientation tells us which way it is facing.')
para('A <b>quaternion</b> stores a three-dimensional rotation using four values. In our state and action arrays, these values are ordered qw, qx, qy, and qz. They work together to describe one orientation; they are not four separate angles.')
box('<b>16 values = 7 joints + 2 fingers + 3 position + 4 orientation</b>')
para('The robot drawing on the slide is schematic. It explains the data groups; it is not a reconstruction of the exact recorded joint motion.', 'small')

chapter(5, 'State describes now. Action gives a command.',
        'Robot state and commands slide / compare the left and right panels.')
para('The stored <b>state</b> has 18 values. It combines the 16 robot_proprio values with two additional values: object_type_id and skill_id.')
box('<b>State: 18 values</b><br/>16 robot values + 1 target-type ID + 1 skill ID')
para('The target-type ID identifies the selected instrument. For example, zero means scalpel. The skill ID identifies the task segment: zero means pick, and one means place. Stage ID is separate metadata for a more detailed phase, such as lowering or closing the gripper.')
para('An <b>action</b> has eight values. Three values specify the commanded x, y, and z position. Four specify the commanded orientation. The last value tells the gripper to open or close.')
box('<b>Action: 8 values</b><br/>3 commanded position + 4 commanded orientation + 1 gripper command')
para('In this dataset, plus one means open, and minus one means close. The action is an absolute target pose in the robot-base coordinate frame. It is not simply a small movement added to the previous position.')
para('For example, the measured gripper may still be above the scissors, while the command asks it to move down toward the grasp position. The measured pose and commanded pose can therefore be different. A command does not mean the robot has already reached it.')
para('The phrase <b>robot-base coordinates</b> means that positions and orientations are expressed relative to a coordinate frame attached to the robot base. This provides a common reference for describing the motion.')

chapter(6, 'What is camera calibration?',
        'Calibration slide / follow the arrows from the image to the 3D point.')
para('<b>Camera calibration</b> provides the information needed to connect image measurements with locations in three-dimensional space.')
para('On the slide, a point is marked at pixel coordinates u and v. The letter u describes the horizontal image position, and v describes the vertical position. For example, pixel 204, 177 identifies a location in the image. It does not directly give a position in meters for the robot.')
para('We combine this pixel location with its depth and the camera\'s <b>intrinsics</b>. Intrinsics describe how the camera projects space into an image. They include focal-length parameters and the image-center coordinates, and are stored in a matrix called K.')
para('Using the matching depth convention and intrinsics, we can convert an image measurement into a 3D point expressed relative to the camera.')
para('We then use the <b>camera pose</b>: the position and orientation of the camera relative to the robot base. This transforms the point from camera coordinates into robot-base coordinates.')
box('Pixel + depth + camera intrinsics<br/><b>gives a 3D point in camera coordinates.</b><br/><br/>Camera point + camera pose<br/><b>gives a 3D point in robot-base coordinates.</b>')
para('Calibration helps us locate a point. It does not, by itself, choose the best grasp or prove that the robot can reach that point. Those are separate decisions.')
para('Presenter cue: point first to the marked pixel, then to K, then to camera pose, and finally to the XYZ axes.', 'small')

chapter(7, 'How to read the calibration row',
        'Calibration slide / explain K (T, 3, 3) and pose (T, 3) + (T, 4).')
para('The calibration row may look complicated, but it describes only two main things: the camera\'s projection parameters and the camera\'s pose.')
table([
    ('Stored component', 'Meaning'),
    ('K: (T, 3, 3)', 'One 3-by-3 intrinsics matrix at each recorded step.'),
    ('Position: (T, 3)', 'The camera position, expressed with three coordinates.'),
    ('Orientation: (T, 4)', 'The camera rotation, stored with four quaternion values.'),
    ('Raw K and cropped K', 'Intrinsics for the original image and for the saved image crop.'),
], [169, 310])
para('Storing calibration at every step does not mean every value changes at every step. A fixed camera may keep the same pose. The wrist camera moves with the robot, so its pose can change during the episode.')
para('Our saved images are 224 by 224 pixel crops. Cropping changes where the image center lies in the saved image coordinates. That is why we retain both the original intrinsics and the intrinsics adjusted for the crop.')
para('When working with the saved 224 by 224 images, we must use the intrinsics that match those images. Using the original image parameters without accounting for the crop can produce an incorrect 3D location.')
para('The calibration data is stored for each of the six cameras. In the H5 structure, it appears under camera_calibration, with separate entries for each camera.')
box('<b>Remember:</b> image pixels, depth, and calibration must refer to the same camera and matching image geometry.')

chapter(8, 'What does the training slide mean?',
        'Current sensor-only training pipeline slide / explain the three horizontal rows.')
para('A <b>policy</b> is the model that learns which movement command to produce from the information it receives. On this slide, each row has a different role in training.')
para('<b>The first row is the policy input.</b> Our current sensor-only pipeline uses RGB images from six cameras plus robot_proprio. These tell the model what the scene looks like and how the robot is currently configured.')
para('<b>The second row is the action target.</b> The recorded demonstration actions provide examples of the commands the model should learn to produce. During training, predicted actions can be compared with these recorded actions.')
para('<b>The third row is the segmentation target.</b> Semantic class masks provide the correct class for each labeled pixel. The segmentation prediction can be compared with these masks to learn pixel-level recognition.')
box('Input: information given to the model.<br/>Training target: the reference answer used for learning.<br/>Prediction: the answer produced by the model.')
para('The word target has two meanings here. A <b>target instrument</b> is the object we want to pick up. A <b>training target</b> is a reference answer used to train the model. These should not be confused.')
para('The phrase sensor-only describes the selected policy inputs. It does not mean the H5 file contains only those inputs. Depth, calibration, task IDs, and simulator ground truth can still be stored for other uses. They are not all direct policy inputs in the pipeline shown.')
para('Semantic labels act as supervision in this diagram. The arrows do not mean the policy receives the correct semantic answer as an ordinary input during deployment.')

chapter(9, 'Other H5 fields and a closing script',
        'Optional questions / use the closing paragraph to finish the dataset explanation.')
para('Some H5 fields help us organize and check the recording. Stage names and stage IDs identify movement phases. The step_ids values count steps within a stage and restart when the stage changes. To follow the full stored sequence, use the dataset row order.')
para('Dones marks the end of a segment, and rewards stores a reward value for each step. Normalization statistics store means and standard deviations for state and action values.')
para('The supervision_gt group contains label or teacher data, such as conditioning masks and a reference grasp pose. The debug_gt group contains simulator ground truth for analysis, including object poses and a target-slot location. These are different from the robot\'s own measured condition.')
para('H5 attributes store descriptions and recording metadata, such as the target instrument, success status, recording interval, class mapping, spawn information, and tray occupancy.')
para('Training reads the numerical arrays from HDF5 files. PNG previews and GIF animations are for inspection and presentation. Deleting a preview does not remove the semantic arrays inside the H5 file.')
para('CLOSING SCRIPT', 'label')
box('At each step, our dataset records what six cameras see, the condition of the robot, and the movement command. The model learns to produce commands from the images and robot condition. Semantic labels support learning to recognize objects, while calibration connects image measurements to 3D robot coordinates.')
para('Basis of this script: inspected scalpel pick file episode_000009.h5 from panel run 20260918_231630_309506, with 152 samples. The training explanation follows training/export_sensor_only.py and training/sensor_policy.py. Shapes and values here describe that inspected example; segment lengths may differ.', 'small')


def page_style(canvas, doc):
    canvas.setFillColor(colors.white)
    canvas.rect(0, 0, A4[0], A4[1], fill=1, stroke=0)
    canvas.setStrokeColor(colors.HexColor('#d5dfe2'))
    canvas.line(58, 51, A4[0]-58, 51)
    canvas.setFont('Helvetica', 9)
    canvas.setFillColor(MUTED)
    canvas.drawString(58, 34, 'ROBOT DATASET EXPLAINED  |  English speaker notes')
    canvas.drawRightString(A4[0]-58, 34, f'{doc.page} / 9')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    QA.mkdir(parents=True, exist_ok=True)
    path = OUT / 'robot_dataset_explained_english_speaking_script.pdf'
    doc = SimpleDocTemplate(str(path), pagesize=A4, rightMargin=58,
                            leftMargin=58, topMargin=49, bottomMargin=68,
                            title='Robot Dataset Explained - English Speaking Script',
                            author='P4 Project', pageCompression=1)
    doc.build(story, onFirstPage=page_style, onLaterPages=page_style)
    pdf = fitz.open(path)
    assert len(pdf) == 9, f'Unexpected overflow: {len(pdf)} pages'
    for index, page in enumerate(pdf):
        page.get_pixmap(matrix=fitz.Matrix(1.2, 1.2)).save(QA / f'page_{index+1:02d}.png')
        for block in page.get_text('blocks'):
            assert 0 <= block[0] < block[2] <= A4[0]+1, (index, block)
            assert 0 <= block[1] < block[3] <= A4[1]+1, (index, block)
    print(path.resolve())
    print(f'Validated {len(pdf)} pages; rendered all pages to {QA}')


if __name__ == '__main__':
    main()
