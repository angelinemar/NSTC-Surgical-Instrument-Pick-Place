"""English presenter notes for slides 8-41 of the user's 20092026.pdf."""
from pathlib import Path
import json
import re

import fitz
from PIL import Image as PILImage, ImageDraw
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'debug/output/pdf'
QA = ROOT / 'debug/tmp/pdfs/20092026_script'
SOURCE = ROOT / 'debug/tmp/pdfs/20092026_source'
INK = colors.HexColor('#173c43')
TEAL = colors.HexColor('#087f8c')
MUTED = colors.HexColor('#596871')

# Each entry follows the original slide number, including transition slides.
SLIDES = {
8: ('Introducing the new environment', 'Introduce the next section.', [
    'Now I will present Phase Three. This part focuses on our updated hospital simulation environment. I will explain the workspace, the five instruments, the recorded data, and the control panel used to collect demonstrations.',
    'After that, I will show the simulation preview and our next steps.'
]),
9: ('The table, robot, and workspace', 'Point to the simulation, then the top-view layout.', [
    'This slide shows our updated workspace. The robot works above a hospital table, and the tray is placed beside the instrument area.',
    'The table is 2.2 meters long and 0.9 meters wide. The spawning area is divided into ten cells, each 20 by 20 centimeters. We use these cells to collect examples from different starting locations.',
    'The tray has five fixed instrument slots. The robot-base coordinates in the table describe its location, not its physical size.'
]),
10: ('Instrument one: scissors', 'Point to the top view and side view. Dimensions are model measurements.', [
    'This slide shows the scissors model. The top view explains its length and width. The side view explains its overall height.',
    'The measured length is about 134 millimeters, the width is 51.8 millimeters, and the height is 6.7 millimeters. The arrows show which direction is being measured, and the dotted lines mark the ends of each measurement.',
    'These values describe the model at its simulation scale. They are not manufacturer specifications. The small RGB image shows how the instrument appears in a recorded camera frame.'
]),
11: ('Instrument two: scalpel type two', 'This is the narrow scalpel_type2 model.', [
    'This is the second scalpel model, called scalpel type two in our dataset. It is about 150.2 millimeters long, 9.4 millimeters wide, and 2.8 millimeters high.',
    'Compared with the scissors, this model has a much narrower shape. The top and side views make that difference easier to see.',
    'We keep it as a separate instrument class, so the dataset can distinguish it from the other scalpel model.'
]),
12: ('Instrument three: scalpel', 'Point to the wider blade profile.', [
    'This slide shows the main scalpel model. Its length is about 150.1 millimeters, its width is 13.9 millimeters, and its overall height is 8.0 millimeters.',
    'It has a similar length to scalpel type two, but its shape and width are different. That is why they have separate class labels.',
    'The height here describes the full model profile in the side view. It should not be interpreted as the thickness of the blade at one small point.'
]),
13: ('Instrument four: Love retractor', 'Point to the hook in the side view.', [
    'This instrument is called the Love retractor in our project. It is about 150.2 millimeters long and 6.8 millimeters wide.',
    'Its overall height is about 10.4 millimeters. The curved end contributes to this height, which is why the side view is useful.',
    'The model measurements help us describe its shape and the space it occupies in the simulated workspace.'
]),
14: ('Instrument five: Kelly', 'Use the project class name; describe only the model shown.', [
    'This is the model labeled Kelly in our project. Its measured length is about 131.5 millimeters, its width is 18.1 millimeters, and its overall height is 20.5 millimeters.',
    'The side view shows the raised end clearly. Again, this height measures the full profile, including the bend. It does not mean the metal is 20.5 millimeters thick everywhere.',
    'These five models form the instrument classes used in our recording setup.'
]),
15: ('Instrument labels and tray slots', 'Point to the semantic ID column, then the tray slot column.', [
    'Each instrument has a semantic class ID and a fixed tray slot. These numbers serve different purposes. The semantic ID tells us what the object is. The slot number tells us where it belongs in the tray.',
    'For example, scalpel is semantic class three, but it belongs to slot zero. The other instrument slots are one for scissors, two for Love retractor, three for Kelly, and four for scalpel type two.',
    'The slot assignment follows instrument class, not the order in which objects are picked. Background, robot, and tray also have semantic labels, but they do not have instrument slots.'
]),
16: ('Six camera viewpoints', 'Point to the front, top, side, tray, and wrist camera entries.', [
    'We record six camera views: front, wrist, top, left, right, and tray. They give different views of the same workspace.',
    'The side and top cameras provide alternative viewpoints when an object is difficult to see from the front. The tray camera shows the placement area, while the wrist camera follows the robot hand.',
    'The coordinate table describes the camera setup. Camera coordinates must be interpreted in their defined reference frame. The wrist camera moves with the robot, so its pose during recording is not simply one fixed world position.'
]),
17: ('How the camera images are cropped', 'Trace the border removed from the original frame.', [
    'The camera renders an image that is 448 pixels wide and 336 pixels high. We save the central 224 by 224 pixel region.',
    'This removes 112 pixels from each side and 56 pixels from the top and bottom. It is a center crop, not a resize of the entire image.',
    'The result gives every camera the same saved image size. However, objects outside the crop are not visible in that saved view. The camera calibration must also match this cropped image.'
]),
18: ('The stages of a pick-and-place demonstration', 'Follow the stage images in order. The chart is a recorded example.', [
    'We divide the demonstration into named stages so that the motion is easier to inspect.',
    'First, OPEN HOVER moves the open gripper above the target. LOWER PRE brings it closer. LOWER GRASP lowers it to the grasping height. CLOSE closes the gripper, and LIFT CLEAR raises the instrument away from the table.',
    'For placement, MOVE TO TARGET carries the instrument toward its tray slot. LOWER PLACE lowers it into position. OPEN releases it, and RETREAT moves the gripper away.',
    'The chart and table describe the timing and movement of the illustrated recording. Path length is the distance traveled along the motion. A straight-line distance only compares the start and end points, so those two measurements can differ.',
    'During CLOSE and OPEN, the gripper fingers move while the arm can stay almost in the same position. That is why the end-effector travel can be small in these stages.'
]),
19: ('T means the number of recorded steps', 'Point to T = 152, then to the six camera images.', [
    'An H5 file stores a sequence of recorded steps. T means the number of steps in that file. In this example, T is 152.',
    'Each camera therefore has 152 RGB images. The robot state and action arrays also have matching rows. At one row, we can read the camera views and robot information for that step.',
    'The shape T by 224 by 224 by three means T images, each 224 pixels high and wide, with three color channels: red, green, and blue. This shape applies to each camera separately.',
    'The recording interval is 20 milliseconds, or 50 steps per second of simulation time. The GIF is slowed down for explanation. Its playback duration is not the simulation duration.'
]),
20: ('RGB, depth, and semantic labels', 'Point to the same marked pixel in all three views.', [
    'These three images describe the same camera view in different ways. RGB shows appearance. Depth stores a depth measurement in meters. Semantic segmentation tells us which object class each pixel belongs to.',
    'The marked pixel helps us compare the same location. Its RGB value describes color, its depth value describes depth, and its semantic value identifies the class. Here, class three means scalpel.',
    'Depth and semantic arrays have the shape T by 224 by 224, because each pixel has one stored value. Float16 can store decimal depth values. Uint16 stores whole-number class IDs.',
    'The semantic colors are only a visual display of those IDs. The H5 file stores the labels themselves, rather than depending on the preview colors.'
]),
21: ('Understanding 16, 18, and 8', 'Explain slowly. Use your hand to demonstrate position versus orientation.', [
    'The numbers 16, 18, and 8 are counts of stored values at one step. Think of one row in a spreadsheet: one row has sixteen columns, another has eighteen, and another has eight. They do not count robots or movements.',
    'On the left, robot proprio means information about the robot\'s own condition. It contains seven arm-joint positions, two finger positions, three values for the hand position, and four values for the hand orientation. Seven plus two plus three plus four gives sixteen.',
    'Tool position means where the robot hand is. We describe that location with X, Y, and Z, measured relative to the robot base. Here, tool means the robot\'s working end, not the exact pose of the surgical instrument.',
    'Orientation means which way the hand faces. Imagine keeping your hand in the same place while turning your palm upward. The position stays almost the same, but the orientation changes. We store one orientation using four numbers called a quaternion. They are not four separate angles.',
    'State contains those sixteen robot values plus two labels: the target instrument type and the skill, pick or place. That makes eighteen values.',
    'On the right, action means the command. It contains three values for the desired hand position, four for the desired orientation, and one gripper command. Three plus four plus one gives eight. Plus one opens the gripper, and minus one closes it.',
    'So the left side means: what is the robot doing now? The right side means: where do we want the robot hand to go? A recorded command is not proof that the hand has already reached that position.'
]),
22: ('Calibration: from a pixel to a 3D location', 'Follow pixel + depth, then K, then camera pose, then robot-base XYZ.', [
    'Calibration connects a location in an image with a location in three-dimensional space. A pixel address alone cannot tell the robot where to move in meters.',
    'The letters u and v describe horizontal and vertical pixel coordinates. We combine that pixel location with depth and camera intrinsics. Intrinsics describe how the camera projects the scene into an image. They include focal-length parameters and the image center, stored in a matrix called K.',
    'With the matching depth convention, this gives a point in camera coordinates. We then use the camera pose, meaning where the camera is and which way it faces, to express that point relative to the robot base.',
    'K has the shape T by three by three: one small matrix for each step. Camera pose uses three position values and four orientation values at each step. Fixed cameras may keep the same values, while the wrist camera moves.',
    'We save intrinsics for both the original image and the cropped image. When using our saved 224 by 224 images, we must use the calibration that matches that crop.',
    'Calibration helps locate a point. It does not automatically choose the best grasp or guarantee that the robot can reach the point.'
]),
23: ('What the model receives and learns', 'Emphasize the difference between model input and a training answer.', [
    'A policy is the model that learns which movement command to produce. In the current pipeline, its input is RGB from six cameras plus robot proprio. These describe what the robot sees and its own current condition.',
    'Recorded actions are examples of the commands we want it to learn. Semantic masks provide the correct object class for each pixel and support segmentation learning.',
    'Using recorded actions as training answers is not the same as giving the model the answer during a test. During learning, it studies examples. During testing, it must produce a new action from the images and robot condition.',
    'The policy input shown here does not include the exact instrument pose from the simulator. Any simulator information used to create expert demonstrations must be described separately from the learned policy\'s inputs.',
    'Depth, calibration, and other labels can still be stored in H5 without being direct inputs to this policy. A successful expert demonstration also does not, by itself, prove that the trained model succeeds.'
]),
24: ('Introducing the recorder control panel', 'Transition to the interface section.', [
    'Next, I will introduce the two-dimensional control panel for the recorder. This interface helps us configure a run, inspect the scene, monitor progress, and find the saved recordings.',
    'I will explain the main settings one by one.'
]),
25: ('Single collection mode', 'Point to Single in the Collection menu.', [
    'Single mode lets us request a total number of successful episodes for the selected target. For example, entering fifty means that the goal is fifty saved successes.',
    'It does not mean that the recorder will stop after fifty attempts. Failed attempts do not count toward the requested successful total.'
]),
26: ('Grid-cycle collection mode', 'Point to the ten cells and the number of rounds.', [
    'Grid-cycle mode spreads the target examples across the ten workspace cells. One round aims to collect one successful episode from each cell.',
    'For example, fifty rounds across ten cells gives a goal of five hundred successful episodes, with fifty successes per cell.',
    'If an attempt fails, that cell is retried instead of counting the failure as completed coverage. This balances starting locations, but it does not guarantee coverage of every possible pose.'
]),
27: ('Manual or automatic spawning', 'Point to the Spawn menu. The missing illustration does not affect the explanation.', [
    'Spawn means placing the instruments in their starting positions. In manual mode, I choose their positions through the interface.',
    'Manual positioning still has workspace and separation checks. It does not allow every possible location without restrictions.',
    'In automatic mode, the recorder randomizes starting positions within the configured rules. Grid-cycle collection uses automatic spawning within the selected cell.'
]),
28: ('Selecting the target instrument', 'Read the five choices once.', [
    'This menu selects which instrument the robot should handle. The five choices are scalpel, scissors, Love retractor, Kelly, and scalpel type two.',
    'The selected instrument becomes the target for that run. The other instrument classes can appear as distractors on the table or as objects already in the tray.'
]),
29: ('Choosing which skills to save', 'Point to Both, Pick, and Place.', [
    'This setting selects which skill recordings to save. Both saves pick and place. Pick saves the pick segment, and Place saves the place segment.',
    'When both are saved, pick and place are stored separately. One H5 file should therefore not automatically be interpreted as a complete pick-and-place sequence.',
    'This menu describes the saved output. It should not be read as a promise that the robot can skip every setup movement needed for the selected skill.'
]),
30: ('Reading the collection target correctly', 'Compare the two fields showing 50.', [
    'The same number means different things depending on collection mode. In Single mode, fifty means fifty successful episodes in total.',
    'In Grid Cycles mode, fifty means fifty rounds. With ten cells, that becomes five hundred successful episodes.',
    'This comparison is useful before launching a long run, because it makes the actual collection goal clear.'
]),
31: ('The workspace display and recording log', 'Point to the tray, spawn grid, and log area.', [
    'The left side provides a top-view workspace display. It shows the robot, the tray slots, and the instrument spawning area. This helps us inspect the arrangement before recording.',
    'The instrument outlines and surrounding circles help with placement and separation. The log on the right reports recorder events, progress, and failures.',
    'The workspace display is a layout view or a settled-scene snapshot, depending on the recorder state. It should not always be interpreted as a live reconstruction of every robot movement.'
]),
32: ('Starting, inspecting, and stopping a run', 'Point to each button in order.', [
    'Launch and Record starts the setup and recording workflow. Prepare Preview sets up the scene and allows the objects to settle, so I can inspect their positions first.',
    'Start Prepared begins recording after that preparation. Discard Attempt rejects the current unfinished attempt. Stop Run requests a stop at the next recorder check.',
    'The stop button is a software stop request, rather than a claim that every operation stops instantly.'
]),
33: ('Recorder status lights', 'Point to the three indicators.', [
    'These indicators show the current recorder phase. Reset or Wait means the system is preparing the scene or waiting. Recording means that an attempt is being recorded. End or Saving means that the attempt is being finalized or saved.',
    'They make it easier to understand the current activity without reading every log message. The saved-success counter confirms how many successful recordings have actually been stored.'
]),
34: ('Choosing the initial tray contents', 'Point to Random, Empty, Full, and Manual.', [
    'The tray mode controls which non-target instruments start in the tray. Random chooses a subset of the other four classes. Empty starts with no instruments in the tray.',
    'Full puts the four non-target classes in their assigned tray slots. Manual lets me choose the tray contents using the counters below.',
    'The target stays outside the tray so that the robot has an object to pick and place. These modes create different starting scene arrangements.'
]),
35: ('What the zero and one counters mean', 'Point to an instrument counter.', [
    'These counters apply only in manual tray mode. Zero means the instrument starts on the table. One means it starts in its assigned tray slot.',
    'They do not request many copies of the same instrument. The current setup uses one physical instance per class.',
    'The target must remain zero, because it should not already occupy its destination slot before the task starts.'
]),
36: ('A simple manual-tray example', 'This repeats the previous slide. Use an example rather than repeating every definition.', [
    'For example, if scalpel is the target, its counter stays at zero. I can set the scissors counter to one to put the scissors in its tray slot before recording.',
    'The other non-target counters determine which instruments remain on the table. This lets us control the initial clutter and tray occupancy.'
]),
37: ('Finding saved recordings', 'Point to the selected path and the Pick H5 / Place H5 buttons.', [
    'This tab helps us open the selected recording folder. It provides shortcuts to the run folder, pick H5 files, place H5 files, and available previews.',
    'H5 files contain the numerical recordings used by the training pipeline. GIFs and preview images help people inspect the data. They are not required training inputs.',
    'A preview folder may be empty if previews have not been exported or have already been removed. That does not, by itself, mean the H5 recording is missing.'
]),
38: ('Introducing the simulation preview', 'Transition to the demonstration.', [
    'Now I will show the simulation preview. This connects the workspace, recorder settings, and robot motion that I have just explained.'
]),
39: ('Narrating the simulation preview', 'If the original slide contains a video, play it and pause your speech as needed.', [
    'The preview shows the hospital workspace together with the recorder panel. The robot operates in the instrument area, while the panel helps us monitor the collection process.',
    'During the demonstration, we can follow the target instrument, the gripper movement, and the recorder status.',
    'This illustrates the simulation and recording workflow. It should not be presented as evidence of a successfully trained policy unless that particular run was performed by the trained model.'
]),
40: ('Current progress and next steps', 'Say recording is ongoing. The slide lists planned dates, not confirmed training results.', [
    'As of this progress update, recording toward the two-thousand-five-hundred-example goal is still ongoing. The goal should not be described as fully completed yet.',
    'The next planned step is to start object-recognition training on September twenty-first. The following progress presentation is planned for September twenty-third.',
    'The immediate focus is to finish and check the collected data, then move into model training and evaluation.'
]),
41: ('Closing', 'Pause, thank the audience, and invite questions.', [
    'That concludes my progress update. I have presented the updated simulation environment, the instrument and dataset structure, and the recorder control panel.',
    'Thank you for your attention. I am happy to take your questions.'
]),
}

GROUPS = [(8,9),(10,11),(12,13),(14,15),(16,17),(18,),
          (19,20),(21,),(22,),(23,),(24,25),(26,27),(28,29),
          (30,31),(32,33),(34,35),(36,37),(38,39),(40,41)]

styles = {
    'title': ParagraphStyle('title', fontName='Helvetica-Bold', fontSize=16,
                            leading=20, textColor=INK, spaceAfter=7),
    'number': ParagraphStyle('number', fontName='Helvetica-Bold', fontSize=10,
                             leading=13, textColor=TEAL, spaceAfter=5),
    'cue': ParagraphStyle('cue', fontName='Helvetica-Oblique', fontSize=9.3,
                          leading=13, textColor=MUTED),
    'body': ParagraphStyle('body', fontName='Helvetica', fontSize=12,
                           leading=17, textColor=INK, spaceAfter=9),
}


def footer(canvas, doc):
    canvas.setFillColor(colors.white)
    canvas.rect(0,0,A4[0],A4[1],fill=1,stroke=0)
    canvas.setFont('Helvetica',8.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(49,A4[1]-30,'20 SEPTEMBER 2026  |  ENGLISH PRESENTATION SCRIPT  |  SLIDES 8-41')
    canvas.setStrokeColor(colors.HexColor('#d7e0e2'))
    canvas.line(49,49,A4[0]-49,49)
    canvas.drawString(49,33,'Read the main paragraphs aloud. Italic text is a presenter cue.')
    canvas.drawRightString(A4[0]-49,33,f'{doc.page} / {len(GROUPS)}')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    QA.mkdir(parents=True,exist_ok=True)
    assert sorted(SLIDES)==list(range(8,42))
    story=[]
    for group_index,group in enumerate(GROUPS):
        if group_index: story.append(PageBreak())
        for position,n in enumerate(group):
            title,cue,paragraphs=SLIDES[n]
            if position: story.append(Spacer(1,20))
            slide_path=SOURCE/f'slide_{n:02d}.png'
            with PILImage.open(slide_path) as im: iw,ih=im.size
            heading=[Paragraph(f'SLIDE {n:02d}',styles['number']),
                     Paragraph(title,styles['title']),Paragraph(cue,styles['cue'])]
            t=Table([[heading,Image(str(slide_path),width=128,height=128*ih/iw)]],
                    colWidths=[359,138],hAlign='LEFT')
            t.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),
                ('LEFTPADDING',(0,0),(-1,-1),0),('RIGHTPADDING',(0,0),(-1,-1),10),
                ('TOPPADDING',(0,0),(-1,-1),0),('BOTTOMPADDING',(0,0),(-1,-1),0)]))
            story.extend([t,Spacer(1,12)])
            for p in paragraphs: story.append(Paragraph(p,styles['body']))
    path=OUT/'20092026_slides_08_to_41_english_script.pdf'
    doc=SimpleDocTemplate(str(path),pagesize=A4,leftMargin=49,rightMargin=49,
        topMargin=53,bottomMargin=66,title='20 September 2026 - English Speaking Script - Slides 8 to 41',
        author='P4 Project',pageCompression=1)
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    pdf=fitz.open(path)
    assert len(pdf)==len(GROUPS),(len(pdf),len(GROUPS))
    for i,page in enumerate(pdf):
        txt=page.get_text()
        for n in GROUPS[i]: assert f'SLIDE {n:02d}' in txt,(i,n)
        for block in page.get_text('blocks'):
            assert block[0]>=0 and block[2]<=A4[0]+1 and block[1]>=0 and block[3]<=A4[1]+1,(i,block)
        page.get_pixmap(matrix=fitz.Matrix(1.3,1.3)).save(QA/f'page_{i+1:02d}.png')
    for start in range(1,len(pdf)+1,4):
        sheet=PILImage.new('RGB',(1300,1840),'#d4d9dc')
        for j,num in enumerate(range(start,min(start+4,len(pdf)+1))):
            im=PILImage.open(QA/f'page_{num:02d}.png').convert('RGB')
            im.thumbnail((648,918))
            sheet.paste(im,((j%2)*650,(j//2)*920))
        sheet.save(QA/f'contact_{start:02d}.png')
    words=sum(len(re.findall(r'\b[\w-]+\b',p)) for _,_,ps in SLIDES.values() for p in ps)
    (QA/'validation.json').write_text(json.dumps(dict(pages=len(pdf),slides=sorted(SLIDES),
        spoken_words=words,estimated_minutes_at_120_wpm=round(words/120,1),source=r'C:\Users\user\Downloads\20092026.pdf'),indent=2))
    print(path.resolve())
    print(f'{len(pdf)} pages; all 34 slides covered; {words} spoken words; about {words/120:.1f} minutes at 120 words/minute.')


if __name__=='__main__': main()
