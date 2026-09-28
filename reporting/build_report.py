"""Dimensioned P4 report; vector diagrams from extracted USD/H5 facts."""
import json,math
from pathlib import Path
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor,white
from reportlab.lib.pagesizes import landscape,A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Table,TableStyle,Paragraph
from reportlab.lib.styles import ParagraphStyle

HERE=Path(__file__).resolve().parent
ROOT=Path(r'C:\IsaacLab\scripts\custom\i4h_project\p4')
D=json.loads((HERE/'report_data.json').read_text(encoding='utf-8')); L=D['layout']
OUT=ROOT/'output/pdf/p4_scene_dimensions_and_recorder_report_20260916.pdf'
OUT.parent.mkdir(parents=True,exist_ok=True)
pdfmetrics.registerFont(TTFont('Arial',r'C:\Windows\Fonts\arial.ttf'))
pdfmetrics.registerFont(TTFont('ArialB',r'C:\Windows\Fonts\arialbd.ttf'))
W,H=landscape(A4); C=canvas.Canvas(str(OUT),pagesize=(W,H))
C.setTitle('P4 | Scene Dimensions, Cameras and Recorder Control'); C.setAuthor('P4 workspace audit')
INK='#13253A'; MUTED='#546579'; TEAL='#008A81'; BLUE='#2265C9'; GREEN='#D8F2E3'; BORDER='#D6DFE9'
names=list(D['instruments']); palette=['#A46A00','#CC315A','#00866B','#237BC0','#8254B6']
page=0
def text(x,y,s,size=10,color=INK,bold=False):
    C.setFillColor(HexColor(color)); C.setFont('ArialB' if bold else 'Arial',size); C.drawString(x,y,str(s))
def para(x,y,s,width=760,size=10,color=INK):
    p=Paragraph(s,ParagraphStyle('p',fontName='Arial',fontSize=size,leading=size*1.38,textColor=HexColor(color)))
    _,h=p.wrap(width,500); p.drawOn(C,x,y-h); return y-h
def line(a,b,color=TEAL,dash=False,width=1):
    C.saveState(); C.setStrokeColor(HexColor(color)); C.setLineWidth(width)
    if dash:C.setDash(2,3)
    C.line(*a,*b); C.restoreState()
def dot(p,color=TEAL,r=3):
    C.setFillColor(HexColor(color)); C.circle(*p,r,stroke=0,fill=1)
def poly(pts,stroke=TEAL,fill=None,width=1):
    p=C.beginPath(); p.moveTo(*pts[0])
    for xy in pts[1:]:p.lineTo(*xy)
    p.close(); C.setStrokeColor(HexColor(stroke)); C.setLineWidth(width)
    if fill:C.setFillColor(HexColor(fill))
    C.drawPath(p,stroke=1,fill=int(fill is not None))
def dim(a,b,label,offset=8):
    line(a,b,TEAL,True)
    dx,dy=b[0]-a[0],b[1]-a[1]; n=math.hypot(dx,dy); ux,uy=dx/n,dy/n
    for end,sign in ((a,1),(b,-1)):
        poly([end,(end[0]+sign*ux*5-uy*2,end[1]+sign*uy*5+ux*2),(end[0]+sign*ux*5+uy*2,end[1]+sign*uy*5-ux*2)],fill=TEAL)
    mx,my=(a[0]+b[0])/2,(a[1]+b[1])/2
    C.saveState(); C.translate(mx,my+offset); C.rotate(math.degrees(math.atan2(dy,dx)) if abs(dy)>abs(dx) else 0)
    C.setFont('ArialB',9); C.setFillColor(HexColor(TEAL)); C.drawCentredString(0,0,label); C.restoreState()
def table(rows,widths,x,y,size=9,padding=7):
    sty=ParagraphStyle('cell',fontName='Arial',fontSize=size,leading=size*1.28,textColor=HexColor(INK))
    data=[[Paragraph(str(v),sty) for v in row] for row in rows]
    t=Table(data,colWidths=widths,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),HexColor('#E2EBF4')),('VALIGN',(0,0),(-1,-1),'TOP'),
      ('ROWBACKGROUNDS',(0,1),(-1,-1),[white,HexColor('#F4F7FA')]),('LINEBELOW',(0,0),(-1,0),1,HexColor(BORDER)),
      ('BOTTOMPADDING',(0,0),(-1,-1),padding),('TOPPADDING',(0,0),(-1,-1),padding)]))
    _,h=t.wrap(sum(widths),H)
    assert y-h>=45,(page,'table overflow',y-h)
    t.drawOn(C,x,y-h); return y-h
def start(title,sub):
    global page
    if page:C.showPage()
    page+=1
    C.setFillColor(HexColor(INK)); C.rect(0,H-13,W,13,fill=1,stroke=0)
    text(38,H-39,'P4 / TECHNICAL WORKSPACE REPORT',9,TEAL,True)
    text(38,H-70,title,23,INK,True); text(38,H-90,sub,9,MUTED)
    line((38,33),(W-38,33),BORDER)
    text(38,20,'16 Sep 2026  |  Local USD + configuration + validated H5  |  SI units unless noted',8,MUTED)
    text(W-82,20,f'{page:02d} / 10',8,MUTED)
def map_fn(ox=85,oy=462,scale=300,xmin=-.3,ymin=-1.1):
    return lambda x,y:(ox+(y-ymin)*scale,oy-(x-xmin)*scale)
def workspace(f,show_ids=True):
    poly([f(-.3,-1.1),f(-.3,1.1),f(.6,1.1),f(.6,-1.1)],fill='#F0F4F8',stroke='#7B8998')
    gx,gy=L['grid_x'],L['grid_y']; poly([f(gx[0],gy[0]),f(gx[0],gy[1]),f(gx[1],gy[1]),f(gx[1],gy[0])],fill=GREEN)
    for i in range(3):line(f(gx[0]+i*.2,gy[0]),f(gx[0]+i*.2,gy[1]),'#8BBAA8')
    for i in range(6):line(f(gx[0],gy[0]+i*.2),f(gx[1],gy[0]+i*.2),'#8BBAA8')
    if show_ids:
        for i in range(10):
            r,c=divmod(i,2);p=f(.22+c*.2,-.5924+r*.2);text(p[0]-3,p[1]-3,str(i),9,MUTED)
    tx,ty=L['tray_xy'];poly([f(tx-.275,ty-.11),f(tx-.275,ty+.11),f(tx+.275,ty+.11),f(tx+.275,ty-.11)],fill='#DDE3EA',stroke='#687583')
    for slot in D['tray']['slots']:
        p=f(*slot['center_xy']);text(p[0]-2,p[1]-3,str(slot['slot_id']),8,MUTED)
    p=f(*L['robot_pos'][:2]);dot(p,INK,11);text(p[0]+15,p[1]-3,'Robot',10,INK,True)
def silhouette(item,pose):
    a=math.radians(pose[2]);c,s=math.cos(a),math.sin(a)
    return [(pose[0]+c*x-s*y,pose[1]+s*x+c*y) for x,y in item['hull_xy_m']]
def fmt(v,n=3):return '['+', '.join(f'{x:.{n}f}' for x in v)+']'

start('Scene layout yang dipakai sekarang','Satu meja native rumah sakit; diagram mengikuti orientasi top-view control panel.')
f=map_fn();workspace(f)
dim((85,483),(745,483),'2.200 m - panjang meja / world Y')
dim((772,192),(772,462),'0.900 m - lebar / world X',10)
text(100,435,'TRAY',10,MUTED,True);text(440,291,'GRID 10 SEL',12,TEAL,True)
line(f(*L['robot_pos'][:2]),f(.32,-.1924),BLUE,True)
text(380,346,'320 mm',9,BLUE,True)
text(80,170,'Arah gambar: kanan = +Y; bawah = +X; +Z keluar dari permukaan meja.',10,MUTED)
table([['Meja aktif','Grid spawn','Tray','Robot base'],['Table_04<br/>2.200 x 0.900 m','1.000 x 0.400 m<br/>10 sel, 0.200 x 0.200 m','0.550 x 0.220 m<br/>5 slot kelas tetap',fmt(L['robot_pos'])+' m']], [188]*4,44,148,10)
para(44,75,'<b>Catatan:</b> Z=0 adalah permukaan meja, bukan lantai. Meja dan scene lain tidak dipindahkan oleh perubahan panel ini.',750,9)

start('Dimensi, ketinggian dan slot tray','Dimensi berasal dari geometri aset yang telah diskalakan; root transform bukan permukaan kontak.')
# elevation
text(45,477,'ELEVASI SAMPING (skematis)',11,INK,True)
poly([(58,434),(358,434),(358,418),(58,418)],fill='#DDE3EA')
poly([(78,418),(90,418),(90,220),(78,220)],fill='#B5C4D3')
poly([(326,418),(338,418),(338,220),(326,220)],fill='#B5C4D3')
line((45,220),(379,220),MUTED,True);text(65,441,'Z = 0.000 m',10,TEAL)
dim((385,220),(385,434),'0.9045 m - tinggi assembly',10)
text(59,194,'Top slab: 49.0 mm; assembly hingga bagian terbawah.',9,MUTED)
text(450,477,'TRAY - TOP VIEW LOKAL',11,INK,True)
poly([(448,275),(792,275),(792,425),(448,425)],fill='#EDF1F5')
for i,n in enumerate(names):
    xx=482+i*68; line((xx,290),(xx,407),palette[i],False,2);dot((xx,350),palette[i]);text(xx-5,260,str(i),10,palette[i],True)
line((448,350),(792,350),TEAL,True)
dim((448,444),(792,444),'550 mm')
dim((810,275),(810,425),'220 mm',7)
text(461,236,'Pitch slot 104 mm; area tiap slot 200 x 104 mm.',9,TEAL)
table([['Elemen','Position / dimension (meter)','Makna'],['Meja center XY',fmt(L['table_center_xy']),'World X [-0.30, 0.60]; Y [-1.10, 1.10]'],
 ['Tray center XY',fmt(L['tray_xy']),'Root Z=0.008; yaw=90 deg; tinggi aset=10.43 mm'],
 ['Tray support / lip','0.003194 / 0.013214','Hasil raycast scene pada run Love tervalidasi'],
 ['Release objek','Target gap 8 mm; diterima 3-18 mm','Di atas lantai tray; bukan EE menyentuh tray']], [150,240,364],44,180,9)

start('Instrumen: ukuran dan massa simulasi','L/W/T di sini adalah tiga rentang bounding box link, diurutkan; T bukan ketebalan lembar logam.')
for i,n in enumerate(names):
    item=D['instruments'][n];cx=113+i*154;cy=388
    poly([(cx+y*820,cy-x*820) for x,y in item['hull_xy_m']],palette[i],None,1.4)
    text(cx-64,467,n,10,palette[i],True)
    text(cx-60,292,f"yaw 0 axis: {item['heading_offset_deg']:.1f} deg",9,MUTED)
rows=[['Instrumen','L x W x T envelope (mm)','Tinggi settle-ref (mm)','Mass target / distractor','Asset scale']]
for n in names:
    d=D['instruments'][n];rows.append([n,' x '.join(f'{v*1000:.2f}' for v in d['length_width_thickness_m']),f"{d['footprint_yaw0_xyz_m'][2]*1000:.2f}",f"{d['target_mass_kg']*1000:.0f} / {d['distractor_mass_kg']*1000:.0f} g",f"{d['scale'][0]:.8g}"])
bottom=table(rows,[130,201,132,167,124],44,266,8.5,padding=5)
para(44,bottom-12,'Massa tersebut adalah parameter rigid-body simulasi, bukan hasil menimbang alat fisik. Weight force pada g=9.81 m/s2: 100 g = 0.981 N; 120 g = 1.177 N. Angka scale tidak bersatuan dan bukan ukuran meter.',752,9)
para(44,87,'Outline = convex hull luar, bukan gambar lubang/rahang lengkap. Sebelum Prepare: prediksi dari pose referensi settle. Setelah Prepare: outline diukur dari scene. Panah adalah sumbu link positif, bukan label anatomis ujung instrumen.',752,9)

start('Posisi dan jarak antar objek','Contoh layout default panel: semua instrumen di meja, yaw=0. Random spawn menghasilkan jarak berbeda.')
f=map_fn(75,470,270);workspace(f,False)
positions={n:(.42,-.5924+i*.2) for i,n in enumerate(names)}
for i,n in enumerate(names):
    poly([f(x,y) for x,y in silhouette(D['instruments'][n],(*positions[n],0))],palette[i],None,1.2)
    p=f(*positions[n]);dot(p,palette[i]);text(p[0]-3,p[1]+15,str(i),10,palette[i],True)
    if i<4:
        a=f(.56,positions[n][1]);b=f(.56,positions[names[i+1]][1]);dim(a,b,'200 mm')
dim(f(.12,-.78),f(.12,-.6924),'87.6 mm',12)
text(80,216,'Jarak center-to-center default (m). ID 0..4 = scalpel, scissor, Love, Kelly, type2.',10,MUTED)
rows=[['ID','Center X,Y','Ke robot XY','Ke tray center','Ke ID 0','1','2','3','4']]
for i,n in enumerate(names):
    xy=positions[n];rows.append([str(i),fmt(xy),f'{math.dist(xy,L["robot_pos"][:2]):.3f}',f'{math.dist(xy,L["tray_xy"]):.3f}']+[f'{math.dist(xy,positions[m]):.3f}' for m in names])
table(rows,[35,130,110,115,72,72,72,72,76],44,201,9)
para(695,450,'Robot-grid center = 320 mm.<br/><br/>Robot-tray center = '+f'{math.dist(L["robot_pos"][:2],L["tray_xy"])*1000:.1f}'+' mm.<br/><br/>Gap 87.6 mm: jarak tepi grid ke tray pada arah Y.',102,9)

start('Camera placement: world positions','Pose tidak diubah. Kamera wrist bergerak bersama robot; posisinya di file layout adalah lokal pada mount.')
f=map_fn(60,464,155,xmin=-.45,ymin=-1.4);workspace(f,False)
for i,(n,d) in enumerate(D['cameras'].items()):
    if n=='grip_cam_b':continue
    x,y,z=d['pos'];p=f(x,y);dot(p,BLUE,4)
    # ROS optical forward is +Z; all stored quaternions use w,x,y,z.
    q=np.array(d['rot']);q/=np.linalg.norm(q);v=np.array([0.,0.,1.]);t=2*np.cross(q[1:],v);v=v+q[0]*t+np.cross(q[1:],t)
    if v[2]<-.01:
        hit=np.array(d['pos'])-z/v[2]*v
        line(p,f(*hit[:2]),BLUE,True);dot(f(*hit[:2]),BLUE,2)
    text(p[0]+6,p[1]+8,n,9,BLUE,True)
text(59,80,'Titik biru: posisi XY kamera. Garis putus-putus: optical axis ke Z=0, bukan batas field-of-view.',9,MUTED)
rows=[['Camera','X / Y / Z (m)','Frame']]
for n,d in D['cameras'].items():rows.append([n,fmt(d['pos'],4),'hand mount local' if n=='grip_cam_b' else 'world / env_0'])
table(rows,[103,159,99],444,478,8.8)
para(451,179,'cam_tray tetap memakai pose yang tersimpan sekarang, bukan kamera overhead ideal yang diasumsikan. Lokasi dan arah yang dilaporkan mengikuti camera_layout.json.',342,9)

start('Camera calibration dan crop 224','Penting: 224 x 224 adalah hasil center crop recorder, bukan native sensor 224 x 224.')
rows=[['Camera','Quaternion w,x,y,z (saved layout)','Frame']]
for n,d in D['cameras'].items():rows.append([n,fmt(d['rot'],5),'ROS; mount local' if n=='grip_cam_b' else 'ROS; world'])
y=table(rows,[130,470,154],44,476,9)
text(49,y-28,'448 x 336 -> CENTER CROP -> 224 x 224',13,TEAL,True)
# crop illustration
C.setStrokeColor(HexColor(MUTED));C.rect(65,86,224,168,stroke=1,fill=0)
C.setFillColor(HexColor(GREEN));C.rect(121,114,112,112,stroke=1,fill=1)
text(68,260,'Sensor raw (diagram skala 1:2)',9,MUTED)
text(128,167,'224 x 224',11,TEAL,True)
text(64,65,'Kiri/kanan: 112 px; atas/bawah: 56 px.',9,MUTED)
table([['Calibration output','Fixed views','Wrist'],['fx = fy (pixel)','342.0663','299.3080'],['cx, cy (pixel)','112, 112','112, 112'],['Matrix shape','(T,3,3)','(T,3,3)'],['Pose stored in H5','position_b (T,3)','quaternion_b_ros (T,4)']], [172,141,155],330,257,9)
para(330,82,'Crop menggeser principal point; focal length pixel tidak di-rescale. Extrinsics H5 dinyatakan relatif base robot dan direkam per frame; wrist berubah selama gerakan.',468,9)

start('Gerakan tercatat: jarak dan timing aktual','Contoh Love episode_000000 dari panel_retry_validation_v3; bukan budget maksimum dan bukan target untuk semua objek.')
f=map_fn(62,470,228,xmin=-.3,ymin=-1.1);workspace(f,False)
xyz=D['trajectory_xyz']
for a,b in zip(xyz,xyz[1:]):line(f(*a[:2]),f(*b[:2]),BLUE,True,1.6)
for idx,offset in ((0,(20,26)),(2,(-32,-15)),(4,(25,-18)),(5,(30,-12)),(8,(20,26))):
    s=D['stages'][idx];p=f(*s['end_xyz'][:2]);dot(p,BLUE)
    q=(p[0]+offset[0],p[1]+offset[1]);line(p,q,BLUE,True,.7);text(q[0],q[1],str(idx),9,BLUE,True)
text(62,238,'XY path terukur; angka menunjuk stage di tabel.',9,MUTED)
# elevation path with x as time frame to avoid overlapping z motions
text(580,468,'Z terhadap urutan sampel',10,INK,True)
for i,(a,b) in enumerate(zip(xyz,xyz[1:])):
    line((585+i/len(xyz)*180,286+a[2]*360),(585+(i+1)/len(xyz)*180,286+b[2]*360),TEAL,True)
line((580,286),(785,286),MUTED);text(580,266,'Z=0 meja; rentang grafik ~0.4 m.',8,MUTED)
rows=[['# / stage','Steps','Sim (s)','Chord (mm)','Path (mm)']]
for i,s in enumerate(D['stages']):rows.append([f"{i} / {s['name'].replace('LOVE_','')}",str(s['steps']),f"{s['sim_seconds']:.2f}",f"{s['chord_m']*1000:.2f}",f"{s['path_m']*1000:.2f}"])
table(rows,[262,92,120,140,140],44,226,7.9,padding=3)
para(44,61,'Chord = garis lurus dari sampel pertama ke terakhir dalam stage; path = jumlah perpindahan EE antarsampel. 1 step = 0.02 s simulasi. Wall time lebih lama karena render, observasi dan penyimpanan.',752,8)

start('Isi dataset: schema dan label','76 dataset topics pada contoh H5; alias kompatibilitas tidak berarti ada kamera fisik tambahan.')
table([['Data','Shape','Dtype / frame'],['RGB: front, wrist, top, left, right, tray','(T,224,224,3)','uint8; 6 stream'],['Depth','(T,224,224)','float16; meter; tetap direkam'],['Semantic','(T,224,224)','uint16; 6 stream'],['robot_proprio','(T,16)','float32: 7 joints, 2 fingers, EE xyz + quat'],['state','(T,18)','proprio + object_type_id + skill_id'],['actions','(T,8)','float32: EE xyz + quat wxyz + grip'],['stage_id / step_ids / dones','(T,1) / (T) / (T)','int32 / int32 / bool'],['Calibration','K (T,3,3), pose (T,3)+(T,4)','raw + cropped K; base frame'],['debug_gt / supervision_gt','poses, masks, teacher grasp','Simulator labels; not default policy observations']], [245,238,271],44,474,9)
rows=[['Class','Semantic ID','Object / tray slot ID']]
sem=json.loads(D['sample_attrs']['semantic_class_ids']);oid=json.loads(D['sample_attrs']['object_type_ids'])
for n in sem:rows.append([n,str(sem[n]),str(oid[n]) if n in oid else '-'])
table(rows,[140,92,140],44,204,8,padding=2)
para(441,196,'<b>Pick:</b> OPEN_HOVER, LOWER_PRE, LOWER_GRASP, CLOSE, LIFT_CLEAR.<br/><br/><b>Place:</b> MOVE_TO_TARGET, LOWER_PLACE, OPEN, RETREAT.<br/><br/>Contoh: pick T=155, place T=158. LOWER_EXTRA masih terdapat sebagai ID legacy dalam mapping, tetapi tidak muncul pada rekaman contoh ini.',353,10)
para(441,77,'Untuk policy visual tanpa posisi objek simulator: gunakan RGB + proprio + conditioning tugas. Semantic/GT boleh untuk supervision tambahan; jangan diam-diam menjadi input inference.',353,9)

start('Control panel dan definisi indikator','Counter berbasis attempt + hasil save. Tiga lampu merah menandai fase; tidak menyatakan semua run pasti berhasil.')
C.setFillColor(HexColor('#151B24'));C.roundRect(45,281,752,193,8,fill=1,stroke=0)
text(61,449,'P4 RECORDER',14,'#E8EFF7',True)
text(468,449,'Episode target 8/30 | Saved 7/30',12,'#E8EFF7',True)
text(548,427,'Attempt 10 | Fail 2',12,'#E8EFF7',True)
text(555,408,'LOVE_CLOSE',10,'#88DCD1')
poly([(62,391),(418,391),(418,318),(62,318)],'#67AF90','#214A3F')
for i in range(1,5):line((62+i*71.2,318),(62+i*71.2,391),'#67AF90')
line((62,354),(418,354),'#67AF90')
text(447,377,'Log: stage, sim time, fail reason, save',10,'#D8E3EF')
for i,label in enumerate(('RESET / WAIT','RECORDING','END / SAVING')):
    x=484+i*116;C.setFillColor(HexColor('#FF424F' if i==1 else '#49242C'));C.circle(x,321,6,stroke=0,fill=1);text(x-38,302,label,8,'#E8EFF7')
text(48,267,'Skema UI, bukan screenshot simulator. Attempt 10: 7 saved + 2 failed + 1 sedang berjalan.',9,MUTED)
table([['Indikator','Arti yang tepat'],['Episode target','Nomor sukses berikutnya yang sedang diusahakan; bukan jumlah percobaan'],['Saved / Goal','Bertambah hanya setelah [SAVE SPLIT], bukan setelah [RESULT] success=True'],['Attempt / Fail','Attempt termasuk percobaan berjalan; fail tidak dihitung dua kali bila log berulang'],['Reset / Wait','Startup, spawn, settle, retry atau menunggu Start'],['Recording','Stage gerakan aktif; nama stage tampil di kanan atas'],['End / Saving','QC / kompresi / save atau proses berakhir. Lihat teks Complete / Incomplete.']], [185,569],44,245,9)
para(44,65,'Single: jumlah saved success. Grid cycles: 3 putaran x 10 sel = 30 saved success; gagal mengulang sel yang sama. Tidak ada batas attempts pada panel, tetapi Stop dan fatal-error reporting tetap berlaku.',752,9)

start('Batas validasi dan sumber pengukuran','Snapshot laporan: 16 September 2026. P3 tidak diubah; laporan ini menggambarkan P4 saat audit.')
y=table([['Bagian','Sumber / metode','Batas interpretasi'],['Table / grid / tray','scene_layout.json; phase4_scene.py; hospital_isaac.usdz / Table_04; SurgicalTray.usd','Dimensi render/collider aset simulasi; bukan pengukuran rumah sakit nyata'],['Instrument size / outline','Mesh vertices di link frame x configured scale; settled target H5 masing-masing kelas','Envelope luar; prediksi yaw sebelum Prepare tidak identik dengan seluruh hasil dinamika'],['Mass','phase3_shared_env_cfg.py: rigid_cfg canonical target versus distractor spec','Configured kg; target dan distractor bisa berbeda. Bukan medical specification'],['Camera poses / K','camera_layout.json + camera_calibration dalam H5 Love','Static cameras world, wrist local; H5 extrinsics base frame'],['Motion lengths','robot_proprio EE; base-to-world transform; semua sampel tiap stage','Satu demonstrasi Love; bukan jangkauan aman universal atau panjang instrumen'],['Run validation','panel_retry_validation_v3/run_metrics.json','2 attempts, 2 saved successes, 0 fails; satu custom pose Love']], [148,350,256],44,474,9)-16
y=para(44,y,'<b>Yang belum dibuktikan:</b> semua instrumen pada semua sel dan semua yaw belum diuji secara fisik penuh. Cakupan grid yang lengkap juga tidak membuktikan dataset cukup untuk model generalisasi. Headless pernah menghasilkan RGB flat; GUI dan pemeriksaan seluruh frame tetap diperlukan.',750,11)
y=para(44,y-17,'<b>Data referensi motion:</b> test_runs/panel_retry_validation_v3/pick_policy/love_retractor/episode_000000.h5 dan pasangan place-nya. Referensi outline kelas lain berasal dari tray_slots_scalpel_retest_v2, tray_slots_full_validation (scissor, Kelly), dan tray_slots_type2_retest.',750,9)
para(44,y-15,'<b>Reproduksi:</b> reporting/report_data.json menyimpan angka hasil ekstraksi; instrument_preview_geometry.json menyimpan source H5 dan SHA-256 asset per kelas. Ekstraksi membaca data/aset, tidak mengubah scene, massa, kamera, atau gerakan recorder.',750,9)

C.save()
# Installed PyMuPDF provides deterministic page rendering where Poppler isn't available.
import fitz
qa=ROOT/'tmp/pdfs/p4_scene_report_20260916';qa.mkdir(parents=True,exist_ok=True)
doc=fitz.open(OUT)
assert len(doc)==10
for i,p in enumerate(doc):
    p.get_pixmap(matrix=fitz.Matrix(1.35,1.35),alpha=False).save(str(qa/f'page_{i+1:02d}.png'))
    assert p.get_text().strip(),i
print(OUT);print('Rendered',len(doc),'pages to',qa)
