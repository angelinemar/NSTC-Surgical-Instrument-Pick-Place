"""Measured 2D instrument sheets from active USD meshes and recorded RGB."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pxr import Gf, Usd, UsdGeom, UsdPhysics

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'compat'))
from phase3_shared_env_cfg import INSTRUMENTS

OUT = ROOT / 'output/instrument_dimension_sheets'
NAMES = dict(scalpel='SCALPEL', scissor='SCISSOR', love_retractor='LOVE RETRACTOR',
             kelly='KELLY', scalpel_type2='SCALPEL TYPE 2')
INK, ACCENT, GUIDE = '#183b44', '#087f8c', '#819da4'
BG = '#f5f5ef'


def font(size):
    return ImageFont.truetype('C:/Windows/Fonts/bahnschrift.ttf', size)


def mesh_for(name):
    spec = INSTRUMENTS[name]
    path = ROOT / 'assets' / spec.usd
    stage = Usd.Stage.Open(str(path))
    rigid = next(p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI))
    root_matrix = UsdGeom.Xformable(rigid).ComputeLocalToWorldTransform(0)
    inverse = root_matrix.GetInverse()
    scale = np.abs(np.asarray(Gf.Transform(root_matrix).GetScale())) * spec.scale
    vertices, faces, offset = [], [], 0
    for prim in Usd.PrimRange(rigid, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh) or UsdGeom.Imageable(prim).ComputeVisibility() == 'invisible':
            continue
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
        matrix = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0) * inverse)
        vertices.append((np.column_stack((points, np.ones(len(points)))) @ matrix)[:, :3] * scale)
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
        start = 0
        for count in mesh.GetFaceVertexCountsAttr().Get():
            polygon = indices[start:start+count] + offset
            faces.extend((polygon[0], polygon[j], polygon[j+1]) for j in range(1, count-1))
            start += count
        offset += len(points)
    v, f = np.concatenate(vertices), np.asarray(faces, dtype=int)
    tri = v[f]
    areas = np.linalg.norm(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=1) / 2
    total = areas.sum()
    sums = tri.sum(axis=1)
    center = (areas[:, None] * sums / 3).sum(axis=0) / total
    # Exact surface-area moments avoid dependence on mesh vertex density.
    second = (np.einsum('n,nki,nkj->ij', areas, tri, tri)
              + np.einsum('n,ni,nj->ij', areas, sums, sums)) / (12*total)
    values, vectors = np.linalg.eigh(second-np.outer(center, center))
    basis = vectors[:, np.argsort(values)[::-1]]
    if np.linalg.det(basis) < 0:
        basis[:, 2] *= -1
    aligned = (v-center) @ basis
    low, high = aligned.min(axis=0), aligned.max(axis=0)
    aligned -= (low+high)/2
    aligned *= 1000
    dimensions = np.ptp(aligned, axis=0)
    assert 50 < dimensions[0] < 300 and 0 < dimensions[1] < 100 and 0 < dimensions[2] < 100
    metadata = dict(asset=str(path), asset_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        configured_scale=list(spec.scale), effective_rigid_scale=scale.tolist(),
        principal_axes_columns=basis.tolist(), length_mm=float(dimensions[0]),
        width_mm=float(dimensions[1]), height_mm=float(dimensions[2]),
        method='Visible mesh envelope along surface-area principal axes; active simulation scale; not manufacturer dimensions')
    return aligned, f, metadata


def mesh_capture(vertices, faces, view, size=(1500, 600), fixed_scale=None):
    # Orthographic projection with geometric face shading; no invented textures.
    axes = [0, 1, 2] if view == 'top' else [0, 2, 1]
    p = vertices[:, axes]
    triangles = p[faces]
    normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    normals *= np.where(normals[:, 2:3] < 0, -1, 1)
    light = np.array([-0.25, 0.45, 0.86]); light /= np.linalg.norm(light)
    diffuse = np.maximum(normals @ light, 0)
    shade = np.clip(0.30 + 0.60*diffuse + 0.20*diffuse**12, 0, 1)
    colors = np.clip(shade[:, None] * np.array([200, 220, 226]), 0, 255).astype(np.uint8)
    image = Image.new('RGBA', size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    scale = fixed_scale if fixed_scale is not None else min((size[0]-30)/np.ptp(p[:, 0]), (size[1]-30)/max(np.ptp(p[:, 1]), 1e-6))
    xy = np.column_stack((size[0]/2+p[:, 0]*scale, size[1]/2-p[:, 1]*scale))
    for index in np.argsort(triangles[:, :, 2].mean(axis=1)):
        draw.polygon([tuple(point) for point in xy[faces[index]]], fill=(*map(int, colors[index]), 255))
    return image, scale


def recorded_crop(name, reference):
    cid = list(NAMES).index(name)+3
    best = None
    path = ROOT / reference
    with h5py.File(path, 'r') as h:
        stages = h['stage_names'].asstr()[:]
        candidates = np.flatnonzero(np.char.endswith(stages.astype(str), 'OPEN_HOVER'))
        for camera in ('front', 'wrist', 'cam_top', 'cam_left', 'cam_right', 'cam_tray'):
            semkey = 'grip_b' if camera == 'wrist' else camera
            for index in candidates[::max(1, len(candidates)//10)]:
                sem = h[f'observations/{semkey}_semantic'][index]
                yy, xx = np.where(sem == cid)
                if len(xx) < 10 or xx.min() < 4 or yy.min() < 4 or xx.max() > 219 or yy.max() > 219:
                    continue
                pad = 8
                box = (max(0, int(xx.min())-pad), max(0, int(yy.min())-pad),
                       min(224, int(xx.max())+pad+1), min(224, int(yy.max())+pad+1))
                region = sem[box[1]:box[3], box[0]:box[2]]
                score = len(xx) / (1+3*np.count_nonzero(region == 1)/region.size)
                if best is None or score > best[0]:
                    rgb = h[f'observations/{camera}_rgb'][index]
                    best = (score, Image.fromarray(rgb).crop(box), dict(h5=str(path.resolve()), camera=camera,
                             frame=int(index), crop_xyxy=list(box), native_crop_pixels=[box[2]-box[0], box[3]-box[1]]))
    if best is None:
        raise ValueError('No untruncated recorded crop for '+name)
    return best[1:]


def dotted(draw, start, end):
    a, b = np.asarray(start, float), np.asarray(end, float)
    length = np.linalg.norm(b-a)
    for t in np.arange(0, length, 14):
        p, q = a+(b-a)*t/length, a+(b-a)*min(t+4, length)/length
        draw.line((*p, *q), fill=GUIDE, width=3)


def arrow(draw, start, end):
    a, b = np.asarray(start, float), np.asarray(end, float)
    direction = (b-a)/np.linalg.norm(b-a)
    perp = np.array([-direction[1], direction[0]])
    draw.line((*a, *b), fill=ACCENT, width=4)
    outside = np.linalg.norm(b-a) < 36
    for tip, inward in ((a, direction), (b, -direction)):
        if outside:
            inward = -inward
            draw.line((*tip, *(tip+inward*30)), fill=ACCENT, width=4)
        draw.polygon([tuple(tip), tuple(tip+inward*13+perp*6), tuple(tip+inward*13-perp*6)], fill=ACCENT)


def label(draw, position, text, size=30, anchor='mm'):
    box = draw.textbbox(position, text, font=font(size), anchor=anchor)
    draw.rounded_rectangle((box[0]-12, box[1]-8, box[2]+12, box[3]+8), radius=8, fill='white')
    draw.text(position, text, font=font(size), fill=INK, anchor=anchor)


def sheet(name, number, vertices, faces, metadata, crop, capture_info):
    image = Image.new('RGB', (2400, 1350), BG)
    draw = ImageDraw.Draw(image)
    draw.text((80, 45), f'INSTRUMENT {number+1:02d} / 2D DIMENSIONAL REFERENCE', font=font(25), fill=ACCENT)
    draw.text((80, 92), NAMES[name], font=font(68), fill=INK)
    draw.text((80, 181), 'Active simulation model  |  Dimensions in millimetres', font=font(27), fill='#536d73')
    draw.rounded_rectangle((65, 257, 666, 1200), radius=24, fill='white')
    draw.rounded_rectangle((716, 257, 2335, 780), radius=24, fill='white')
    draw.rounded_rectangle((716, 815, 2335, 1200), radius=24, fill='white')
    draw.text((100, 292), 'RECORDED RGB CAPTURE', font=font(28), fill=INK)
    draw.text((100, 338), capture_info['camera'].upper()+' / CROPPED FOR VISIBILITY', font=font(19), fill='#536d73')
    # Contain the actual capture, preserving its aspect ratio.
    display = crop.copy()
    factor = min(490/crop.width, 440/crop.height)
    display = display.resize((round(crop.width*factor), round(crop.height*factor)), Image.Resampling.LANCZOS)
    image.paste(display, (365-display.width//2, 605-display.height//2))
    draw.text((100, 866), 'MEASURED MODEL ENVELOPE', font=font(24), fill=INK)
    for row, (symbol, key, desc) in enumerate((('L', 'length_mm', 'Length'), ('W', 'width_mm', 'Width'), ('H', 'height_mm', 'Height / thickness'))):
        y = 930+row*76
        draw.ellipse((100, y-20, 143, y+23), fill=ACCENT)
        draw.text((121, y+1), symbol, font=font(24), anchor='mm', fill='white')
        draw.text((160, y), desc, font=font(25), anchor='lm', fill=INK)
        draw.text((624, y), f'{metadata[key]:.1f} mm', font=font(33), anchor='rm', fill=INK)
    length, width, height = (metadata[k] for k in ('length_mm', 'width_mm', 'height_mm'))
    center_x = 1395
    shared_scale = min(1150/length, 260/width, 180/height)
    for view, ycenter, title_y, extent in (('top', 550, 292, width), ('side', 1010, 850, height)):
        draw.text((754, title_y), 'TOP VIEW / L x W' if view == 'top' else 'SIDE VIEW / L x H', font=font(27), fill=INK)
        render, scale = mesh_capture(vertices, faces, view, (1180, 290 if view == 'top' else 210), shared_scale)
        image.paste(render, (center_x-render.width//2, ycenter-render.height//2), render)
        x0, x1 = center_x-length*scale/2, center_x+length*scale/2
        y0, y1 = ycenter-extent*scale/2, ycenter+extent*scale/2
        if view == 'top':
            dim_y = 397
            for x in (x0, x1):
                dotted(draw, (x, y0-6), (x, dim_y-15))
                draw.ellipse((x-5, y0-5, x+5, y0+5), fill=ACCENT)
            arrow(draw, (x0, dim_y), (x1, dim_y))
            label(draw, (center_x, dim_y), f'L  {length:.1f} mm')
        dim_x = 2050
        for y in (y0, y1):
            dotted(draw, (x1+8, y), (dim_x+18, y))
            draw.ellipse((x1-5, y-5, x1+5, y+5), fill=ACCENT)
        arrow(draw, (dim_x, y0), (dim_x, y1))
        label(draw, (2090, ycenter), f'{"W" if view == "top" else "H"}  {extent:.1f} mm', anchor='lm')
    draw.text((82, 1240), 'H = overall profile height, not local metal thickness. Dotted lines mark the measured extents.', font=font(28), fill=INK)
    draw.text((82, 1290), 'Mesh-based, principal-axis bounds at simulation scale. Model views are orthographic; recorded RGB is enlarged.', font=font(23), fill='#536d73')
    image.save(OUT/f'{number+1:02d}_{name}_dimensions.png', dpi=(200, 200))
    return image


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    references = json.loads((ROOT/'env/instrument_preview_geometry.json').read_text())['instruments']
    reports, sheets = {}, []
    for number, name in enumerate(NAMES):
        vertices, faces, metadata = mesh_for(name)
        crop, captured = recorded_crop(name, references[name]['reference_h5'])
        crop.save(OUT/f'{name}_recorded_rgb_crop.png')
        for view in ('top', 'side'):
            mesh_capture(vertices, faces, view, (2000, 800))[0].save(OUT/f'{name}_{view}_model.png', dpi=(200, 200))
        sheets.append(sheet(name, number, vertices, faces, metadata, crop, captured))
        reports[name] = dict(**metadata, recorded_capture=captured)
        print(name, 'L/W/H mm:', [round(metadata[k], 2) for k in ('length_mm', 'width_mm', 'height_mm')], flush=True)
    overview = Image.new('RGB', (2400, 3*675), BG)
    for i, image in enumerate(sheets):
        overview.paste(image.resize((1200, 675), Image.Resampling.LANCZOS), ((i%2)*1200, (i//2)*675))
    d = ImageDraw.Draw(overview)
    d.text((1280, 1430), 'HOW TO READ THE SHEETS', font=font(42), fill=INK)
    for row, line in enumerate(('L = length along the main axis', 'W = width in the top view', 'H = overall height in the side view', 'All measurements: millimetres', 'Model names follow the project registry.')):
        d.text((1280, 1520+row*65), line, font=font(30), fill=INK)
    overview.save(OUT/'all_instruments_overview.png', dpi=(200, 200))
    (OUT/'measurements.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')
    (OUT/'README.txt').write_text('Five instrument sheets for PowerPoint. English labels; 2400 x 1350 PNG.\n'
        'Each sheet includes an actual recorded RGB crop and two orthographic views of the active USD mesh.\n'
        'Model views use neutral geometric shading, not the original renderer material or an AI image.\n'
        'L/W/H are bounding extents along surface-area principal axes at the active simulation scale.\n'
        'H means overall profile height; it does not measure local wall/blade metal thickness.\n'
        'Numbers describe the project assets, not a manufacturer specification.\n'
        'Recorded RGB crops retain their real pixel detail; enlarging cannot restore missing detail.\n'
        'Asset hashes, scales, axes and RGB source frames are recorded in measurements.json.\n', encoding='utf-8')
    with zipfile.ZipFile(OUT/'instrument_dimension_sheets.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(OUT.iterdir()):
            if path.suffix in ('.png', '.json', '.txt'):
                archive.write(path, path.name)
    print('Saved:', OUT.resolve(), flush=True)


if __name__ == '__main__':
    main()
