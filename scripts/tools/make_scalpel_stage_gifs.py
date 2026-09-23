"""Export presentation GIFs from recorded front RGB, split by actual stage names."""
import argparse
import json
from pathlib import Path
import zipfile

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont

STAGES = (
    ('OPEN_HOVER', 'Move the open gripper above the scalpel.'),
    ('LOWER_PRE', 'Lower to the pre-grasp position.'),
    ('LOWER_GRASP', 'Lower slowly to the grasping height.'),
    ('CLOSE', 'Close the gripper to grasp the scalpel.'),
    ('LIFT_CLEAR', 'Lift the scalpel clear of the table.'),
    ('MOVE_TO_TARGET', 'Carry the scalpel toward the target tray.'),
    ('LOWER_PLACE', 'Lower the scalpel into its tray slot.'),
    ('OPEN', 'Open the gripper to release the scalpel.'),
    ('RETREAT', 'Move the gripper away after release.'),
)


def font(size):
    for name in ('C:/Windows/Fonts/bahnschrift.ttf', 'C:/Windows/Fonts/segoeui.ttf'):
        if Path(name).exists():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--episode', type=int, default=9)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source_frames = {}
    sources = {}
    control_dt = None
    for phase in ('pick', 'place'):
        path = args.run / (phase + '_policy') / 'scalpel' / f'episode_{args.episode:06d}.h5'
        with h5py.File(path, 'r') as h:
            assert bool(h.attrs['success'])
            dt = float(h.attrs['control_dt_s'])
            assert control_dt is None or control_dt == dt
            control_dt = dt
            names = h['stage_names'].asstr()[:]
            for name in dict.fromkeys(names):
                suffix = name.removeprefix('SCALPEL_')
                indices = np.flatnonzero(names == name)
                assert len(indices) and np.all(np.diff(indices) == 1), name
                source_frames[suffix] = h['observations/front_rgb'][indices]
                sources[suffix] = dict(h5=str(path.resolve()), phase=phase,
                                       frame_start=int(indices[0]), frame_end_inclusive=int(indices[-1]))
    assert set(source_frames) == {name for name, _ in STAGES}
    playback_ms = int(round(control_dt * 4 * 1000 / 10)) * 10
    width, header, footer = 640, 102, 100
    title_font, body_font, small_font = font(32), font(23), font(17)
    manifest = dict(episode=args.episode, camera='front_rgb', native_resolution=[224, 224],
                    display_resolution=[width, width], speed='0.25x simulation speed',
                    source_control_dt_s=control_dt, frame_duration_ms=playback_ms,
                    note='All frames are actual recorded front RGB. No semantic images or generated motion.', stages=[])
    overview = Image.new('RGB', (960, 3*462), '#ecefe9')
    for number, (stage, caption) in enumerate(STAGES):
        rgb_frames = source_frames[stage]
        rendered = []
        for index, rgb in enumerate(rgb_frames):
            canvas = Image.new('RGB', (width, header+width+footer), '#f4f3ec')
            draw = ImageDraw.Draw(canvas)
            draw.rectangle((0, 0, width, header), fill='#153f3b')
            draw.text((22, 12), 'SCALPEL  /  FRONT RGB', font=small_font, fill='#c9ded8')
            draw.text((22, 40), f'{number}  {stage}', font=title_font, fill='white')
            canvas.paste(Image.fromarray(rgb).resize((width, width), Image.Resampling.BICUBIC), (0, header))
            y = header+width
            draw.text((20, y+12), caption, font=body_font, fill='#153f3b')
            note = 'Placement area is outside the front camera crop.' if number >= 6 else 'Playback slowed down 4x for demonstration.'
            draw.text((20, y+47), note, font=small_font, fill='#6a6046')
            draw.rectangle((20, y+83, width-20, y+87), fill='#d5dcd4')
            draw.rectangle((20, y+83, 20+int((width-40)*(index+1)/len(rgb_frames)), y+87), fill='#248578')
            rendered.append(canvas)
        # One shared palette avoids palette-induced color flicker between frames.
        samples = np.linspace(0, len(rendered)-1, min(12, len(rendered)), dtype=int)
        palette_sheet = Image.new('RGB', (width, rendered[0].height*len(samples)))
        for row, i in enumerate(samples):
            palette_sheet.paste(rendered[i], (0, row*rendered[i].height))
        palette = palette_sheet.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
        frames = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in rendered]
        durations = [playback_ms]*len(frames)
        durations[0] += 500
        durations[-1] += 900
        path = args.output / f'{number:02d}_{stage}.gif'
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations,
                       loop=0, disposal=2, optimize=False)
        with Image.open(path) as check:
            assert check.n_frames == len(frames), (stage, check.n_frames, len(frames))
            assert check.size == (width, header+width+footer)
            for i in range(check.n_frames):
                check.seek(i)
                check.load()
        manifest['stages'].append(dict(order=number, stage=stage, description=caption,
            file=path.name, source_frames=len(frames), loop_duration_ms=sum(durations),
            placement_outside_crop=number >= 6, **sources[stage]))
        thumb = rendered[len(rendered)//2].resize((320,421), Image.Resampling.LANCZOS)
        overview.paste(thumb, ((number%3)*320, (number//3)*462))
        print(f'{path.name}: {len(frames)} frames, {sum(durations)/1000:.2f}s per loop', flush=True)
    overview.save(args.output/'stage_overview.png')
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    readme = ['SCALPEL - FRONT RGB - 9 STAGE GIFS', '',
              f'Source: episode {args.episode:06d}, a successful pick/place H5 pair.',
              'GIFs follow the original stage_names boundaries without mixing stages.',
              'Numbers 0-8 indicate presentation order, not internal recorder stage IDs.',
              'Size: 640 x 842; native 224 x 224 RGB enlarged for presentation.',
              'Playback: 0.25x simulation speed; initial hold 0.5 s, final hold 0.9 s.',
              'GIFs loop automatically. All frames are original front camera RGB.', '',
              'CAMERA VIEW LIMITATION',
              'The target tray is outside the front camera crop during placement.',
              'LOWER_PLACE, OPEN, and part of RETREAT do not show the detailed',
              'gripper/object contact with the tray. Captions explain the stage;',
              'motion outside the frame has not been reconstructed.', '',
              'POWERPOINT',
              'Use Insert > Pictures > This Device, then select a GIF file.',
              'Animations play in Slide Show; editing view may display a still frame.', '',
              'STAGE ORDER']
    readme.extend(f'{i}. {name}: {caption}' for i,(name,caption) in enumerate(STAGES))
    (args.output/'README.txt').write_text('\n'.join(readme)+'\n', encoding='utf-8')
    zip_path = args.output/'scalpel_front_9_stages.zip'
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(args.output.glob('*.gif')):
            archive.write(path, path.name)
        for name in ('README.txt', 'stage_overview.png', 'manifest.json'):
            archive.write(args.output/name, name)
    print('Output:', args.output.resolve(), flush=True)


if __name__ == '__main__':
    main()
