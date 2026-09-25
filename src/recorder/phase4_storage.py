"""Journaled segment publication. Only a commit manifest makes an episode complete."""
import json
import hashlib
import os
from pathlib import Path
import uuid


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(payload, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def contained(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Transaction path escapes output root')
    return path


def recover(root):
    """Preserve interrupted writes in quarantine; never treat them as successes."""
    import h5py
    root = Path(root).resolve()
    for journal in sorted((root / '.pending').glob('*/journal.json')):
        data = json.loads(journal.read_text())
        commit = contained(root, data['commit'])
        if commit.exists():
            committed = json.loads(commit.read_text())
            if committed['transaction_id'] == data['transaction_id']:
                continue
            # A fresh run may have committed this index after an earlier staging-only crash.
            quarantine = root / '.quarantine' / journal.parent.name
            quarantine.mkdir(parents=True, exist_ok=True)
            os.rename(journal.parent, quarantine / 'staged')
            continue
        quarantine = root / '.quarantine' / journal.parent.name
        quarantine.mkdir(parents=True, exist_ok=True)
        for relative in data['files']:
            final = contained(root, relative)
            if final.exists():
                with h5py.File(final, 'r') as h:
                    if h.attrs.get('pair_transaction_id') != data['transaction_id']:
                        raise RuntimeError('Refusing to quarantine a file from another transaction')
                destination = quarantine / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.rename(final, destination)
                preview = final.with_name(final.stem + '_preview')
                if preview.exists():
                    os.rename(preview, destination.with_name(preview.name))
        destination = quarantine / 'staged'
        os.rename(journal.parent, destination)
        print('[P4 RECOVERY] quarantined incomplete episode: ' + str(quarantine), flush=True)


def require_commit(path, h5):
    if h5.attrs.get('storage_contract') not in ('journaled_episode_v1', 'journaled_episode_v2'):
        return  # Legacy files remain subject to their existing independent audit.
    root = Path(path).resolve().parents[2]
    manifest = root / '.commits' / (str(h5.attrs['target_object']) + '_' + Path(path).stem + '.json')
    if not manifest.exists():
        raise ValueError('Episode has no commit manifest')
    data = json.loads(manifest.read_text())
    relative = str(Path(path).resolve().relative_to(root))
    if data['transaction_id'] != h5.attrs['pair_transaction_id'] or relative not in data['files']:
        raise ValueError('Commit does not match episode')
    if any(not contained(root, p).is_file() for p in data['files']):
        raise ValueError('Committed episode is missing a segment')
    if h5.attrs['storage_contract'] == 'journaled_episode_v2':
        expected = data['sha256'][relative]
        with Path(path).open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
                raise ValueError('Committed H5 checksum differs')


class EpisodeTransaction:
    def __init__(self, root, object_name, index, mode):
        self.root = Path(root).resolve()
        self.name = f'episode_{index:06d}'
        self.skills = ('pick', 'place') if mode == 'both' else (mode,)
        self.object_name = object_name
        self.token = uuid.uuid4().hex
        self.stage = self.root / '.pending' / self.token
        self.stage.mkdir(parents=True)
        files = [str(Path(s + '_policy') / object_name / (self.name + '.h5')) for s in self.skills]
        self.commit = self.root / '.commits' / (object_name + '_' + self.name + '.json')
        if self.commit.exists() or any(contained(self.root, p).exists() for p in files):
            raise FileExistsError('Refusing to replace existing episode')
        self.journal = dict(transaction_id=self.token, files=files,
                            commit=str(self.commit.relative_to(self.root)), storage_contract='journaled_episode_v2')
        atomic_json(self.stage / 'journal.json', self.journal)
        self.saved = set()

    def save(self, writer, bound):
        skill = bound.arguments['segment']
        if skill not in self.skills:
            return True
        destination = self.stage / (skill + '_policy') / self.object_name
        bound.arguments['out_dir'] = str(destination)
        bound.arguments['meta'].update(storage_contract='journaled_episode_v2', pair_transaction_id=self.token)
        if not writer(*bound.args, **bound.kwargs):
            raise RuntimeError('Segment writer did not complete')
        from src.recorder.instance_labels import append_segment
        append_segment(destination / (self.name + '.h5'), bound.arguments['recorder'], skill)
        from validate_feedback_dataset import audit
        audit(destination / (self.name + '.h5'))
        self.saved.add(skill)
        if self.saved != set(self.skills):
            return True
        self.journal['sha256'] = {}
        for relative in self.journal['files']:
            with (self.stage / relative).open('r+b') as stream:
                os.fsync(stream.fileno())
                self.journal['sha256'][relative] = hashlib.file_digest(stream, 'sha256').hexdigest()
        atomic_json(self.stage / 'journal.json', self.journal)
        for relative in self.journal['files']:
            source = self.stage / relative
            final = contained(self.root, relative)
            final.parent.mkdir(parents=True, exist_ok=True)
            # Windows rename refuses replacement; no preexisting demonstration is overwritten.
            os.rename(source, final)
            preview = source.with_name(self.name + '_preview')
            if preview.exists():
                os.rename(preview, final.with_name(preview.name))
        atomic_json(self.commit, self.journal)
        return True
