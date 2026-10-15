"""Normalize machine-specific paths in baseline output while preserving meaning.

The static runner stores tool commands, stdout, stderr, and source frames for
review. This helper replaces workstation-specific roots with portable markers
before those records enter the repository. It does not alter scan behavior.
"""
from pathlib import Path
import re
ROOT = Path(__file__).resolve().parents[2]

def text(value):
    value = str(value)
    # Longest prefixes first; these are computed, never machine-specific literals.
    for prefix,replacement in [(str(ROOT) + '/', ''), (str(ROOT), '.'),
                               (str(ROOT.parent / '.venv') + '/', '<tools-env>/'),
                               (str(Path.home()) + '/', '<home>/')]:
        value = value.replace(prefix, replacement)
    # Preserve internet URLs and compiler operator names; normalize OS path roots.
    pattern = r'(?<![\w:])/(home|tmp|usr|opt|var|root|build|workspace|workspaces|mnt|etc|data|proc|sys|dev)/([^\s\"\'<>\\,;\)\]]*)'
    value = re.sub(pattern, lambda m: '<' + m[1] + '>/' + m[2], value)
    value = value.replace('file://.', '.').replace('file://<', '<')
    return value

def data(value):
    if isinstance(value,str):return text(value)
    if isinstance(value,dict):return {text(k):data(v) for k,v in value.items()}
    if isinstance(value,list):return [data(v) for v in value]
    if isinstance(value,tuple):return tuple(data(v) for v in value)
    return value

def project_frames(frames):
    """A readable subset; unknown native symbols are not invented source locations."""
    return [data(f) for f in frames if str(f.get('filename','')).startswith(str(ROOT) + '/')
            or str(f.get('filename','')).startswith(('cases/','scripts/'))]
