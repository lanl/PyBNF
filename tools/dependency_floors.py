#!/usr/bin/env python3
"""Pin PyBNF's runtime dependencies to the lowest versions pyproject.toml allows.

    python tools/dependency_floors.py > floors.txt          # a pip/uv constraints file
    uv pip install --constraint floors.txt ...
    .venv/bin/python tools/dependency_floors.py --check     # did the pins take?

The first form prints one ``name==floor`` line for each entry in ``[project]
dependencies``. CI's dependency-floors leg installs with that file, so the oldest
version of each package PyBNF says it works with is a version the test suite runs
against (#1023). Every other leg installs the newest releases, and before this leg
no declared floor had ever been tested: pyproject.toml allowed pyparsing 2.4,
which the config parser cannot run on, and numpy and scipy releases that have no
wheels for any Python PyBNF supports.

A runtime dependency with no lower bound is refused. There would be no floor to
test, and the leg would quietly test whichever version the resolver picked.

``--check``, run with the interpreter of the environment that was installed,
compares what is installed with the floors and exits non-zero on any difference.
The leg runs it after installing, so a constraints file that was never applied
fails the job instead of passing it on the newest versions.

Standard library only, so the interpreter that runs it needs nothing installed.
"""
import argparse
import importlib.metadata
import re
import sys
import tomllib
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parents[1] / 'pyproject.toml'

_REQUIREMENT = re.compile(
    r'^([A-Za-z][A-Za-z0-9._-]*)'   # distribution name
    r'(?:\[[^\]]*\])?'             # extras, which do not change the floor
    r'\s*(.*)$'                     # the version specifier
)


def _floor(specifier):
    """The lowest version ``specifier`` admits, or None when it sets no lower bound.

    Raises ValueError for a lower bound that does not name one lowest version: ``>`` (the
    version it names is excluded), ``===`` (arbitrary string equality, not a version), and a
    ``!=`` that excludes the floor itself.
    """
    clauses = [clause for clause in specifier.replace(' ', '').split(',') if clause]
    floor = None
    for clause in clauses:
        if clause.startswith('==='):
            raise ValueError(f'"{clause}" is arbitrary equality, not a version to pin')
        if clause.startswith('>') and not clause.startswith('>='):
            raise ValueError(f'"{clause}" excludes the version it names, so it has no single '
                             f'lowest version; write ">=" with the lowest version that works')
        for operator in ('>=', '==', '~='):
            if clause.startswith(operator):
                floor = clause[len(operator):]
                break
        if floor is not None:
            break
    if floor is None:
        return None
    for clause in clauses:
        if clause.startswith('!=') and _release(clause[2:]) == _release(floor):
            raise ValueError(f'"{clause}" excludes the floor {floor} itself')
    return floor


def floor_pins(dependencies):
    """``['numpy>=1.26', 'bngsim[antimony]>=0.17.0,<1']`` -> ``['numpy==1.26', 'bngsim==0.17.0']``.

    Raises ValueError naming every dependency that declares no lower bound, and every one
    whose lower bound does not name a single version to pin.
    """
    pins, unbounded, unusable = [], [], []
    for requirement in dependencies:
        text, _, marker = requirement.partition(';')
        match = _REQUIREMENT.match(text.strip())
        if match is None:
            raise ValueError(f'cannot read the requirement {requirement!r}')
        name, specifier = match.groups()
        try:
            floor = _floor(specifier)
        except ValueError as error:
            unusable.append(f'{requirement} ({error})')
            continue
        if floor is None:
            unbounded.append(requirement)
            continue
        pin = f'{name}=={floor}'
        if marker.strip():
            pin += f'; {marker.strip()}'
        pins.append(pin)
    problems = []
    if unbounded:
        problems.append(
            'these runtime dependencies declare no lower bound, so there is no floor to '
            'test: ' + ', '.join(unbounded) + '. Give each a ">=" bound in pyproject.toml.')
    if unusable:
        problems.append('these runtime dependencies have a lower bound that is not one version '
                        'to pin: ' + '; '.join(unusable) + '.')
    if problems:
        raise ValueError(' '.join(problems))
    return pins


def _release(version):
    """``'1.26'`` -> ``(1, 26)``, ``'1.26.0'`` -> ``(1, 26)``: compare releases, not spellings."""
    parts = [int(part) for part in re.match(r'\d+(?:\.\d+)*', version).group().split('.')]
    while parts and parts[-1] == 0:
        parts.pop()
    return tuple(parts)


def mismatches(pins, installed_version):
    """Every pin the environment does not match, as ``(name, floor, installed or None)``.

    ``installed_version(name)`` returns the installed version, or raises
    ``importlib.metadata.PackageNotFoundError`` when the package is absent.
    """
    found = []
    for pin in pins:
        name, floor = pin.split(';')[0].split('==')
        try:
            installed = installed_version(name)
        except importlib.metadata.PackageNotFoundError:
            found.append((name, floor, None))
            continue
        if _release(installed) != _release(floor):
            found.append((name, floor, installed))
    return found


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--check', action='store_true',
                        help='compare the installed versions with the floors instead of '
                             'printing the floors')
    args = parser.parse_args(argv)
    dependencies = tomllib.loads(PYPROJECT.read_text())['project']['dependencies']
    try:
        pins = floor_pins(dependencies)
    except ValueError as error:
        print(f'error: {error}', file=sys.stderr)
        return 1
    if not args.check:
        print('\n'.join(pins))
        return 0
    wrong = mismatches(pins, importlib.metadata.version)
    for name, floor, installed in wrong:
        found = 'is not installed' if installed is None else f'is {installed}'
        print(f'error: {name} {found}, but the declared floor is {floor}', file=sys.stderr)
    if wrong:
        print('error: this environment is not the dependency floors, so a test run in it '
              'says nothing about them', file=sys.stderr)
        return 1
    print(f'all {len(pins)} runtime dependencies are at their declared floors')
    return 0


if __name__ == '__main__':
    sys.exit(main())
