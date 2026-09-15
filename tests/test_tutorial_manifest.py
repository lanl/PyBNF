"""The tutorial's recovery checks accept what they should and nothing more (#703).

``tests/test_tutorial_examples.py`` runs every committed tutorial conf through the real
backend, so it is opt-in (``recovery``) and a mistake in what it asserts would surface only
there, if at all: a check that accepts too much still passes. What each check accepts is
decided by the manifest and by one comparison, ``_assert_recovered``, and neither needs a
backend, so this module holds them to account in the default tier.

Lesson 25 is the case that needs it. Its plasma curve has two minima, and its check accepts
either one (``other_minima``). The check must still reject a fit between them, such as the
early stop of #648 (ADR-0127), which fitted the curve to 2e-5 at k_transit 11.18 and k_abs
11.50 with k_elim correct to three digits and was caught only by this lesson.
"""
import pytest

from .test_tutorial_examples import EXAMPLES, _assert_recovered, _manifest


def test_other_minima_and_seeds_are_well_formed():
    """Every accepted minimum names exactly the parameters of the documented values, since
    the check reads the fit's best values for those names only, and every conf is run from
    at least one seed, none of them twice."""
    for example in EXAMPLES:
        for check in example.confs:
            where = f'{example.folder}/{check.conf}'
            for minimum in check.other_minima:
                assert set(minimum) == set(check.recover), where
            assert check.seeds, where
            assert len(set(check.seeds)) == len(check.seeds), where


@pytest.fixture
def lesson_25():
    (check,) = _manifest.example_by_folder('25_island_de').confs
    assert check.other_minima, 'lesson 25 should accept its second minimum (#703)'
    assert len(check.seeds) > 1, 'lesson 25 should be checked from several seeds (#703)'
    return check


def test_lesson_25_accepts_either_minimum(lesson_25):
    _assert_recovered('lesson 25', lesson_25, dict(lesson_25.recover))
    for minimum in lesson_25.other_minima:
        _assert_recovered('lesson 25', lesson_25, dict(minimum))


@pytest.mark.parametrize('rec', [
    # Between the minima: the early stop of #648.
    {'k_transit': 11.18, 'k_abs': 11.50, 'k_elim': 0.96},
    # Stalled in the valley, where the lesson's Simplex polish left seed 32 before #703
    # switched it to trf (objective 1.6e-5).
    {'k_transit': 10.7618, 'k_abs': 12.4603, 'k_elim': 0.9605},
    # One rate from each minimum.
    {'k_transit': 12.76, 'k_abs': 14.65, 'k_elim': 0.96},
    # Either pair of rates with the wrong elimination rate.
    {'k_transit': 12.76, 'k_abs': 9.11, 'k_elim': 1.2},
    {'k_transit': 10.09, 'k_abs': 14.65, 'k_elim': 1.2},
], ids=['early stop between the minima', 'stalled in the valley', 'one rate from each',
        'documented rates, wrong k_elim', 'second minimum, wrong k_elim'])
def test_lesson_25_rejects_anything_else(lesson_25, rec):
    with pytest.raises(AssertionError, match='accepted minima'):
        _assert_recovered('lesson 25', lesson_25, rec)
