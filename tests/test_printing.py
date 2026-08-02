"""``PybnfError``'s two message slots: ``user_message`` replaces, ``hint`` appends (#527).

A refusal has two things to tell the user -- *what* went wrong, and *what to do about it* --
and before #527 :class:`~pybnf.printing.PybnfError` had a single slot for both. Passing a
``user_message`` **replaced** the ``log_message`` outright, so a raise site that put its
diagnosis in the log and a generic remedy in ``user_message`` printed only the remedy: the
reason reached the log file and nothing on stdout suggested a reason existed. Every
gradient-path refusal was shaped that way, so four unrelated conditions (a legacy config, a
backend without forward sensitivities, a discrete-event model, an undifferentiable objective)
printed one identical sentence.

``hint`` is the second slot. It is appended to whatever the user was going to see, never
substituted for it, so a raise site can carry a remedy *without* spending the slot that holds
its diagnosis. ``user_message``'s replace semantics stay available for the case they were
written for -- restating a technical diagnosis in user-facing terms (a missing ``_SD`` column,
a failing simulation) -- where the log line and the user line say the same thing at different
depths rather than saying two different things.

The oracle throughout is what ``pybnf.pybnf`` prints and logs: stdout gets ``e.message``, the
log file gets ``e.log_message``.
"""
import pytest

from pybnf.printing import PybnfError


DIAGNOSIS = "model 'decay' contains discrete events"
REMEDY = 'Use a metaheuristic job_type instead.'


def test_a_hint_is_appended_to_the_diagnosis_not_substituted_for_it():
    """The bug #527 reports, stated as its fix: with only a ``log_message`` and a ``hint``,
    the user sees **both** -- the diagnosis first, then the remedy on its own indented line.
    Passing the remedy as ``user_message`` instead would print the remedy alone."""
    e = PybnfError(DIAGNOSIS, hint=REMEDY)

    assert e.message == DIAGNOSIS + '\n  -> ' + REMEDY
    assert DIAGNOSIS in e.message and REMEDY in e.message


def test_a_hint_appends_to_the_user_message_that_replaced_the_log_message():
    """The two slots compose and keep their own jobs: ``user_message`` still replaces the
    log line for the user, and the hint appends to *that* -- so a site can both restate its
    diagnosis in user terms and suggest a remedy."""
    e = PybnfError(DIAGNOSIS, 'This model has events PyBNF cannot differentiate.', hint=REMEDY)

    assert e.message == 'This model has events PyBNF cannot differentiate.\n  -> ' + REMEDY
    assert DIAGNOSIS not in e.message          # user_message replaced it, as documented


def test_several_hints_each_get_their_own_line():
    """A refusal with more than one way out (fix the config, or fall back to a gradient-free
    job_type) lists each remedy as its own ``->`` line rather than running them together --
    the edition-2 gate's shape."""
    e = PybnfError(DIAGNOSIS, hint=["Opt into edition 2 ('edition = 2').", REMEDY])

    assert e.message.splitlines() == [
        DIAGNOSIS,
        "  -> Opt into edition 2 ('edition = 2').",
        '  -> ' + REMEDY,
    ]
    assert e.hints == ["Opt into edition 2 ('edition = 2').", REMEDY]


def test_the_log_message_never_carries_the_hint():
    """The hint is user-facing guidance, so it changes only what stdout gets; the log line
    stays the bare diagnosis every raise site wrote it to be (``logger.error(e.log_message)``)."""
    e = PybnfError(DIAGNOSIS, hint=REMEDY)

    assert e.log_message == DIAGNOSIS


@pytest.mark.parametrize('args, expected', [
    ((DIAGNOSIS,), DIAGNOSIS),                                  # log message reaches the user
    ((DIAGNOSIS, 'A friendlier retelling.'), 'A friendlier retelling.'),   # ... unless replaced
])
def test_an_error_with_no_hint_is_unchanged(args, expected):
    """Parity for the ~90 raise sites that pass no hint: one argument still shows the log
    message to the user, two still replace it, and neither grows a trailing line."""
    e = PybnfError(*args)

    assert e.message == expected
    assert e.log_message == DIAGNOSIS
    assert e.hints == []
