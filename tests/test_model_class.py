from .context import pset, raises
from os import environ
from os import mkdir
from os.path import exists
from shutil import rmtree
from pybnf.printing import PybnfError

import pytest
import re


class TestModel:
    @classmethod
    def setup_class(cls):
        """Define constants to be used in tests"""
        cls.output_root = f"test_model_class_{environ.get('PYTEST_XDIST_WORKER', 'local')}"
        rmtree(cls.output_root, ignore_errors=True)
        mkdir(cls.output_root)
        cls.file1 = 'bngl_files/Simple.bngl'
        cls.file2 = 'bngl_files/ParamsEverywhere.bngl'
        cls.file3 = 'bngl_files/Tricky.bngl'
        cls.file4 = 'bngl_files/NFmodel.bngl'
        cls.file5 = 'bngl_files/TrickyWP_p1_5.net'
        cls.file6 = 'bngl_files/Simple_nogen.bngl'
        cls.file7 = 'bngl_files/Simple_SampleTimes.bngl'

        cls.file1a = 'bngl_files/Simple_Answer.bngl'
        cls.file1b = 'bngl_files/Simple_GenOnly.bngl'
        cls.file1c = 'bngl_files/Simple_AddActions.bngl'

        cls.savefile_prefix = f'{cls.output_root}/NoseTest_Save'
        cls.savefile2_prefix = f'{cls.output_root}/NoseTest_Save2'
        cls.savefile3_prefix = f'{cls.output_root}/NoseTest_Save3'
        cls.savefile4_prefix = f'{cls.output_root}/NoseTest_Save4'
        cls.savefile5_prefix = f'{cls.output_root}/NoseTest_Save5'

        cls.params1 = [
            pset.FreeParameter('kase__FREE', 'normal_var', 0, 1, value=3.8),
            pset.FreeParameter('pase__FREE', 'normal_var', 0, 1, value=0.16),
            pset.FreeParameter('koff__FREE', 'normal_var', 0, 1, value=4.4e-3)
        ]
        cls.params2 = [
            pset.FreeParameter('kase__FREE', 'normal_var', 0, 1, value=3.8),
            pset.FreeParameter('pase__FREE', 'normal_var', 0, 1, value=0.16),
            pset.FreeParameter('wrongname__FREE', 'normal_var', 0, 1, value=4.4e-3)
        ]

    @classmethod
    def teardown_class(cls):
        rmtree(cls.output_root, ignore_errors=True)

    def test_no_gen_command(self):
        model = pset.BNGLModel(self.file6)
        assert model.generates_network
        assert model.generate_network_line == 'generate_network({overwrite=>1})'

    def test_generate_network_option_augments_default(self):
        """#473: the model-scoped ``generate_network`` conf option is injected into the
        synthesized default line when the model carries no explicit ``generate_network``
        (the edition-2 stripped-actions case). Simple_nogen has a simulate but no
        generate_network line, so the option supplies the cap the actions block would
        have carried -- without it a crosslinking model synthesizes an unbounded network."""
        model = pset.BNGLModel(self.file6, generate_network_options='max_stoich=>{EGF=>4,EGFR=>4}')
        assert model.generates_network
        assert model.generate_network_line == 'generate_network({overwrite=>1,max_stoich=>{EGF=>4,EGFR=>4}})'

    def test_generate_network_option_none_is_bare_default(self):
        """#473: with no option (None -- the default for every job that does not set the
        key), the synthesized line is byte-identical to the pre-#473 bare default."""
        model = pset.BNGLModel(self.file6, generate_network_options=None)
        assert model.generate_network_line == 'generate_network({overwrite=>1})'

    def test_generate_network_option_normalizes_leading_comma(self):
        """#473: the injected fragment is forgiving of surrounding whitespace and a stray
        leading/trailing comma, so ``, max_iter=>3 `` still yields a clean single-comma join."""
        model = pset.BNGLModel(self.file6, generate_network_options=' , max_iter=>3 ')
        assert model.generate_network_line == 'generate_network({overwrite=>1,max_iter=>3})'

    def test_generate_network_option_model_line_wins(self):
        """#473 precedence: an explicit ``generate_network`` line in the model always wins
        over the conf option (it is captured in __init__ before any default fires). Simple
        carries ``generate_network({overwrite=>1})``, so the option is ignored."""
        model = pset.BNGLModel(self.file1, generate_network_options='max_stoich=>{EGF=>4,EGFR=>4}')
        assert model.generate_network_line == 'generate_network({overwrite=>1})'

    def test_initialize(self):
        model1 = pset.BNGLModel(self.file1)
        assert model1.param_names == ('kase__FREE', 'koff__FREE', 'pase__FREE')

        model2 = pset.BNGLModel(self.file2)
        assert model2.param_names == (
            'Ag_tot_1__FREE', 'kase__FREE', 'koff__FREE', 'kon__FREE', 'pase__FREE', 't_end__FREE')

        model3 = pset.BNGLModel(self.file3)
        assert model3.param_names == ('__koff2__FREE', 'kase__FREE', 'koff__FREE', 'pase__FREE')

    def test_model_param_names_is_the_full_namespace(self):
        """ADR-0034: ``model_param_names`` is every ``begin parameters`` id (in source
        order), the new-era bind-by-id namespace -- distinct from ``param_names`` (the
        legacy ``__FREE`` tokens). Tricky declares its fit knobs through *expressions*
        (``koff koff__FREE+__koff2__FREE*T``), so the two sets share no member: the
        parameter ids are ``koff``/``kase``/``pase``..., never the ``*__FREE`` tokens."""
        model3 = pset.BNGLModel(self.file3)  # Tricky.bngl
        assert model3.model_param_names == (
            'f', 'NA', 'T', 'Vchannel', 'Nchannel', 'Vecf', 'Vcyt', 'Ag_conc1',
            'Ag_tot_1', 'R_tot', 'kon', 'koff', 'kase', 'pase', 'H_tot', 'kdegran')
        # No __FREE token is a parameter id, and every fit knob's *host* parameter is.
        assert set(model3.model_param_names).isdisjoint(model3.param_names)
        assert {'koff', 'kase', 'pase'} <= set(model3.model_param_names)

        # ParamsEverywhere: a __FREE token (t_end__FREE, an action arg) need not be a
        # parameter id, so param_names and model_param_names are genuinely independent.
        model2 = pset.BNGLModel(self.file2)
        assert 't_end__FREE' in model2.param_names
        assert 't_end' not in model2.model_param_names
        assert 'kase' in model2.model_param_names

    def test_model_param_names_matches_bngsim_param_block_parser(self):
        """The advertised namespace must equal what the in-process bngsim NF backend
        binds: both parse the same ``model_lines`` (ADR-0034 keeps them in step)."""
        from pybnf.bngsim_model.expressions import _parse_bngl_param_block
        for f in (self.file1, self.file2, self.file3):
            model = pset.BNGLModel(f)
            expected = tuple(name for name, _expr in _parse_bngl_param_block(model.model_lines))
            assert model.model_param_names == expected

    def test_no_free_marker_required_when_suppressed(self):
        """A new-era model carries no ``__FREE`` markers (ADR-0034). The legacy load
        still errors on a marker-free model (the historical contract); passing
        ``suppress_free_param_error`` -- the seam config flips under edition >= 2 --
        loads it and still exposes the full bind-by-id namespace."""
        from pybnf.pset import ModelError
        with pytest.raises(ModelError):
            pset.BNGLModel('bngl_files/e2e_ode_decay.bngl')
        model = pset.BNGLModel('bngl_files/e2e_ode_decay.bngl', suppress_free_param_error=True)
        assert model.param_names == ()                       # no __FREE tokens
        assert model.model_param_names == ('S0', 'k')        # full bind-by-id namespace

    def test_model_text_overrides_bare_param_id_in_place(self):
        """ADR-0034: for the file+subprocess backend, a new-era free parameter (a bare
        parameter id, no marker) overrides its parameters-block value *in place* with the
        fit value -- the same value the in-process set_param would apply -- instead of
        running the model at its nominal value (a silent wrong fit)."""
        model = pset.BNGLModel('bngl_files/e2e_ode_decay.bngl', suppress_free_param_error=True)
        ps = pset.PSet([pset.FreeParameter('k', 'uniform_var', 0, 10, value=0.5),
                        pset.FreeParameter('S0', 'uniform_var', 0, 200, value=42.0)])
        text = model.copy_with_param_set(ps).model_text()
        assert '\nk 0.5\n' in text and '\nS0 42.0\n' in text   # fit values applied
        assert '0.3' not in text and '100' not in text          # nominal values gone

    def test_model_text_legacy_free_injection_unchanged(self):
        """The legacy __FREE path is byte-for-byte unchanged: marker values are injected
        and the original parameter lines are carried verbatim (markers are disjoint from
        the parameter ids, so the new bind-by-id branch never fires)."""
        model = pset.BNGLModel(self.file1)  # Simple.bngl
        ps = pset.PSet([pset.FreeParameter('kase__FREE', 'uniform_var', 0, 10, value=3.8),
                        pset.FreeParameter('koff__FREE', 'uniform_var', 0, 10, value=1.1),
                        pset.FreeParameter('pase__FREE', 'uniform_var', 0, 10, value=2.2)])
        text = model.copy_with_param_set(ps).model_text()
        assert 'kase__FREE 3.8' in text                         # marker injected
        assert 'kon 1e7*T/(NA*Vecf)' in text                    # model carried verbatim

    def test_init_with_pset(self):
        ps1 = pset.PSet(self.params1)
        model1 = pset.BNGLModel(self.file1, ps1)
        assert model1.param_set['kase__FREE'] == 3.8

    @raises(ValueError)
    def test_init_with_pset_error(self):
        ps1 = pset.PSet(self.params2)
        model1 = pset.BNGLModel(self.file1, ps1)
        assert model1.param_set['kase__FREE'] == 3.8

    def test_copy_with_param_set(self):
        model1 = pset.BNGLModel(self.file1)
        ps1 = pset.PSet(self.params1)
        model1b = model1.copy_with_param_set(ps1)
        assert model1b.param_set['kase__FREE'] == 3.8

        nmodel1 = pset.NetModel('TrickyWP_p1_5', [], [], [], nf=self.file5)
        ps1 = pset.PSet([pset.FreeParameter('Nchannel', 'normal_var', 0, 1, value=20)])
        nmodel1b = nmodel1.copy_with_param_set(ps1)
        nmodel1b.save(self.savefile4_prefix)

        with open(self.savefile4_prefix + '.net') as f:
            nmodel1b_lines = f.readlines()

        assert re.search(r'Nchannel\s+20\s',nmodel1b_lines[6])

    @raises(PybnfError)
    def test_set_param_set_error(self):
        model1 = pset.BNGLModel(self.file1)
        ps2 = pset.PSet(self.params2)
        model1.copy_with_param_set(ps2)

    def test_model_text(self):
        ps1 = pset.PSet(self.params1)
        model1 = pset.BNGLModel(self.file1, ps1)

        f_answer = open(self.file1a)  # File containing the correct output for model_text()
        answer = f_answer.read()
        f_answer.close()
        assert model1.model_text() == answer

    def test_bnglmodel_save(self):
        ps1 = pset.PSet(self.params1)
        model1 = pset.BNGLModel(self.file1, ps1)

        model1.save(self.savefile_prefix)

        f_myguess = open(self.savefile_prefix + '.bngl')
        myguess = f_myguess.read()
        f_myguess.close()

        f_answer = open(self.file1a)  # File containing the correct output for model_text()
        answer = f_answer.read()
        f_answer.close()

        assert myguess == answer

        model1 = pset.BNGLModel(self.file1, ps1)

        model1.save(self.savefile2_prefix, gen_only=True)
        f_myguess2 = open(self.savefile2_prefix + '.bngl')
        myguess2 = f_myguess2.read()
        f_myguess2.close()

        f_answer2 = open(self.file1b)
        answer2 = f_answer2.read()
        f_answer2.close()

        assert myguess2 == answer2

    def test_bngl_config_actions(self):
        ps1 = pset.PSet(self.params1)
        model1 = pset.BNGLModel(self.file1, ps1)
        a1 = pset.TimeCourse({'time': 50, 'step': 10, 'model': 'Simple', 'suffix': 's2'})
        model1.add_action(a1)
        a2 = pset.ParamScan({'min': 10, 'max': 60, 'step': 10, 'time': 5, 'suffix': 's3', 'model': 'Simple',
                             'param': 'kon'})
        model1.add_action(a2)
        model1.save(self.savefile5_prefix)

        f_myguess = open(self.savefile5_prefix + '.bngl')
        myguess = f_myguess.read()
        f_myguess.close()

        f_answer = open(self.file1c)
        answer = f_answer.read()
        f_answer.close()

        assert myguess == answer

    def test_config_action_sets_stochastic_flag(self):
        # #471: on the edition-2 surface a model carries no `begin actions` block, so the
        # parse-time regex never runs; the simulate/parameter_scan is synthesized from the
        # experiment line via add_action. A stochastic method must set model.stochastic so
        # the `smoothing` misuse check doesn't false-alarm. (Simple.bngl's own actions are
        # ODE, so the flag starts False.)
        for method in ('ssa', 'pla', 'nf', 'rm', 'rulemonkey'):
            model = pset.BNGLModel(self.file1, suppress_free_param_error=True)
            assert not model.stochastic
            model.add_action(pset.TimeCourse({'time': 5, 'suffix': 's', 'method': method}))
            assert model.stochastic, method

    def test_config_action_ode_leaves_stochastic_false(self):
        # Regression companion to the above: a deterministic config action must NOT flip the
        # flag, so an all-ODE edition-2 fit still gets the smoothing warning it deserves.
        model = pset.BNGLModel(self.file1, suppress_free_param_error=True)
        model.add_action(pset.TimeCourse({'time': 5, 'suffix': 's', 'method': 'ode'}))
        model.add_action(pset.ParamScan({'min': 1, 'max': 2, 'step': 1, 'time': 5,
                                         'suffix': 'sc', 'param': 'kon', 'method': 'ode'}))
        assert not model.stochastic

    def test_config_param_scan_sets_stochastic_flag(self):
        # The scan synthesis path (add_action, ParamScan branch) must set the flag too:
        # a network-free dose-response (examples/real-world/tlbr) uses method: nf.
        model = pset.BNGLModel(self.file1, suppress_free_param_error=True)
        model.add_action(pset.ParamScan({'min': 1, 'max': 2, 'step': 1, 'time': 5,
                                         'suffix': 'sc', 'param': 'kon', 'method': 'nf'}))
        assert model.stochastic

    def test_action_suffixes(self):
        m0 = pset.BNGLModel(self.file1)
        assert len(m0.suffixes) == 1
        assert m0.suffixes[0] == ('simulate', 'p1_5')

        m1 = pset.BNGLModel(self.file3)
        assert len(m1.suffixes) == 2
        assert m1.suffixes[1] == ('parameter_scan', 'thing')

    def test_actions(self):
        m0 = pset.BNGLModel(self.file1)
        assert len([a for a in m0.actions if len(a) > 0 and a[0] != '#']) == 2
        for a in m0.actions:
            assert re.search('setOption', a) is None

    def test_find_t_length_n_steps(self):
        # n_steps=>N gives N+1 output rows, so the stored length is N.
        m0 = pset.BNGLModel(self.file1)
        assert m0.find_t_length() == {'p1_5': 50}

    def test_find_t_length_sample_times(self):
        # Regression for issue #390: a simulate action using sample_times=>[...]
        # instead of n_steps must not raise IndexError. With M listed times the
        # simulation produces M output rows, so the stored length is M-1.
        m = pset.BNGLModel(self.file7)
        assert m.find_t_length() == {'p1_5': 5}

    def test_network_check(self):
        model0 = pset.BNGLModel(self.file1)
        assert model0.generates_network
        model1 = pset.BNGLModel(self.file4)
        assert not model1.generates_network

    def test_has_observables(self):
        model = pset.BNGLModel(self.file1)
        assert model.has_observables

    def test_no_observables(self):
        """Model with empty observables block should have has_observables=False"""
        import tempfile
        import os
        bngl_content = """begin model
  begin parameters
    v1 v1__FREE
  end parameters
  begin molecule types
    A()
  end molecule types
  begin seed species
    A() 100
  end seed species
  begin observables
  end observables
  begin reaction rules
    0->A() 1
  end reaction rules
end model
begin actions
  generate_network({overwrite=>1})
  simulate({method=>"ode",t_end=>10,n_steps=>10,suffix=>"tc"})
end actions
"""
        fd, path = tempfile.mkstemp(suffix='.bngl')
        try:
            with os.fdopen(fd, 'w') as f:
                f.write(bngl_content)
            model = pset.BNGLModel(path)
            assert not model.has_observables
        finally:
            os.remove(path)

    def test_netfile_read(self):
        netmodel = pset.NetModel('TrickyWP_p1_5', [], [], [], nf=self.file5)
        assert len(netmodel.netfile_lines) == 48

    def test_netfile_pcopy_and_save(self):
        netmodel = pset.NetModel('TrickyWP_p1_5', [], [], [], nf=self.file5)
        params = [pset.FreeParameter('Vchannel', 'normal_var', 0, 1, value=1e-5),
                  pset.FreeParameter('H_tot', 'normal_var', 0, 1, value=3.4)]
        ps = pset.PSet(params)
        new_netmodel = netmodel.copy_with_param_set(ps)
        pl0 = new_netmodel.netfile_lines[5]
        pl1 = new_netmodel.netfile_lines[16]
        assert re.search('Vchannel.*1e-05', pl0)
        assert re.search('H_tot.*3.4', pl1)
        for i in range(len(new_netmodel.netfile_lines)):
            if i == 5 or i == 16:
                continue
            else:
                assert new_netmodel.netfile_lines[i] == netmodel.netfile_lines[i]
        new_netmodel.save(self.savefile3_prefix)
        assert exists(self.savefile3_prefix + '.net')
        assert exists(self.savefile3_prefix + '.bngl')

        with open(self.savefile3_prefix + '.bngl') as bf:
            bf_lines = bf.readlines()

        assert re.match('readFile', bf_lines[0])

    def test_base_model_abstract_methods_raise(self):
        """ROB-6: the abstract Model.copy_with_param_set / save build a
        NotImplementedError but must actually raise it -- otherwise a subclass
        that forgets to override silently returns None instead of erroring."""
        m = pset.Model()
        with pytest.raises(NotImplementedError):
            m.copy_with_param_set(None)
        with pytest.raises(NotImplementedError):
            m.save('some_prefix')
