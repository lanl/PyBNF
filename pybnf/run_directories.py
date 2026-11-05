"""The folders and files main() deletes before every run that is not a ``--resume``."""

#: Folders of ``output_dir``.
RUN_FOLDERS = ('Simulations', 'Results', 'Initialize', 'FailedSimLogs')

#: Files of ``output_dir``: the pickled algorithms a ``--resume`` reloads.
RUN_FILES = ('alg_backup.bp', 'alg_finished.bp', 'alg_refine_finished.bp')

#: The folder of ``simulation_dir``.
SIMULATION_FOLDER = 'Simulations'
