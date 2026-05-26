"""
Regression tests for subprocess execution in model.execute() methods.

These tests verify the behavioral contracts of subprocess handling without
requiring BioNetGen. They use simple Python commands as stand-in subprocesses.
The contracts tested here must survive any refactoring of the subprocess
internals (e.g. switching from subprocess.run to subprocess.Popen).
"""

import os
import sys
import signal
import tempfile
import time
from subprocess import run, Popen, STDOUT, PIPE, CalledProcessError, TimeoutExpired

import pytest


PYTHON = sys.executable


class TestSubprocessTimeout:
    """Verify that subprocess timeout raises TimeoutExpired and kills the process."""

    def test_timeout_raises(self):
        """A subprocess that exceeds its timeout should raise TimeoutExpired."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as lf:
            log_path = lf.name
        try:
            with open(log_path, 'w') as lf:
                with pytest.raises(TimeoutExpired):
                    run([PYTHON, '-c', 'import time; time.sleep(60)'],
                        check=True, stderr=STDOUT, stdout=lf, timeout=1)
        finally:
            os.unlink(log_path)

    def test_timeout_kills_direct_child(self):
        """After timeout, the direct child process should be dead."""
        # Spawn a process that writes its PID to a file, then sleeps
        with tempfile.TemporaryDirectory() as tmpdir:
            pid_file = os.path.join(tmpdir, 'child.pid')
            script = (
                f"import os, time; "
                f"open('{pid_file}', 'w').write(str(os.getpid())); "
                f"time.sleep(60)"
            )
            log_path = os.path.join(tmpdir, 'test.log')
            with open(log_path, 'w') as lf:
                try:
                    run([PYTHON, '-c', script],
                        check=True, stderr=STDOUT, stdout=lf, timeout=2)
                except TimeoutExpired:
                    pass

            # Give OS a moment to clean up
            time.sleep(0.5)
            with open(pid_file) as f:
                child_pid = int(f.read().strip())
            # The direct child should be dead
            try:
                os.kill(child_pid, 0)  # signal 0 = test if alive
                alive = True
            except OSError:
                alive = False
            assert not alive, f"Direct child process {child_pid} is still alive after timeout"

    def test_timeout_leaves_grandchild_zombie(self):
        """
        Demonstrate the current bug: a grandchild process survives timeout.
        subprocess.run() only kills the direct child, not its children.

        This test documents the BROKEN behavior. After the fix (process group
        killing), this test should be updated to assert the grandchild is dead.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            pid_file = os.path.join(tmpdir, 'grandchild.pid')
            # Parent spawns a child that writes its PID and sleeps
            script = (
                f"import subprocess, sys, time; "
                f"p = subprocess.Popen([sys.executable, '-c', "
                f"\"import os, time; open('{pid_file}', 'w').write(str(os.getpid())); time.sleep(60)\"]); "
                f"time.sleep(60)"
            )
            log_path = os.path.join(tmpdir, 'test.log')
            with open(log_path, 'w') as lf:
                try:
                    run([PYTHON, '-c', script],
                        check=True, stderr=STDOUT, stdout=lf, timeout=2)
                except TimeoutExpired:
                    pass

            # Wait for grandchild to write its PID
            time.sleep(1)
            if not os.path.exists(pid_file):
                pytest.skip("Grandchild didn't start in time")

            with open(pid_file) as f:
                grandchild_pid = int(f.read().strip())

            try:
                os.kill(grandchild_pid, 0)
                grandchild_alive = True
            except OSError:
                grandchild_alive = False

            # BUG: grandchild is still alive — this is the zombie problem (#83)
            # After the fix, change this to: assert not grandchild_alive
            if grandchild_alive:
                os.kill(grandchild_pid, signal.SIGKILL)
                # This documents the known bug
                pass
            # For now, just clean up — the fix will change this assertion


class TestSubprocessNormalExecution:
    """Verify that normal subprocess completion works correctly."""

    def test_successful_run_no_exception(self):
        """A subprocess that exits 0 should not raise."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, 'test.log')
            with open(log_path, 'w') as lf:
                run([PYTHON, '-c', 'print("hello")'],
                    check=True, stderr=STDOUT, stdout=lf, timeout=10)
            with open(log_path) as f:
                assert 'hello' in f.read()

    def test_stdout_captured_to_file(self):
        """Subprocess stdout should be redirected to the log file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, 'test.log')
            with open(log_path, 'w') as lf:
                run([PYTHON, '-c', 'print("test_output_12345")'],
                    check=True, stderr=STDOUT, stdout=lf, timeout=10)
            with open(log_path) as f:
                content = f.read()
            assert 'test_output_12345' in content

    def test_stderr_merged_to_stdout(self):
        """Subprocess stderr should be merged with stdout into the log file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, 'test.log')
            with open(log_path, 'w') as lf:
                run([PYTHON, '-c', 'import sys; sys.stderr.write("err_msg_67890\\n")'],
                    check=True, stderr=STDOUT, stdout=lf, timeout=10)
            with open(log_path) as f:
                content = f.read()
            assert 'err_msg_67890' in content

    def test_stdout_to_pipe(self):
        """Subprocess stdout via PIPE should be accessible in result (SBML pattern)."""
        result = run([PYTHON, '-c', 'import sys; sys.stdout.buffer.write(b"binary_data")'],
                     stdout=PIPE, check=True, timeout=10)
        assert result.stdout == b'binary_data'

    def test_stdin_input(self):
        """Subprocess should receive input via stdin (SBML pattern)."""
        result = run([PYTHON, '-c', 'import sys; data = sys.stdin.buffer.read(); sys.stdout.buffer.write(data)'],
                     stdout=PIPE, check=True, timeout=10, input=b'round_trip_test')
        assert result.stdout == b'round_trip_test'


class TestSubprocessErrorHandling:
    """Verify that subprocess errors raise the correct exceptions."""

    def test_nonzero_exit_raises_calledprocesserror(self):
        """A subprocess that exits non-zero with check=True should raise CalledProcessError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, 'test.log')
            with open(log_path, 'w') as lf:
                with pytest.raises(CalledProcessError):
                    run([PYTHON, '-c', 'import sys; sys.exit(1)'],
                        check=True, stderr=STDOUT, stdout=lf, timeout=10)

    def test_calledprocesserror_has_returncode(self):
        """CalledProcessError should contain the actual return code."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, 'test.log')
            with open(log_path, 'w') as lf:
                try:
                    run([PYTHON, '-c', 'import sys; sys.exit(42)'],
                        check=True, stderr=STDOUT, stdout=lf, timeout=10)
                    assert False, "Should have raised"
                except CalledProcessError as e:
                    assert e.returncode == 42

    def test_signal_death_raises_calledprocesserror(self):
        """A subprocess killed by signal should raise CalledProcessError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, 'test.log')
            script = 'import os, signal; os.kill(os.getpid(), signal.SIGTERM)'
            with open(log_path, 'w') as lf:
                with pytest.raises(CalledProcessError):
                    run([PYTHON, '-c', script],
                        check=True, stderr=STDOUT, stdout=lf, timeout=10)


class TestProcessGroupKilling:
    """
    Tests for process-group-based killing on timeout.

    These tests verify that when a subprocess times out, its entire process
    tree (including grandchildren) is killed. This is the fix for issue #83.
    """

    def test_grandchild_killed_on_timeout(self):
        """
        After timeout, grandchild processes should also be killed.
        This is the core fix for issue #83 (zombie run_network processes).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            grandchild_pid_file = os.path.join(tmpdir, 'grandchild.pid')
            # Script that spawns a grandchild, which writes its PID and sleeps
            script = (
                f"import subprocess, sys, time; "
                f"p = subprocess.Popen([sys.executable, '-c', "
                f"\"import os, time; open('{grandchild_pid_file}', 'w').write(str(os.getpid())); time.sleep(60)\"]); "
                f"time.sleep(60)"
            )
            log_path = os.path.join(tmpdir, 'test.log')

            # Use process group killing pattern (the fix)
            with open(log_path, 'w') as lf:
                proc = Popen([PYTHON, '-c', script],
                             stderr=STDOUT, stdout=lf, start_new_session=True)
                try:
                    proc.wait(timeout=2)
                except TimeoutExpired:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    proc.wait()

            # Wait for OS cleanup
            time.sleep(0.5)

            if not os.path.exists(grandchild_pid_file):
                pytest.skip("Grandchild didn't start in time")

            with open(grandchild_pid_file) as f:
                grandchild_pid = int(f.read().strip())

            try:
                os.kill(grandchild_pid, 0)
                alive = True
            except OSError:
                alive = False

            assert not alive, (
                f"Grandchild process {grandchild_pid} is still alive after "
                f"process group kill — zombie processes would result"
            )

    def test_process_group_kill_with_stdin(self):
        """
        Process group killing should work with stdin input (SBML pattern).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            grandchild_pid_file = os.path.join(tmpdir, 'grandchild.pid')
            script = (
                f"import subprocess, sys, time; "
                f"data = sys.stdin.buffer.read(4); "  # read some input
                f"p = subprocess.Popen([sys.executable, '-c', "
                f"\"import os, time; open('{grandchild_pid_file}', 'w').write(str(os.getpid())); time.sleep(60)\"]); "
                f"time.sleep(60)"
            )

            proc = Popen([PYTHON, '-c', script],
                         stdout=PIPE, stdin=PIPE, start_new_session=True)
            try:
                proc.communicate(input=b'test', timeout=2)
            except TimeoutExpired:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                proc.wait()

            time.sleep(0.5)

            if not os.path.exists(grandchild_pid_file):
                pytest.skip("Grandchild didn't start in time")

            with open(grandchild_pid_file) as f:
                grandchild_pid = int(f.read().strip())

            try:
                os.kill(grandchild_pid, 0)
                alive = True
            except OSError:
                alive = False

            assert not alive, f"Grandchild {grandchild_pid} survived process group kill"

    def test_normal_completion_with_process_group(self):
        """
        Using start_new_session should not affect normal subprocess completion.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, 'test.log')
            with open(log_path, 'w') as lf:
                proc = Popen([PYTHON, '-c', 'print("pgid_test_ok")'],
                             stderr=STDOUT, stdout=lf, start_new_session=True)
                proc.wait(timeout=10)
            assert proc.returncode == 0
            with open(log_path) as f:
                assert 'pgid_test_ok' in f.read()

    def test_error_with_process_group(self):
        """
        Using start_new_session should not affect error detection.
        """
        proc = Popen([PYTHON, '-c', 'import sys; sys.exit(7)'],
                     start_new_session=True)
        proc.wait(timeout=10)
        assert proc.returncode == 7
