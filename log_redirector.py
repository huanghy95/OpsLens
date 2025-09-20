import os
import sys
import threading
import fcntl
from contextlib import contextmanager
from typing import Optional
import traceback


class LogRedirector:
    """
    log redirector

    Features:
    - Supports context manager pattern
    - Automatically creates directory structure
    - Concurrent-safe file writing
    - Automatically restores original output streams
    """

    def __init__(self, log_file_path: str, redirect_stdout: bool = True, redirect_stderr: bool = True):
        """
        Initialize log redirector

        Args:
            log_file_path: log file path
            redirect_stdout: whether to redirect standard output
            redirect_stderr: whether to redirect standard error output
        """
        self.log_file_path = log_file_path
        self.redirect_stdout = redirect_stdout
        self.redirect_stderr = redirect_stderr

        # save original output streams
        self.original_stdout = None
        self.original_stderr = None

        # log file handle
        self.log_file = None

        # thread lock, ensure concurrent safety
        self.lock = threading.Lock()

    def _ensure_directory_exists(self):
        """Ensure log file directory exists"""
        try:
            log_dir = os.path.dirname(self.log_file_path)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
                print(f"[LogRedirector] Created directory: {log_dir}")
        except Exception as e:
            print(f"[LogRedirector] Error creating directory: {e}")
            print(traceback.format_exc())

    def _open_log_file(self):
        """Open log file"""
        try:
            self._ensure_directory_exists()
            self.log_file = open(self.log_file_path, 'w', encoding='utf-8', buffering=1)
            print(f"[LogRedirector] Opened log file: {self.log_file_path}")
        except Exception as e:
            print(f"[LogRedirector] Error opening log file {self.log_file_path}: {e}")
            print(traceback.format_exc())
            raise

    def _close_log_file(self):
        """Close log file"""
        if self.log_file:
            try:
                self.log_file.close()
                print(f"[LogRedirector] Closed log file: {self.log_file_path}")
            except Exception as e:
                print(f"[LogRedirector] Error closing log file: {e}")
            finally:
                self.log_file = None

    def __enter__(self):
        """Enter context manager"""
        with self.lock:
            try:
                # save original output streams
                self.original_stdout = sys.stdout
                self.original_stderr = sys.stderr

                # open log file
                self._open_log_file()

                # output redirecting start info to original output
                print(f"[LogRedirector] Started redirecting to: {self.log_file_path}")

                # redirect output streams
                if self.redirect_stdout and self.log_file:
                    sys.stdout = self.log_file

                if self.redirect_stderr and self.log_file:
                    sys.stderr = self.log_file

                return self

            except Exception as e:
                # if error, restore original state
                self._restore_streams()
                self._close_log_file()
                print(f"[LogRedirector] Error in __enter__: {e}")
                print(traceback.format_exc())
                raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager"""
        with self.lock:
            try:
                # restore original output streams
                self._restore_streams()

                # close log file
                self._close_log_file()

                print(f"[LogRedirector] Stopped redirecting from: {self.log_file_path}")

            except Exception as e:
                print(f"[LogRedirector] Error in __exit__: {e}")
                print(traceback.format_exc())

    def _restore_streams(self):
        """Restore original output streams"""
        if self.original_stdout:
            sys.stdout = self.original_stdout
        if self.original_stderr:
            sys.stderr = self.original_stderr


def create_log_path(dataset: str, model: str, idx: int, base_dir: str = "/runlog") -> str:
    """
    Create log file path

    Args:
        dataset: dataset name
        model: model name
        idx: task index
        base_dir: base directory, default is /runlog

    Returns:
        str: full log file path
    """
    # handle special characters in dataset name
    safe_dataset = dataset.replace('/', '_').replace('\\', '_')
    safe_model = model.replace('/', '_').replace('\\', '_')

    log_path = os.path.join(base_dir, safe_dataset, safe_model, f"{idx}.log")
    return log_path


@contextmanager
def redirect_output_to_file(dataset: str, model: str, idx: int, base_dir: str = "/runlog"):
    """
    Convenience context manager, used to redirect output to specified format log file

    Args:
        dataset: dataset name
        model: model name
        idx: task index
        base_dir: base directory, default is /runlog

    Usage:
        with redirect_output_to_file("Bank", "gemini", 0):
            print("This will be redirected to /runlog/Bank/gemini/0.log")
    """
    log_path = create_log_path(dataset, model, idx, base_dir)
    redirector = LogRedirector(log_path)

    with redirector:
        yield log_path


class ThreadSafeFileWriter:
    """
    Thread-safe file writer, used for concurrent log writing

    Features:
    - Supports concurrent log writing
    - Automatically creates directory structure
    - Uses file lock to ensure safe writing
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.lock = threading.Lock()

    def write_with_lock(self, content: str):
        """
        Write content with file lock

        Args:
            content: content to write
        """
        with self.lock:
            try:
                # ensure directory exists
                log_dir = os.path.dirname(self.file_path)
                if log_dir and not os.path.exists(log_dir):
                    os.makedirs(log_dir, exist_ok=True)

                # write with file lock
                with open(self.file_path, 'w', encoding='utf-8') as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    try:
                        f.write(content)
                        f.flush()
                    finally:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)

            except Exception as e:
                print(f"[ThreadSafeFileWriter] Error writing to {self.file_path}: {e}")
                print(traceback.format_exc())


def test_log_redirector():
    """Test log redirector"""
    print("Testing log redirector...")

    # test basic redirect
    with redirect_output_to_file("TestDataset", "TestModel", 999, "/tmp/test_runlog"):
        print("This should be redirected to log file")
        print("Multiple lines")
        print("Should all appear in the log")

    print("Back to normal output")


if __name__ == "__main__":
    test_log_redirector()