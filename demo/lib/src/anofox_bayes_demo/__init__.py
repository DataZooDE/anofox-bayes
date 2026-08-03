"""Shared machinery for the anofox-bayes demo TUIs.

Seven demos, one screen. See `demo/README.md` for what each of them shows and
`app.py` for why the screen is laid out the way it is.
"""

from .app import BayesDemoApp, main, run_headless
from .demo import BayesDemo, Param
from .duck import ExtensionMissing, connect, extension_path, extension_version
from .steps import Kind, Pipeline, Step, StepResult

__all__ = [
    "BayesDemo",
    "BayesDemoApp",
    "ExtensionMissing",
    "Kind",
    "Param",
    "Pipeline",
    "Step",
    "StepResult",
    "connect",
    "extension_path",
    "extension_version",
    "main",
    "run_headless",
]
