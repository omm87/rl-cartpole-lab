"""Standalone entry point for the RL CartPole Lab.

The main GUI implementation lives in cartpole_rl_lab.py. This entry point
adapts the repository paths so the project can be cloned and run as an
independent repository instead of only from inside safe-control-gym.
"""

from pathlib import Path
import tkinter as tk

from cartpole_rl_lab import CartPoleRLLab as _CoreCartPoleRLLab


class CartPoleRLLab(_CoreCartPoleRLLab):
    """Core GUI with repository-local safe-control-gym path resolution."""

    def __init__(self, root):
        super().__init__(root)

        # cartpole_rl_lab.py was originally developed inside the upstream
        # safe-control-gym examples tree. In this standalone repository the
        # project root is one level above src/.
        self.project_root = Path(__file__).resolve().parent.parent

        # The core training method resolves the upstream training script as
        #   self.repo_root / "safe_control_gym" / "experiments" / ...
        # scripts/bootstrap.sh therefore clones safe-control-gym into
        #   <project>/safe_control_gym/
        # and setting repo_root to <project> makes that lookup portable.
        self.repo_root = self.project_root


if __name__ == "__main__":
    root = tk.Tk()
    app = CartPoleRLLab(root)
    root.mainloop()
