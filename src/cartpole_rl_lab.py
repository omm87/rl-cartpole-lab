import copy
import json
import math
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import torch
import yaml

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, Rectangle

from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make


class CartPoleRLLab:
    """Interactive PPO CartPole laboratory for safe-control-gym.

    Design principle:
      * TRAIN changes the learned policy (reward + PPO settings).
      * EVALUATE freezes a selected policy and changes test conditions only.
    """

    TARGET_X = 0.0
    FORCE_SCALE_N = 10.0  # CartPole normalized action a -> 10*a N in the repo.

    def __init__(self, root):
        self.root = root
        self.root.title("RL CartPole Lab — PPO")
        self.root.geometry("1550x950")
        self.root.minsize(1250, 800)

        # ------------------------------------------------------------
        # Base configuration from command-line YAML files.
        # ------------------------------------------------------------
        fac = ConfigFactory()
        self.base_config = fac.merge()

        if getattr(self.base_config, "algo", None) != "ppo":
            messagebox.showwarning(
                "Algorithm",
                "This lab is designed for PPO. Start it with --algo ppo."
            )

        # Paths.
        self.base_dir = Path(__file__).resolve().parent
        self.repo_root = self.base_dir.parent.parent
        self.lab_runs_dir = self.base_dir / "lab_runs"
        self.lab_configs_dir = self.base_dir / "lab_configs"
        self.temp_eval_dir = self.base_dir / "temp_rl_lab_eval"
        self.lab_runs_dir.mkdir(exist_ok=True)
        self.lab_configs_dir.mkdir(exist_ok=True)

        # Runtime objects.
        self.running = False
        self.env = None
        self.ctrl = None
        self.obs = None
        self.obs_ctrl = None
        self.info = None
        self.after_id = None
        self.step_count = 0
        self.max_steps = 0
        self.ctrl_freq = 15.0
        self.dt = 1.0 / self.ctrl_freq

        # Evaluation disturbance/delay state.
        self.delay_buffer = deque()
        self.eval_noise_scale = np.zeros(4)
        self.eval_force_limit = 10.0
        self.eval_input_disturbance = 0.0
        self.eval_action_delay = 0
        self.eval_action_mode = "Deterministic"
        self.playback_speed = 1.0
        self.target_x = self.TARGET_X

        # Training process state.
        self.training_process = None
        self.training_thread = None
        self.training_queue = queue.Queue()
        self.pending_run_dir = None
        self.training_settings_dirty = False
        self._applying_reward_preset = False

        # Histories.
        self.clear_history_arrays()
        self.current_result = None
        self.result_A = None
        self.result_B = None
        self.comparison_artists = []

        # Get training range from current task config.
        self.training_limits = self.get_training_limits(
            copy.deepcopy(dict(self.base_config.task_config))
        )

        # Main layout.
        self.left_frame = ttk.Frame(root, padding=10)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y)

        self.right_frame = ttk.Frame(root, padding=8)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.build_left_panel()
        self.build_plot_panel()
        self.refresh_models()
        self.update_theory()
        self.update_reward_equation_label()
        self.update_gae_state()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(120, self.poll_training_queue)

    # ==================================================================
    # Basic helpers
    # ==================================================================

    def clear_history_arrays(self):
        self.time_history = []
        self.x_history = []
        self.xdot_history = []
        self.theta_history = []          # degrees for display/metrics
        self.thetadot_history = []       # rad/s
        self.force_history = []          # applied physical force [N]
        self.reward_history = []
        self.action_history = []         # raw PPO normalized action
        self.command_force_history = []  # delayed command + disturbance before clipping

    def deep_dict(self, value):
        return copy.deepcopy(dict(value))

    def get_training_limits(self, task_config):
        defaults = {
            "x": (-2.0, 2.0),
            "xdot": (-2.0, 2.0),
            "theta": (-0.16, 0.16),
            "thetadot": (-1.0, 1.0),
        }
        try:
            info = task_config["init_state_randomization_info"]
            return {
                "x": (float(info["init_x"]["low"]), float(info["init_x"]["high"])),
                "xdot": (float(info["init_x_dot"]["low"]), float(info["init_x_dot"]["high"])),
                "theta": (float(info["init_theta"]["low"]), float(info["init_theta"]["high"])),
                "thetadot": (float(info["init_theta_dot"]["low"]), float(info["init_theta_dot"]["high"])),
            }
        except Exception:
            return defaults

    def training_half_ranges(self):
        lim = self.training_limits
        return np.array([
            max(abs(lim["x"][0]), abs(lim["x"][1])),
            max(abs(lim["xdot"][0]), abs(lim["xdot"][1])),
            max(abs(lim["theta"][0]), abs(lim["theta"][1])),
            max(abs(lim["thetadot"][0]), abs(lim["thetadot"][1])),
        ], dtype=float)

    # ==================================================================
    # Left panel / tabs
    # ==================================================================

    def build_left_panel(self):
        ttk.Label(
            self.left_frame,
            text="RL CARTPOLE LAB",
            font=("Arial", 18, "bold")
        ).pack(anchor="w", pady=(0, 3))

        ttk.Label(
            self.left_frame,
            text="PPO · Continuous control",
            font=("Arial", 10)
        ).pack(anchor="w", pady=(0, 10))

        self.notebook = ttk.Notebook(self.left_frame, width=390, height=590)
        self.notebook.pack(fill=tk.BOTH, expand=False)

        self.tab_experiment = ttk.Frame(self.notebook, padding=12)
        self.tab_reward = ttk.Frame(self.notebook, padding=12)
        self.tab_ppo = ttk.Frame(self.notebook, padding=12)
        self.tab_theory = ttk.Frame(self.notebook, padding=12)
        self.tab_results = ttk.Frame(self.notebook, padding=12)

        self.notebook.add(self.tab_experiment, text="Experiment")
        self.notebook.add(self.tab_reward, text="Reward")
        self.notebook.add(self.tab_ppo, text="PPO")
        self.notebook.add(self.tab_theory, text="Theory")
        self.notebook.add(self.tab_results, text="Results")

        self.build_experiment_tab()
        self.build_reward_tab()
        self.build_ppo_tab()
        self.build_theory_tab()
        self.build_results_tab()

        # --------------------------------------------------------------
        # Model selection + main actions.
        # --------------------------------------------------------------
        ttk.Separator(self.left_frame).pack(fill=tk.X, pady=8)

        ttk.Label(
            self.left_frame,
            text="Selected policy model",
            font=("Arial", 10, "bold")
        ).pack(anchor="w")

        model_row = ttk.Frame(self.left_frame)
        model_row.pack(fill=tk.X, pady=(4, 6))

        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(
            model_row,
            textvariable=self.model_var,
            width=43,
            state="normal"
        )
        self.model_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)

        ttk.Button(
            model_row,
            text="Browse",
            command=self.browse_model
        ).pack(side=tk.LEFT, padx=(5, 0))

        ttk.Button(
            model_row,
            text="↻",
            width=3,
            command=self.refresh_models
        ).pack(side=tk.LEFT, padx=(4, 0))

        button_row = ttk.Frame(self.left_frame)
        button_row.pack(fill=tk.X, pady=(5, 3))

        self.train_button = ttk.Button(
            button_row,
            text="TRAIN NEW POLICY",
            command=self.train_new_policy
        )
        self.train_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        self.eval_button = ttk.Button(
            button_row,
            text="EVALUATE POLICY",
            command=self.start_evaluation
        )
        self.eval_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))

        secondary_row = ttk.Frame(self.left_frame)
        secondary_row.pack(fill=tk.X, pady=(3, 4))

        ttk.Button(
            secondary_row,
            text="STOP EVALUATION",
            command=self.stop_evaluation
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        ttk.Button(
            secondary_row,
            text="CANCEL TRAIN",
            command=self.cancel_training
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))

        self.train_progress = ttk.Progressbar(
            self.left_frame,
            mode="indeterminate"
        )
        self.train_progress.pack(fill=tk.X, pady=(3, 3))

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(
            self.left_frame,
            textvariable=self.status_var,
            font=("Arial", 10, "bold")
        ).pack(anchor="w")

        self.retrain_var = tk.StringVar(value="")
        self.retrain_label = tk.Label(
            self.left_frame,
            textvariable=self.retrain_var,
            justify="left",
            anchor="w",
            fg="darkorange"
        )
        self.retrain_label.pack(anchor="w", pady=(2, 0))

    # ==================================================================
    # Experiment tab
    # ==================================================================

    def build_experiment_tab(self):
        ttk.Label(
            self.tab_experiment,
            text="Evaluation initial state",
            font=("Arial", 12, "bold")
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))

        self.x_var = tk.StringVar(value="-1.0")
        self.xdot_var = tk.StringVar(value="0.0")
        self.theta_var = tk.StringVar(value="5.0")
        self.thetadot_var = tk.StringVar(value="0.0")
        self.episode_var = tk.StringVar(value="10.0")
        self.playback_var = tk.StringVar(value="1.0")

        rows = [
            ("x₀ [m]", self.x_var),
            ("ẋ₀ [m/s]", self.xdot_var),
            ("θ₀ [deg]", self.theta_var),
            ("θ̇₀ [rad/s]", self.thetadot_var),
            ("Episode [s]", self.episode_var),
            ("Playback", self.playback_var),
        ]

        for i, (label, var) in enumerate(rows, start=1):
            ttk.Label(self.tab_experiment, text=label).grid(
                row=i, column=0, sticky="w", pady=4
            )
            ttk.Entry(self.tab_experiment, textvariable=var, width=13).grid(
                row=i, column=1, sticky="e", pady=4
            )

        ttk.Separator(self.tab_experiment).grid(
            row=7, column=0, columnspan=2, sticky="ew", pady=10
        )

        ttk.Label(
            self.tab_experiment,
            text="Robustness tests (evaluation only)",
            font=("Arial", 11, "bold")
        ).grid(row=8, column=0, columnspan=2, sticky="w", pady=(0, 6))

        self.force_limit_var = tk.StringVar(value="10.0")
        self.obs_noise_pct_var = tk.StringVar(value="0.0")
        self.input_disturb_var = tk.StringVar(value="0.0")
        self.action_delay_var = tk.StringVar(value="0")
        self.action_mode_var = tk.StringVar(value="Deterministic")

        robust_rows = [
            ("Force limit [N]", self.force_limit_var),
            ("Obs noise [% range]", self.obs_noise_pct_var),
            ("Input disturbance [N]", self.input_disturb_var),
            ("Action delay [steps]", self.action_delay_var),
        ]

        for i, (label, var) in enumerate(robust_rows, start=9):
            ttk.Label(self.tab_experiment, text=label).grid(
                row=i, column=0, sticky="w", pady=4
            )
            ttk.Entry(self.tab_experiment, textvariable=var, width=13).grid(
                row=i, column=1, sticky="e", pady=4
            )

        ttk.Label(self.tab_experiment, text="Action mode").grid(
            row=13, column=0, sticky="w", pady=4
        )
        ttk.Combobox(
            self.tab_experiment,
            textvariable=self.action_mode_var,
            values=["Deterministic", "Stochastic"],
            state="readonly",
            width=14
        ).grid(row=13, column=1, sticky="e", pady=4)

        ttk.Separator(self.tab_experiment).grid(
            row=14, column=0, columnspan=2, sticky="ew", pady=10
        )

        self.range_var = tk.StringVar(value="INITIAL STATE: waiting")
        self.range_label = tk.Label(
            self.tab_experiment,
            textvariable=self.range_var,
            font=("Arial", 10, "bold"),
            fg="gray"
        )
        self.range_label.grid(row=15, column=0, columnspan=2, sticky="w")

        self.range_detail_var = tk.StringVar(value="")
        self.range_detail_label = tk.Label(
            self.tab_experiment,
            textvariable=self.range_detail_var,
            justify="left",
            anchor="w",
            fg="gray"
        )
        self.range_detail_label.grid(
            row=16, column=0, columnspan=2, sticky="w", pady=(3, 0)
        )

        lim = self.training_limits
        range_text = (
            "Training initial-state range:\n"
            f"x: [{lim['x'][0]:.1f}, {lim['x'][1]:.1f}] m\n"
            f"ẋ: [{lim['xdot'][0]:.1f}, {lim['xdot'][1]:.1f}] m/s\n"
            f"θ: [{math.degrees(lim['theta'][0]):.1f}, {math.degrees(lim['theta'][1]):.1f}] deg\n"
            f"θ̇: [{lim['thetadot'][0]:.1f}, {lim['thetadot'][1]:.1f}] rad/s"
        )
        ttk.Label(
            self.tab_experiment,
            text=range_text,
            justify="left",
            font=("Arial", 9)
        ).grid(row=17, column=0, columnspan=2, sticky="w", pady=(10, 0))

    # ==================================================================
    # Reward tab
    # ==================================================================

    def build_reward_tab(self):
        ttk.Label(
            self.tab_reward,
            text="Reward / cost design",
            font=("Arial", 12, "bold")
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))

        ttk.Label(
            self.tab_reward,
            text="Changing these values requires retraining.",
            font=("Arial", 9)
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 8))

        task_cfg = self.deep_dict(self.base_config.task_config)
        q_default = list(task_cfg.get("rew_state_weight", [1, 1, 1, 1]))
        r_default = float(task_cfg.get("rew_act_weight", 0.1))
        exp_default = bool(task_cfg.get("rew_exponential", True))

        self.reward_preset_var = tk.StringVar(value="Repository default")
        ttk.Label(self.tab_reward, text="Preset").grid(row=2, column=0, sticky="w", pady=4)
        preset_combo = ttk.Combobox(
            self.tab_reward,
            textvariable=self.reward_preset_var,
            values=[
                "Repository default",
                "Angle priority",
                "Position priority",
                "Low control effort",
                "Custom",
            ],
            state="readonly",
            width=18
        )
        preset_combo.grid(row=2, column=1, sticky="e", pady=4)
        preset_combo.bind("<<ComboboxSelected>>", self.apply_reward_preset)

        self.qx_var = tk.StringVar(value=str(q_default[0]))
        self.qxdot_var = tk.StringVar(value=str(q_default[1]))
        self.qtheta_var = tk.StringVar(value=str(q_default[2]))
        self.qthetadot_var = tk.StringVar(value=str(q_default[3]))
        self.ru_var = tk.StringVar(value=str(r_default))

        reward_rows = [
            ("qₓ  position", self.qx_var),
            ("qₓdot velocity", self.qxdot_var),
            ("qθ  angle", self.qtheta_var),
            ("qθdot angular vel.", self.qthetadot_var),
            ("Rᵤ  control effort", self.ru_var),
        ]

        for i, (label, var) in enumerate(reward_rows, start=3):
            ttk.Label(self.tab_reward, text=label).grid(
                row=i, column=0, sticky="w", pady=4
            )
            ttk.Entry(self.tab_reward, textvariable=var, width=13).grid(
                row=i, column=1, sticky="e", pady=4
            )

        self.reward_type_var = tk.StringVar(
            value="Exponential bounded" if exp_default else "Negative quadratic"
        )
        ttk.Label(self.tab_reward, text="Reward type").grid(
            row=8, column=0, sticky="w", pady=4
        )
        reward_type_combo = ttk.Combobox(
            self.tab_reward,
            textvariable=self.reward_type_var,
            values=["Exponential bounded", "Negative quadratic"],
            state="readonly",
            width=18
        )
        reward_type_combo.grid(row=8, column=1, sticky="e", pady=4)
        reward_type_combo.bind("<<ComboboxSelected>>", lambda _e: self.on_training_setting_changed())
        reward_type_combo.bind("<<ComboboxSelected>>", lambda _e: self.update_reward_equation_label(), add="+")

        ttk.Separator(self.tab_reward).grid(
            row=9, column=0, columnspan=2, sticky="ew", pady=10
        )

        self.reward_equation_var = tk.StringVar()
        tk.Label(
            self.tab_reward,
            textvariable=self.reward_equation_var,
            justify="left",
            anchor="w",
            font=("Menlo", 9),
            wraplength=340
        ).grid(row=10, column=0, columnspan=2, sticky="w")

        # Mark edits as requiring retraining.
        for var in [
            self.qx_var, self.qxdot_var, self.qtheta_var,
            self.qthetadot_var, self.ru_var
        ]:
            var.trace_add("write", lambda *_args: self.on_reward_entry_changed())

    def apply_reward_preset(self, _event=None):
        preset = self.reward_preset_var.get()
        self._applying_reward_preset = True
        presets = {
            "Repository default": ([1.0, 1.0, 1.0, 1.0], 0.1),
            "Angle priority": ([1.0, 0.5, 5.0, 1.0], 0.1),
            "Position priority": ([5.0, 1.0, 1.0, 0.5], 0.1),
            "Low control effort": ([1.0, 1.0, 3.0, 1.0], 0.8),
        }
        if preset in presets:
            q, r = presets[preset]
            self.qx_var.set(str(q[0]))
            self.qxdot_var.set(str(q[1]))
            self.qtheta_var.set(str(q[2]))
            self.qthetadot_var.set(str(q[3]))
            self.ru_var.set(str(r))
        self._applying_reward_preset = False
        self.on_training_setting_changed()
        self.update_reward_equation_label()

    def on_reward_entry_changed(self):
        if hasattr(self, "reward_preset_var") and not self._applying_reward_preset:
            self.reward_preset_var.set("Custom")
        self.on_training_setting_changed()
        self.update_reward_equation_label()

    def update_reward_equation_label(self):
        if not hasattr(self, "reward_equation_var"):
            return
        try:
            qx, qv, qt, qw, ru = self.read_reward_values()
            j_text = (
                "J = "
                f"{qx:g}(x-xref)² + {qv:g}ẋ² + "
                f"{qt:g}θ² + {qw:g}θ̇² + {ru:g}u²"
            )
            if self.reward_type_var.get() == "Exponential bounded":
                r_text = "r = exp(-J),   0 < r ≤ 1"
            else:
                r_text = "r = -J"
            self.reward_equation_var.set(j_text + "\n" + r_text)
        except Exception:
            self.reward_equation_var.set("Enter valid numeric reward weights.")

    # ==================================================================
    # PPO tab
    # ==================================================================

    def build_ppo_tab(self):
        ttk.Label(
            self.tab_ppo,
            text="PPO learning parameters",
            font=("Arial", 12, "bold")
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))

        ttk.Label(
            self.tab_ppo,
            text="Changing these values requires retraining.",
            font=("Arial", 9)
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 8))

        cfg = self.deep_dict(self.base_config.algo_config)

        self.gamma_var = tk.StringVar(value=str(cfg.get("gamma", 0.98)))
        self.use_gae_var = tk.BooleanVar(value=bool(cfg.get("use_gae", False)))
        self.gae_lambda_var = tk.StringVar(value=str(cfg.get("gae_lambda", 0.8)))
        self.clip_var = tk.StringVar(value=str(cfg.get("clip_param", 0.1)))
        self.actor_lr_var = tk.StringVar(value=str(cfg.get("actor_lr", 7.948e-4)))
        self.critic_lr_var = tk.StringVar(value=str(cfg.get("critic_lr", 7.497e-3)))
        self.entropy_var = tk.StringVar(value=str(cfg.get("entropy_coef", 1.075e-4)))

        ppo_rows = [
            ("Discount γ", self.gamma_var),
            ("GAE λ", self.gae_lambda_var),
            ("PPO clip ε", self.clip_var),
            ("Actor LR", self.actor_lr_var),
            ("Critic LR", self.critic_lr_var),
            ("Entropy coeff.", self.entropy_var),
        ]

        row = 2
        ttk.Label(self.tab_ppo, text=ppo_rows[0][0]).grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(self.tab_ppo, textvariable=ppo_rows[0][1], width=14).grid(row=row, column=1, sticky="e", pady=4)
        row += 1

        self.gae_check = ttk.Checkbutton(
            self.tab_ppo,
            text="Use GAE",
            variable=self.use_gae_var,
            command=self.on_gae_toggle
        )
        self.gae_check.grid(row=row, column=0, columnspan=2, sticky="w", pady=4)
        row += 1

        ttk.Label(self.tab_ppo, text="GAE λ").grid(row=row, column=0, sticky="w", pady=4)
        self.gae_entry = ttk.Entry(self.tab_ppo, textvariable=self.gae_lambda_var, width=14)
        self.gae_entry.grid(row=row, column=1, sticky="e", pady=4)
        row += 1

        for label, var in ppo_rows[2:]:
            ttk.Label(self.tab_ppo, text=label).grid(row=row, column=0, sticky="w", pady=4)
            ttk.Entry(self.tab_ppo, textvariable=var, width=14).grid(row=row, column=1, sticky="e", pady=4)
            row += 1

        ttk.Separator(self.tab_ppo).grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=10
        )
        row += 1

        ttk.Label(
            self.tab_ppo,
            text="Training budget: 300,000 env steps (fixed)",
            font=("Arial", 9)
        ).grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1

        ttk.Button(
            self.tab_ppo,
            text="Reset PPO defaults",
            command=self.reset_ppo_defaults
        ).grid(row=row, column=0, columnspan=2, sticky="ew", pady=(10, 0))

        for var in [
            self.gamma_var, self.gae_lambda_var, self.clip_var,
            self.actor_lr_var, self.critic_lr_var, self.entropy_var
        ]:
            var.trace_add("write", lambda *_args: self.on_training_setting_changed())

    def on_gae_toggle(self):
        self.update_gae_state()
        self.on_training_setting_changed()

    def update_gae_state(self):
        if not hasattr(self, "gae_entry"):
            return
        self.gae_entry.configure(state="normal" if self.use_gae_var.get() else "disabled")

    def reset_ppo_defaults(self):
        cfg = self.deep_dict(self.base_config.algo_config)
        self.gamma_var.set(str(cfg.get("gamma", 0.98)))
        self.use_gae_var.set(bool(cfg.get("use_gae", False)))
        self.gae_lambda_var.set(str(cfg.get("gae_lambda", 0.8)))
        self.clip_var.set(str(cfg.get("clip_param", 0.1)))
        self.actor_lr_var.set(str(cfg.get("actor_lr", 7.948e-4)))
        self.critic_lr_var.set(str(cfg.get("critic_lr", 7.497e-3)))
        self.entropy_var.set(str(cfg.get("entropy_coef", 1.075e-4)))
        self.update_gae_state()
        self.on_training_setting_changed()

    def on_training_setting_changed(self):
        self.training_settings_dirty = True
        if hasattr(self, "retrain_var"):
            self.retrain_var.set("Training settings changed → TRAIN NEW POLICY to test their effect.")

    # ==================================================================
    # Theory tab
    # ==================================================================

    def build_theory_tab(self):

        ttk.Label(
            self.tab_theory,
            text="Theory / Equations",
            font=("Arial", 13, "bold")
        ).pack(
            anchor="w",
            pady=(0, 8)
        )

        self.theory_choice_var = tk.StringVar(
            value="Agent–Environment"
        )

        theory_combo = ttk.Combobox(
            self.tab_theory,
            textvariable=self.theory_choice_var,
            state="readonly",
            values=[
                "Agent–Environment",
                "Policy",
                "Return / Discount γ",
                "Reward / Cost",
                "Value Function",
                "TD Error",
                "Advantage",
                "GAE",
                "PPO Clipped Objective",
            ],
            width=28
        )

        theory_combo.pack(
            fill=tk.X,
            pady=(0, 10)
        )

        theory_combo.bind(
            "<<ComboboxSelected>>",
            lambda _e: self.update_theory()
        )

        # --------------------------------------------------
        # Matplotlib canvas for LaTeX-style equations
        # --------------------------------------------------

        self.theory_fig = Figure(
            figsize=(5.2, 6.8),
            dpi=100
        )

        self.theory_ax = self.theory_fig.add_subplot(111)
        self.theory_ax.axis("off")

        self.theory_canvas = FigureCanvasTkAgg(
            self.theory_fig,
            master=self.tab_theory
        )

        self.theory_canvas.get_tk_widget().pack(
            fill=tk.BOTH,
            expand=True
        )

        self.update_theory()

    def update_theory(self):

        if not hasattr(self, "theory_ax"):
            return

        choice = self.theory_choice_var.get()

        theory_data = {

            "Agent–Environment": {
                "title": "Agent–Environment Interaction",
                "equations": [
                    r"$\mathbf{s}_t=[x_t,\ \dot{x}_t,\ \theta_t,\ \dot{\theta}_t]^T$",
                    r"$\mathbf{s}_t\;\longrightarrow\;\pi_\theta\;\longrightarrow\;a_t$",
                    r"$a_t\;\longrightarrow\;\mathrm{CartPole}\;\longrightarrow\;"
                    r"(\mathbf{s}_{t+1},\ r_{t+1})$"
                ],
                "body":
                    "The PPO agent observes the CartPole state and generates a "
                    "continuous action. The environment returns the next state "
                    "and reward.\n\n"
                    "PPO does not use the CartPole differential equations for "
                    "planning. It learns from sampled interaction, so this is "
                    "model-free reinforcement learning."
            },

            "Policy": {
                "title": "Policy",
                "equations": [
                    r"$a_t \sim \pi_\theta(a\,|\,\mathbf{s}_t)$",
                    r"$\pi_\theta(a\,|\,s)=\mathcal{N}"
                    r"\left(\mu_\theta(s),\sigma_\theta^2(s)\right)$"
                ],
                "body":
                    "The actor represents the policy. During training, the "
                    "continuous PPO actor defines an action distribution.\n\n"
                    "During deterministic evaluation, the mean/mode of the "
                    "learned distribution is used instead of randomly sampling "
                    "an action."
            },

            "Return / Discount γ": {
                "title": "Discounted Return",
                "equations": [
                    r"$G_t=\sum_{k=0}^{\infty}\gamma^k r_{t+k+1}$",
                    r"$0\leq\gamma\leq1$"
                ],
                "body":
                    "The discount factor γ determines how strongly future "
                    "rewards affect the current learning objective.\n\n"
                    "Higher γ → more importance to long-term rewards.\n"
                    "Lower γ → more short-term behavior."
            },

            "Reward / Cost": {
                "title": "CartPole Reward / Cost",
                "equations": [
                    r"$J_t=q_x(x_t-x_{\mathrm{ref}})^2"
                    r"+q_{\dot{x}}\dot{x}_t^2$",
                    r"$\qquad+q_\theta(\theta_t-\theta_{\mathrm{ref}})^2"
                    r"+q_{\dot{\theta}}\dot{\theta}_t^2+R_u u_t^2$",
                    r"$r_t=e^{-J_t}\qquad\mathrm{(Exponential)}$",
                    r"$r_t=-J_t\qquad\mathrm{(Negative\ Quadratic)}$"
                ],
                "body":
                    "The q weights specify which state errors are more important, "
                    "while Rᵤ penalizes control effort.\n\n"
                    "For example, increasing qθ makes pole-angle error more costly. "
                    "Increasing Rᵤ encourages the policy to use less control effort.\n\n"
                    "Changing these values changes the learning objective, so the "
                    "PPO policy must be retrained."
            },

            "Value Function": {
                "title": "State-Value Function",
                "equations": [
                    r"$V^\pi(s)=\mathbb{E}_\pi\left[G_t\,|\,S_t=s\right]$"
                ],
                "body":
                    "The value function estimates the expected future return "
                    "starting from a state while following policy π.\n\n"
                    "PPO is an Actor–Critic method. The actor represents the "
                    "policy and the critic approximates V(s)."
            },

            "TD Error": {
                "title": "Temporal-Difference Error",
                "equations": [
                    r"$\delta_t=r_{t+1}+\gamma V(s_{t+1})-V(s_t)$"
                ],
                "body":
                    "The TD error compares the current value estimate with a "
                    "one-step bootstrapped target.\n\n"
                    "Positive δ means the transition was better than the critic "
                    "expected; negative δ means it was worse."
            },

            "Advantage": {
                "title": "Advantage Function",
                "equations": [
                    r"$A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)$"
                ],
                "body":
                    "Advantage measures whether an action is better or worse than "
                    "the expected behavior at the same state.\n\n"
                    "A > 0 → reinforce the action.\n"
                    "A < 0 → reduce its probability.\n\n"
                    "PPO uses estimated advantages when updating its policy."
            },

            "GAE": {
                "title": "Generalized Advantage Estimation",
                "equations": [
                    r"$\hat{A}_t^{GAE}=\sum_{l=0}^{\infty}"
                    r"(\gamma\lambda)^l\delta_{t+l}$",
                    r"$=\delta_t+\gamma\lambda\delta_{t+1}"
                    r"+(\gamma\lambda)^2\delta_{t+2}+\cdots$"
                ],
                "body":
                    "GAE combines TD errors over multiple future steps.\n\n"
                    "λ controls the bias–variance trade-off:\n"
                    "Low λ → more bias, lower variance.\n"
                    "High λ → less bias, higher variance.\n\n"
                    "In the repository default CartPole PPO configuration, GAE "
                    "is disabled unless Use GAE is selected."
            },

            "PPO Clipped Objective": {
                "title": "PPO Clipped Objective",
                "equations": [
                    r"$\rho_t(\theta)=\frac{\pi_\theta(a_t|s_t)}"
                    r"{\pi_{\theta_{\mathrm{old}}}(a_t|s_t)}$",
                    r"$L^{CLIP}(\theta)=\mathbb{E}_t\left["
                    r"\min\left(\rho_t\hat{A}_t,"
                    r"\mathrm{clip}(\rho_t,1-\epsilon,1+\epsilon)"
                    r"\hat{A}_t\right)\right]$"
                ],
                "body":
                    "ρ measures how much the new policy differs from the policy "
                    "that generated the training data.\n\n"
                    "The clipping parameter ε limits excessively large policy "
                    "updates. Smaller ε gives more conservative updates; larger "
                    "ε permits larger changes."
            },
        }

        data = theory_data.get(
            choice,
            theory_data["Agent–Environment"]
        )

        self.theory_ax.clear()
        self.theory_ax.axis("off")

        self.theory_ax.text(
            0.5,
            0.96,
            data["title"],
            transform=self.theory_ax.transAxes,
            ha="center",
            va="top",
            fontsize=16,
            fontweight="bold"
        )

        y = 0.83

        for equation in data["equations"]:
            self.theory_ax.text(
                0.5,
                y,
                equation,
                transform=self.theory_ax.transAxes,
                ha="center",
                va="center",
                fontsize=14
            )
            y -= 0.105

        explanation_y = min(
            y - 0.03,
            0.50
        )

        self.theory_ax.text(
            0.04,
            explanation_y,
            data["body"],
            transform=self.theory_ax.transAxes,
            ha="left",
            va="top",
            fontsize=10.5,
            wrap=True,
            linespacing=1.5
        )

        self.theory_canvas.draw_idle()

    # ==================================================================
    # Results tab
    # ==================================================================

    def build_results_tab(self):
        ttk.Label(
            self.tab_results,
            text="Evaluation results",
            font=("Arial", 12, "bold")
        ).pack(anchor="w", pady=(0, 6))

        self.results_text = tk.Text(
            self.tab_results,
            width=43,
            height=18,
            wrap="none",
            font=("Menlo", 9)
        )
        self.results_text.pack(fill=tk.BOTH, expand=False)
        self.results_text.insert(tk.END, "Run EVALUATE POLICY to generate metrics.\n")
        self.results_text.configure(state="disabled")

        store_row = ttk.Frame(self.tab_results)
        store_row.pack(fill=tk.X, pady=(8, 4))

        ttk.Button(
            store_row,
            text="Store current as A",
            command=lambda: self.store_result("A")
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 3))

        ttk.Button(
            store_row,
            text="Store current as B",
            command=lambda: self.store_result("B")
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(3, 0))

        ttk.Button(
            self.tab_results,
            text="COMPARE A vs B",
            command=self.compare_results
        ).pack(fill=tk.X, pady=(2, 6))

        self.compare_text = tk.Text(
            self.tab_results,
            width=43,
            height=12,
            wrap="none",
            font=("Menlo", 9)
        )
        self.compare_text.pack(fill=tk.BOTH, expand=True)
        self.compare_text.insert(tk.END, "Store two evaluations as A and B.\n")
        self.compare_text.configure(state="disabled")

        ttk.Separator(self.tab_results).pack(fill=tk.X, pady=8)

        ttk.Label(
            self.tab_results,
            text="Training log",
            font=("Arial", 10, "bold")
        ).pack(anchor="w")

        self.train_log = tk.Text(
            self.tab_results,
            width=43,
            height=7,
            wrap="word",
            font=("Menlo", 8)
        )
        self.train_log.pack(fill=tk.BOTH, expand=True, pady=(3, 0))

    # ==================================================================
    # Plot panel
    # ==================================================================

    def build_plot_panel(self):
        self.fig = Figure(figsize=(11, 8), dpi=100)
        gs = self.fig.add_gridspec(
            3, 2,
            height_ratios=[2.45, 1.0, 1.0],
            hspace=0.48,
            wspace=0.28
        )

        # Main CartPole animation.
        self.ax = self.fig.add_subplot(gs[0, :])
        self.ax.set_xlim(-3.0, 3.0)
        self.ax.set_ylim(-0.3, 1.55)
        self.ax.set_xlabel("Cart position x [m]")
        self.ax.set_ylabel("Height")
        self.ax.set_title("PPO Controlled CartPole")
        self.ax.grid(True, alpha=0.2)
        self.rail_line, = self.ax.plot([-20, 20], [0, 0], linewidth=3)
        self.target_line = self.ax.axvline(
            self.target_x,
            linestyle="--",
            linewidth=1.5,
            label=f"x ref = {self.target_x:.2f} m"
        )
        self.ax.legend(loc="upper right")

        self.cart_width = 0.45
        self.cart_height = 0.20
        self.cart_y = 0.05
        self.pivot_y = self.cart_y + self.cart_height
        self.pole_length_draw = 1.0

        self.cart = Rectangle(
            (-self.cart_width / 2, self.cart_y),
            self.cart_width,
            self.cart_height
        )
        self.ax.add_patch(self.cart)

        self.pole_line, = self.ax.plot(
            [0, 0],
            [self.pivot_y, self.pivot_y + self.pole_length_draw],
            linewidth=7,
            solid_capstyle="round"
        )
        self.pivot_point, = self.ax.plot(
            [0], [self.pivot_y], marker="o", markersize=10
        )
        self.force_arrow = None

        self.state_text = self.ax.text(
            0.02, 0.96, "",
            transform=self.ax.transAxes,
            verticalalignment="top",
            family="monospace",
            fontsize=9.5
        )

        # x(t).
        self.ax_x = self.fig.add_subplot(gs[1, 0])
        self.ax_x.set_title("Cart position")
        self.ax_x.set_xlabel("Time [s]")
        self.ax_x.set_ylabel("x [m]")
        self.ax_x.grid(True, alpha=0.3)
        self.x_line, = self.ax_x.plot([], [], label="Current")
        self.x_ref_line = self.ax_x.axhline(
            self.target_x, linestyle="--", linewidth=1, label="Reference"
        )
        self.ax_x.legend(fontsize=8)

        # theta(t).
        self.ax_theta = self.fig.add_subplot(gs[1, 1])
        self.ax_theta.set_title("Pole angle")
        self.ax_theta.set_xlabel("Time [s]")
        self.ax_theta.set_ylabel("θ [deg]")
        self.ax_theta.grid(True, alpha=0.3)
        self.theta_line, = self.ax_theta.plot([], [], label="Current")
        self.ax_theta.axhline(0.0, linestyle="--", linewidth=1)

        # Force(t).
        self.ax_force = self.fig.add_subplot(gs[2, 0])
        self.ax_force.set_title("Applied physical force")
        self.ax_force.set_xlabel("Time [s]")
        self.ax_force.set_ylabel("F [N]")
        self.ax_force.grid(True, alpha=0.3)
        self.force_line, = self.ax_force.plot([], [], label="Applied force")
        self.ax_force.axhline(0.0, linewidth=1)
        self.force_limit_plus = self.ax_force.axhline(
            10.0, linestyle="--", linewidth=1.3, label="Force limit"
        )
        self.force_limit_minus = self.ax_force.axhline(
            -10.0, linestyle="--", linewidth=1.3
        )
        self.ax_force.set_ylim(-12, 12)
        self.ax_force.legend(fontsize=8, loc="upper right")

        # Reward(t).
        self.ax_reward = self.fig.add_subplot(gs[2, 1])
        self.ax_reward.set_title("RL reward")
        self.ax_reward.set_xlabel("Time [s]")
        self.ax_reward.set_ylabel("r")
        self.ax_reward.grid(True, alpha=0.3)
        self.reward_line, = self.ax_reward.plot([], [], label="Reward")

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ==================================================================
    # Read/validate settings
    # ==================================================================

    def read_reward_values(self):
        vals = [
            float(self.qx_var.get()),
            float(self.qxdot_var.get()),
            float(self.qtheta_var.get()),
            float(self.qthetadot_var.get()),
            float(self.ru_var.get()),
        ]
        if any(v < 0 for v in vals):
            raise ValueError("Reward/cost weights must be non-negative.")
        return vals

    def read_ppo_values(self):
        gamma = float(self.gamma_var.get())
        gae_lambda = float(self.gae_lambda_var.get())
        clip = float(self.clip_var.get())
        actor_lr = float(self.actor_lr_var.get())
        critic_lr = float(self.critic_lr_var.get())
        entropy = float(self.entropy_var.get())

        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1].")
        if not 0.0 <= gae_lambda <= 1.0:
            raise ValueError("GAE lambda must be in [0, 1].")
        if clip <= 0:
            raise ValueError("PPO clip epsilon must be positive.")
        if actor_lr <= 0 or critic_lr <= 0:
            raise ValueError("Learning rates must be positive.")
        if entropy < 0:
            raise ValueError("Entropy coefficient must be non-negative.")

        return {
            "gamma": gamma,
            "use_gae": bool(self.use_gae_var.get()),
            "gae_lambda": gae_lambda,
            "clip_param": clip,
            "actor_lr": actor_lr,
            "critic_lr": critic_lr,
            "entropy_coef": entropy,
        }

    def read_evaluation_values(self):
        x0 = float(self.x_var.get())
        xdot0 = float(self.xdot_var.get())
        theta_deg = float(self.theta_var.get())
        thetadot0 = float(self.thetadot_var.get())
        episode = float(self.episode_var.get())
        playback = float(self.playback_var.get())
        force_limit = float(self.force_limit_var.get())
        noise_pct = float(self.obs_noise_pct_var.get())
        disturbance = float(self.input_disturb_var.get())
        delay = int(self.action_delay_var.get())

        if episode <= 0:
            raise ValueError("Episode time must be positive.")
        if playback <= 0:
            raise ValueError("Playback must be positive.")
        if force_limit <= 0:
            raise ValueError("Force limit must be positive.")
        if noise_pct < 0:
            raise ValueError("Observation noise percentage cannot be negative.")
        if delay < 0:
            raise ValueError("Action delay cannot be negative.")

        return {
            "x0": x0,
            "xdot0": xdot0,
            "theta_deg": theta_deg,
            "thetadot0": thetadot0,
            "episode": episode,
            "playback": playback,
            "force_limit": force_limit,
            "noise_pct": noise_pct,
            "disturbance": disturbance,
            "delay": delay,
            "action_mode": self.action_mode_var.get(),
        }

    # ==================================================================
    # Training-range warning
    # ==================================================================

    def check_initial_training_range(self, vals):
        lim = self.training_limits
        theta_low = math.degrees(lim["theta"][0])
        theta_high = math.degrees(lim["theta"][1])
        violations = []

        if not lim["x"][0] <= vals["x0"] <= lim["x"][1]:
            violations.append(f"x₀={vals['x0']:.2f} m")
        if not lim["xdot"][0] <= vals["xdot0"] <= lim["xdot"][1]:
            violations.append(f"ẋ₀={vals['xdot0']:.2f} m/s")
        if not theta_low <= vals["theta_deg"] <= theta_high:
            violations.append(f"θ₀={vals['theta_deg']:.2f}°")
        if not lim["thetadot"][0] <= vals["thetadot0"] <= lim["thetadot"][1]:
            violations.append(f"θ̇₀={vals['thetadot0']:.2f} rad/s")

        if not violations:
            self.range_var.set("INITIAL STATE: IN TRAINING RANGE")
            self.range_label.config(fg="green")
            self.range_detail_var.set("All four initial-state components are inside the training range.")
            self.range_detail_label.config(fg="green")
        else:
            self.range_var.set("INITIAL STATE: OUT OF TRAINING RANGE")
            self.range_label.config(fg="darkorange")
            self.range_detail_var.set("Outside:\n" + "\n".join(violations))
            self.range_detail_label.config(fg="darkorange")

    # ==================================================================
    # Model selection
    # ==================================================================

    def refresh_models(self):
        candidates = []
        default_model = self.base_dir / "models" / "ppo" / "ppo_model_cartpole_stab.pt"
        if default_model.exists():
            candidates.append(default_model)

        if self.lab_runs_dir.exists():
            for run_dir in sorted(self.lab_runs_dir.iterdir(), reverse=True):
                if not run_dir.is_dir():
                    continue

                # If this RL Lab run has a user-named policy file, show that
                # instead of the generic model_best.pt filename.
                generic_names = {"model_best.pt", "model_latest.pt"}
                named_models = sorted(
                    [
                        p for p in run_dir.glob("*.pt")
                        if p.name not in generic_names
                    ]
                )

                best = run_dir / "model_best.pt"
                latest = run_dir / "model_latest.pt"

                if named_models:
                    candidates.append(named_models[0])
                elif best.exists():
                    candidates.append(best)
                elif latest.exists():
                    candidates.append(latest)

        display = [str(p.relative_to(self.base_dir)) if p.is_relative_to(self.base_dir) else str(p) for p in candidates]
        if hasattr(self, "model_combo"):
            self.model_combo["values"] = display

        if hasattr(self, "model_var") and not self.model_var.get() and display:
            self.model_var.set(display[0])

    def browse_model(self):
        path = filedialog.askopenfilename(
            title="Select PPO model",
            initialdir=str(self.base_dir),
            filetypes=[("PyTorch model", "*.pt"), ("All files", "*")]
        )
        if path:
            self.model_var.set(path)

    def resolve_model_path(self):
        raw = self.model_var.get().strip()
        if not raw:
            raise ValueError("Select a PPO model first.")
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = self.base_dir / p
        p = p.resolve()
        if not p.exists():
            raise FileNotFoundError(f"Model not found:\n{p}")
        return p

    def get_saved_config_for_model(self, model_path):
        config_path = model_path.parent / "config.yaml"
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                return data
            except Exception:
                return None
        return None

    # ==================================================================
    # Training
    # ==================================================================

    def train_new_policy(self):
        if self.training_process is not None and self.training_process.poll() is None:
            messagebox.showinfo("Training", "A training process is already running.")
            return

        if self.running:
            self.stop_evaluation()

        try:
            qx, qv, qt, qw, ru = self.read_reward_values()
            ppo = self.read_ppo_values()
        except Exception as exc:
            messagebox.showerror("Invalid training setting", str(exc))
            return

        # --------------------------------------------------------------
        # Ask for a meaningful policy name BEFORE training starts.
        # The training run still receives a timestamp so old experiments
        # are never overwritten accidentally.
        # --------------------------------------------------------------
        suggested_name = (
            f"ppo_g{ppo['gamma']:.3g}_"
            f"qx{qx:.3g}_qt{qt:.3g}_ru{ru:.3g}"
        )

        requested_name = simpledialog.askstring(
            "New PPO Policy",
            "Enter a name for this policy:\n\n"
            "Example: gamma099_angle_priority",
            initialvalue=suggested_name,
            parent=self.root,
        )

        if requested_name is None:
            # User pressed Cancel.
            return

        requested_name = requested_name.strip()
        if not requested_name:
            messagebox.showwarning(
                "Policy name",
                "Please enter a non-empty policy name."
            )
            return

        # Make the name safe for macOS/Linux filenames while keeping it readable.
        safe_name = "".join(
            c if (c.isalnum() or c in "-_") else "_"
            for c in requested_name
        )
        while "__" in safe_name:
            safe_name = safe_name.replace("__", "_")
        safe_name = safe_name.strip("_-")

        if not safe_name:
            messagebox.showwarning(
                "Policy name",
                "The entered name does not contain usable filename characters."
            )
            return

        run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{safe_name}_{run_stamp}"
        run_dir = self.lab_runs_dir / run_name
        override_path = self.lab_configs_dir / f"{run_name}.yaml"

        # Remember the requested name so model_best.pt can be copied to a
        # human-readable filename after training succeeds.
        self.pending_policy_name = safe_name

        custom_override = {
            "task_config": {
                "task": "stabilization",
                "task_info": {
                    "stabilization_goal": [self.TARGET_X, 0.0],
                    "stabilization_goal_tolerance": 0.0,
                },
                "randomized_init": True,
                "done_on_out_of_bound": True,
                "rew_state_weight": [qx, qv, qt, qw],
                "rew_act_weight": ru,
                "rew_exponential": self.reward_type_var.get() == "Exponential bounded",
            },
            "algo_config": ppo,
        }

        with open(override_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(custom_override, f, sort_keys=False)

        task_yaml = self.base_dir / "config_overrides" / "cartpole" / "cartpole_stab.yaml"
        ppo_yaml = self.base_dir / "config_overrides" / "cartpole" / "ppo_cartpole.yaml"
        train_script = self.repo_root / "safe_control_gym" / "experiments" / "train_rl_controller.py"

        if not train_script.exists():
            messagebox.showerror("Training script", f"Could not find:\n{train_script}")
            return

        cmd = [
            sys.executable,
            "-u",
            str(train_script),
            "--algo", "ppo",
            "--task", "cartpole",
            "--overrides",
            str(ppo_yaml),
            str(task_yaml),
            str(override_path),
            "--output_dir", str(run_dir),
            "--seed", "2",
        ]

        self.pending_run_dir = run_dir
        self.append_train_log("\n" + "=" * 54 + "\n")
        self.append_train_log("Starting new PPO training\n")
        self.append_train_log(f"Policy name: {safe_name}\n")
        self.append_train_log(f"Run directory: {run_name}\n")
        self.append_train_log(f"Reward Q: {[qx, qv, qt, qw]}, Ru={ru}\n")
        self.append_train_log(
            f"gamma={ppo['gamma']}, GAE={ppo['use_gae']}, lambda={ppo['gae_lambda']}, "
            f"clip={ppo['clip_param']}\n"
        )
        self.append_train_log("Fixed budget: 300,000 environment steps\n")

        self.status_var.set("Training PPO... evaluation is disabled until training finishes.")
        self.train_progress.start(12)
        self.train_button.configure(state="disabled")
        self.eval_button.configure(state="disabled")

        def worker():
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            try:
                process = subprocess.Popen(
                    cmd,
                    cwd=str(self.base_dir),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env=env,
                )
                self.training_process = process
                for line in process.stdout:
                    self.training_queue.put(("log", line))
                rc = process.wait()
                self.training_queue.put(("done", rc, str(run_dir)))
            except Exception as exc:
                self.training_queue.put(("error", str(exc)))

        self.training_thread = threading.Thread(target=worker, daemon=True)
        self.training_thread.start()

    def poll_training_queue(self):
        try:
            while True:
                item = self.training_queue.get_nowait()
                kind = item[0]
                if kind == "log":
                    self.append_train_log(item[1])
                elif kind == "done":
                    _, rc, run_dir_str = item
                    self.training_process = None
                    self.train_progress.stop()
                    self.train_button.configure(state="normal")
                    self.eval_button.configure(state="normal")
                    run_dir = Path(run_dir_str)
                    if rc == 0:
                        best = run_dir / "model_best.pt"
                        latest = run_dir / "model_latest.pt"
                        selected = best if best.exists() else latest

                        if selected.exists():
                            policy_name = getattr(
                                self,
                                "pending_policy_name",
                                "ppo_policy"
                            )
                            named_model = run_dir / f"{policy_name}.pt"

                            # Keep the original safe-control-gym output and also
                            # create a readable policy filename for the lab.
                            try:
                                shutil.copy2(selected, named_model)
                                selected = named_model
                            except Exception as exc:
                                self.append_train_log(
                                    f"Warning: could not create named model copy: {exc}\n"
                                )

                            try:
                                display = str(selected.relative_to(self.base_dir))
                            except Exception:
                                display = str(selected)
                            self.model_var.set(display)

                        self.refresh_models()
                        self.status_var.set(
                            "Training completed. New named policy selected for evaluation."
                        )
                        self.training_settings_dirty = False
                        self.retrain_var.set("")
                        self.append_train_log(
                            f"Training completed successfully.\n"
                            f"Selected policy: {selected.name if selected.exists() else 'not found'}\n"
                        )
                    else:
                        self.status_var.set(f"Training failed (return code {rc}).")
                        self.append_train_log(f"Training failed with return code {rc}.\n")
                elif kind == "error":
                    self.training_process = None
                    self.train_progress.stop()
                    self.train_button.configure(state="normal")
                    self.eval_button.configure(state="normal")
                    self.status_var.set("Training process error.")
                    messagebox.showerror("Training error", item[1])
        except queue.Empty:
            pass
        self.root.after(120, self.poll_training_queue)

    def cancel_training(self):
        if self.training_process is not None and self.training_process.poll() is None:
            try:
                self.training_process.terminate()
                self.status_var.set("Cancelling training...")
                self.append_train_log("Cancellation requested.\n")
            except Exception as exc:
                messagebox.showerror("Cancel training", str(exc))
        else:
            self.status_var.set("No active training process.")

    def append_train_log(self, text):
        if not hasattr(self, "train_log"):
            return
        self.train_log.configure(state="normal")
        self.train_log.insert(tk.END, text)
        self.train_log.see(tk.END)
        self.train_log.configure(state="disabled")

    # ==================================================================
    # Evaluation setup
    # ==================================================================

    def start_evaluation(self):
        if self.training_process is not None and self.training_process.poll() is None:
            messagebox.showwarning("Training active", "Wait for training to finish or cancel it first.")
            return

        if self.running:
            return

        try:
            vals = self.read_evaluation_values()
            model_path = self.resolve_model_path()
        except Exception as exc:
            messagebox.showerror("Evaluation setup", str(exc))
            return

        self.check_initial_training_range(vals)
        self.cleanup_evaluation()
        self.clear_plot_histories()
        self.clear_comparison_artists()
        self.current_result = None

        # --------------------------------------------------------------
        # Use the trained model's saved config when available.
        # This prevents "changing reward after training" from pretending
        # to change the learned policy.
        # --------------------------------------------------------------
        saved = self.get_saved_config_for_model(model_path)
        if saved and "task_config" in saved:
            task_config = copy.deepcopy(saved["task_config"])
            algo_config = copy.deepcopy(saved.get("algo_config", self.deep_dict(self.base_config.algo_config)))
            config_source = "saved training config"
        else:
            task_config = self.deep_dict(self.base_config.task_config)
            algo_config = self.deep_dict(self.base_config.algo_config)
            config_source = "current base config"

        # Force stabilization target = model/config target if available.
        try:
            self.target_x = float(task_config["task_info"]["stabilization_goal"][0])
        except Exception:
            self.target_x = self.TARGET_X

        # Evaluation-only overrides.
        task_config["randomized_init"] = False
        task_config["done_on_out_of_bound"] = False
        task_config["episode_len_sec"] = vals["episode"]
        task_config["init_state"] = {
            "init_x": vals["x0"],
            "init_x_dot": vals["xdot0"],
            "init_theta": math.radians(vals["theta_deg"]),
            "init_theta_dot": vals["thetadot0"],
        }

        algo_config["training"] = False

        env_func = partial(
            make,
            "cartpole",
            output_dir=str(self.temp_eval_dir),
            **task_config
        )

        try:
            self.env = env_func(gui=False)

            # Evaluation-only actuator force bound.
            self.eval_force_limit = vals["force_limit"]
            self.env.physical_action_bounds = (
                -np.atleast_1d(self.eval_force_limit),
                np.atleast_1d(self.eval_force_limit)
            )

            self.ctrl = make(
                "ppo",
                env_func,
                **algo_config,
                output_dir=str(self.temp_eval_dir / "controller")
            )
            self.ctrl.load(str(model_path))
            self.ctrl.agent.eval()
            self.ctrl.obs_normalizer.set_read_only()

            self.obs, self.info = self.env.reset()

        except Exception as exc:
            self.cleanup_evaluation()
            messagebox.showerror("Evaluation initialization", str(exc))
            return

        # Robustness settings.
        self.playback_speed = vals["playback"]
        self.eval_input_disturbance = vals["disturbance"]
        self.eval_action_delay = vals["delay"]
        self.eval_action_mode = vals["action_mode"]
        self.delay_buffer = deque([0.0] * self.eval_action_delay)

        # Noise is specified as a percentage of each training half-range.
        self.eval_noise_scale = (
            vals["noise_pct"] / 100.0
        ) * self.training_half_ranges()

        self.obs_ctrl = self.ctrl.obs_normalizer(
            self.make_policy_observation(self.obs)
        )

        self.ctrl_freq = float(task_config.get("ctrl_freq", 15.0))
        self.dt = 1.0 / self.ctrl_freq
        self.max_steps = int(vals["episode"] * self.ctrl_freq)
        self.step_count = 0
        self.running = True

        # Update plot references and ranges.
        self.update_target_lines()
        plot_limit = max(3.0, abs(vals["x0"]) + 2.0, abs(self.target_x) + 2.0)
        self.ax.set_xlim(-plot_limit, plot_limit)

        for axis in [self.ax_x, self.ax_theta, self.ax_force, self.ax_reward]:
            axis.set_xlim(0, vals["episode"])

        force_plot_lim = max(12.0, self.eval_force_limit * 1.2)
        self.ax_force.set_ylim(-force_plot_lim, force_plot_lim)
        self.force_limit_plus.set_ydata([self.eval_force_limit, self.eval_force_limit])
        self.force_limit_minus.set_ydata([-self.eval_force_limit, -self.eval_force_limit])

        self.status_var.set(
            f"Evaluating {model_path.name} @ {self.ctrl_freq:.1f} Hz ({config_source})"
        )

        self.update_drawing(
            self.obs,
            raw_action=0.0,
            delayed_action=0.0,
            command_force=0.0,
            applied_force=0.0,
            reward=0.0,
        )
        self.schedule_next_step(0.0)

    def make_policy_observation(self, true_obs):
        obs = np.asarray(true_obs, dtype=float).copy()
        if np.any(self.eval_noise_scale > 0):
            noise = np.random.normal(loc=0.0, scale=self.eval_noise_scale, size=4)
            obs[:4] += noise
        return obs

    def choose_policy_action(self):
        if self.eval_action_mode == "Stochastic":
            with torch.inference_mode():
                obs_tensor = torch.FloatTensor(self.obs_ctrl).to(self.ctrl.device)
                action, _, _ = self.ctrl.agent.ac.step(obs_tensor)
                return np.asarray(action, dtype=np.float32)
        return np.asarray(
            self.ctrl.select_action(obs=self.obs_ctrl, info=self.info),
            dtype=np.float32
        )

    # ==================================================================
    # Evaluation control loop
    # ==================================================================

    def schedule_next_step(self, elapsed_time):
        if not self.running:
            return
        target_period = self.dt / self.playback_speed
        remaining = max(0.001, target_period - elapsed_time)
        self.after_id = self.root.after(max(1, int(remaining * 1000)), self.control_step)

    def control_step(self):
        if not self.running:
            return

        tic = time.perf_counter()

        if self.step_count >= self.max_steps:
            self.finish_evaluation("Episode completed")
            return

        try:
            # Policy output in normalized action coordinates.
            action = self.choose_policy_action()
            raw_action = float(np.asarray(action).squeeze())

            # Action delay: first N samples are zero, then delayed policy action.
            if self.eval_action_delay > 0:
                self.delay_buffer.append(raw_action)
                delayed_action = float(self.delay_buffer.popleft())
            else:
                delayed_action = raw_action

            # The environment maps normalized action to physical force using 10 N scale.
            policy_force = delayed_action * float(self.env.action_scale)
            command_force = policy_force + self.eval_input_disturbance

            # Convert disturbed physical command back to normalized input expected by env.step().
            env_action = np.array(
                [command_force / float(self.env.action_scale)],
                dtype=np.float32
            )

            obs, reward, done, info = self.env.step(env_action)
            applied_force = float(np.asarray(self.env.current_clipped_action).squeeze())

            self.obs = obs
            self.info = info
            self.obs_ctrl = self.ctrl.obs_normalizer(
                self.make_policy_observation(obs)
            )

            self.step_count += 1
            sim_time = self.step_count * self.dt

            state = np.asarray(obs, dtype=float).flatten()
            x, xdot, theta, thetadot = map(float, state[:4])
            theta_deg = math.degrees(theta)
            reward_value = float(np.asarray(reward).squeeze())

            self.time_history.append(sim_time)
            self.x_history.append(x)
            self.xdot_history.append(xdot)
            self.theta_history.append(theta_deg)
            self.thetadot_history.append(thetadot)
            self.force_history.append(applied_force)
            self.reward_history.append(reward_value)
            self.action_history.append(raw_action)
            self.command_force_history.append(command_force)

            self.update_drawing(
                obs,
                raw_action=raw_action,
                delayed_action=delayed_action,
                command_force=command_force,
                applied_force=applied_force,
                reward=reward_value,
            )

            if done:
                self.finish_evaluation("Environment terminated episode")
                return

            elapsed = time.perf_counter() - tic
            self.schedule_next_step(elapsed)

        except Exception as exc:
            self.running = False
            self.status_var.set("Evaluation error")
            messagebox.showerror("Evaluation error", str(exc))

    # ==================================================================
    # Visualization
    # ==================================================================

    def update_target_lines(self):
        self.target_line.set_xdata([self.target_x, self.target_x])
        self.target_line.set_label(f"x ref = {self.target_x:.2f} m")
        self.x_ref_line.set_ydata([self.target_x, self.target_x])
        self.ax.legend(loc="upper right")

    def update_drawing(
        self,
        obs,
        raw_action,
        delayed_action,
        command_force,
        applied_force,
        reward,
    ):
        state = np.asarray(obs, dtype=float).flatten()
        x, xdot, theta, thetadot = map(float, state[:4])
        theta_deg = math.degrees(theta)

        # Cart.
        self.cart.set_xy((x - self.cart_width / 2, self.cart_y))

        # Pole.
        tip_x = x + self.pole_length_draw * math.sin(theta)
        tip_y = self.pivot_y + self.pole_length_draw * math.cos(theta)
        self.pole_line.set_data([x, tip_x], [self.pivot_y, tip_y])
        self.pivot_point.set_data([x], [self.pivot_y])

        # Force arrow.
        if self.force_arrow is not None:
            try:
                self.force_arrow.remove()
            except Exception:
                pass
            self.force_arrow = None

        self.force_arrow = FancyArrowPatch(
            (x, self.cart_y + self.cart_height / 2),
            (x + 0.055 * applied_force, self.cart_y + self.cart_height / 2),
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=2,
        )
        self.ax.add_patch(self.force_arrow)

        # Keep cart visible.
        xmin, xmax = self.ax.get_xlim()
        if x < xmin + 0.8 or x > xmax - 0.8:
            new_limit = max(abs(x) + 2.0, abs(self.target_x) + 2.0, 3.0)
            self.ax.set_xlim(-new_limit, new_limit)

        sim_time = self.step_count * self.dt
        saturated = abs(applied_force) >= (self.eval_force_limit - 1e-3)

        self.state_text.set_text(
            f"t             = {sim_time:6.2f} s\n"
            f"x             = {x:+8.3f} m\n"
            f"x_dot         = {xdot:+8.3f} m/s\n"
            f"theta         = {theta_deg:+8.2f} deg\n"
            f"theta_dot     = {thetadot:+8.3f} rad/s\n"
            f"PPO raw a     = {raw_action:+8.3f}\n"
            f"Delayed a     = {delayed_action:+8.3f}\n"
            f"Command force = {command_force:+8.2f} N\n"
            f"Applied force = {applied_force:+8.2f} N\n"
            f"Saturation    = {'YES' if saturated else 'no'}\n"
            f"Reward        = {float(reward):+8.4f}"
        )

        if self.time_history:
            self.x_line.set_data(self.time_history, self.x_history)
            self.theta_line.set_data(self.time_history, self.theta_history)
            self.force_line.set_data(self.time_history, self.force_history)
            self.reward_line.set_data(self.time_history, self.reward_history)

            self.dynamic_y_limits()

        self.canvas.draw_idle()

    def dynamic_y_limits(self):
        # x axis.
        xvals = self.x_history + [self.target_x]
        xmin, xmax = min(xvals), max(xvals)
        margin = max(0.2, 0.15 * (xmax - xmin + 1e-6))
        self.ax_x.set_ylim(xmin - margin, xmax + margin)

        # theta axis.
        tvals = self.theta_history + [0.0]
        tmin, tmax = min(tvals), max(tvals)
        margin = max(2.0, 0.15 * (tmax - tmin + 1e-6))
        self.ax_theta.set_ylim(tmin - margin, tmax + margin)

        # reward axis.
        rmin, rmax = min(self.reward_history), max(self.reward_history)
        margin = max(0.05, 0.15 * (rmax - rmin + 1e-6))
        self.ax_reward.set_ylim(rmin - margin, rmax + margin)

    def clear_plot_histories(self):
        self.clear_history_arrays()
        self.x_line.set_data([], [])
        self.theta_line.set_data([], [])
        self.force_line.set_data([], [])
        self.reward_line.set_data([], [])
        self.state_text.set_text("")
        self.canvas.draw_idle()

    # ==================================================================
    # Stop / finish / metrics
    # ==================================================================

    def stop_evaluation(self):
        was_running = self.running
        self.running = False
        if self.after_id is not None:
            try:
                self.root.after_cancel(self.after_id)
            except Exception:
                pass
            self.after_id = None

        if self.time_history:
            self.calculate_results(partial=True)
        if was_running:
            self.status_var.set("Evaluation stopped")

    def finish_evaluation(self, message):
        self.running = False
        if self.after_id is not None:
            try:
                self.root.after_cancel(self.after_id)
            except Exception:
                pass
            self.after_id = None
        self.calculate_results(partial=False)
        self.status_var.set(message)

    def calculate_results(self, partial=False):
        if not self.time_history:
            return

        t = np.asarray(self.time_history)
        x = np.asarray(self.x_history)
        xd = np.asarray(self.xdot_history)
        th_deg = np.asarray(self.theta_history)
        thd = np.asarray(self.thetadot_history)
        force = np.asarray(self.force_history)
        rew = np.asarray(self.reward_history)

        ex = x - self.target_x
        rmse_x = float(np.sqrt(np.mean(ex ** 2)))
        rmse_theta = float(np.sqrt(np.mean(th_deg ** 2)))
        rms_force = float(np.sqrt(np.mean(force ** 2)))

        # Settling definition: all remaining samples satisfy the four bands.
        settled_mask = (
            (np.abs(ex) <= 0.05)
            & (np.abs(th_deg) <= 1.0)
            & (np.abs(xd) <= 0.10)
            & (np.abs(thd) <= 0.10)
        )
        settling_time = None
        for i in range(len(t)):
            if np.all(settled_mask[i:]):
                settling_time = float(t[i])
                break

        model_text = self.model_var.get().strip()
        self.current_result = {
            "partial": partial,
            "model": model_text,
            "target_x": self.target_x,
            "time": list(self.time_history),
            "x": list(self.x_history),
            "theta": list(self.theta_history),
            "force": list(self.force_history),
            "reward": list(self.reward_history),
            "rmse_x": rmse_x,
            "rmse_theta": rmse_theta,
            "rms_force": rms_force,
            "final_x_error": float(ex[-1]),
            "final_theta_error": float(th_deg[-1]),
            "max_theta": float(np.max(np.abs(th_deg))),
            "max_force": float(np.max(np.abs(force))),
            "cumulative_reward": float(np.sum(rew)),
            "settling_time": settling_time,
            "action_delay": self.eval_action_delay,
            "noise_pct": float(self.obs_noise_pct_var.get()),
            "disturbance": self.eval_input_disturbance,
            "force_limit": self.eval_force_limit,
            "action_mode": self.eval_action_mode,
        }
        self.display_current_results()

    def display_current_results(self):
        if not self.current_result:
            return
        r = self.current_result
        settling = "not settled" if r["settling_time"] is None else f"{r['settling_time']:.3f} s"
        text = (
            f"{'PARTIAL ' if r['partial'] else ''}EVALUATION\n"
            f"Model: {Path(r['model']).name}\n\n"
            f"Final x error       {r['final_x_error']:+.4f} m\n"
            f"Final theta error   {r['final_theta_error']:+.3f} deg\n"
            f"RMSE x              {r['rmse_x']:.4f} m\n"
            f"RMSE theta          {r['rmse_theta']:.3f} deg\n"
            f"RMS force           {r['rms_force']:.3f} N\n"
            f"Max |theta|         {r['max_theta']:.3f} deg\n"
            f"Max |force|         {r['max_force']:.3f} N\n"
            f"Cumulative reward   {r['cumulative_reward']:.3f}\n"
            f"Settling time*      {settling}\n\n"
            "* |ex|<=0.05 m, |theta|<=1 deg,\n"
            "  |xdot|<=0.10 m/s, |thetadot|<=0.10 rad/s\n\n"
            f"Noise               {r['noise_pct']:.2f}% range\n"
            f"Input disturbance   {r['disturbance']:+.2f} N\n"
            f"Action delay        {r['action_delay']} steps\n"
            f"Force limit         ±{r['force_limit']:.2f} N\n"
            f"Action mode         {r['action_mode']}\n"
        )
        self.results_text.configure(state="normal")
        self.results_text.delete("1.0", tk.END)
        self.results_text.insert(tk.END, text)
        self.results_text.configure(state="disabled")
        self.notebook.select(self.tab_results)

    # ==================================================================
    # Comparison A/B
    # ==================================================================

    def store_result(self, slot):
        if not self.current_result:
            messagebox.showinfo("Comparison", "Run an evaluation first.")
            return
        if slot == "A":
            self.result_A = copy.deepcopy(self.current_result)
            self.status_var.set("Current evaluation stored as A")
        else:
            self.result_B = copy.deepcopy(self.current_result)
            self.status_var.set("Current evaluation stored as B")

    def compare_results(self):
        if not self.result_A or not self.result_B:
            messagebox.showinfo("Comparison", "Store one result as A and another as B first.")
            return

        A, B = self.result_A, self.result_B
        metrics = [
            ("RMSE x [m]", "rmse_x"),
            ("RMSE theta [deg]", "rmse_theta"),
            ("RMS force [N]", "rms_force"),
            ("Max |theta| [deg]", "max_theta"),
            ("Max |force| [N]", "max_force"),
            ("Cumulative reward", "cumulative_reward"),
        ]

        lines = [
            "A vs B COMPARISON",
            "",
            f"A: {Path(A['model']).name}",
            f"B: {Path(B['model']).name}",
            "",
            f"{'Metric':<22}{'A':>11}{'B':>11}",
            "-" * 44,
        ]
        for label, key in metrics:
            lines.append(f"{label:<22}{A[key]:>11.4f}{B[key]:>11.4f}")

        sa = "N/A" if A["settling_time"] is None else f"{A['settling_time']:.3f}"
        sb = "N/A" if B["settling_time"] is None else f"{B['settling_time']:.3f}"
        lines.append(f"{'Settling time [s]':<22}{sa:>11}{sb:>11}")

        self.compare_text.configure(state="normal")
        self.compare_text.delete("1.0", tk.END)
        self.compare_text.insert(tk.END, "\n".join(lines) + "\n")
        self.compare_text.configure(state="disabled")

        self.show_comparison_overlay()

    def clear_comparison_artists(self):
        for artist in self.comparison_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self.comparison_artists = []

    def show_comparison_overlay(self):
        self.clear_comparison_artists()
        A, B = self.result_A, self.result_B
        if not A or not B:
            return

        for axis, key in [
            (self.ax_x, "x"),
            (self.ax_theta, "theta"),
            (self.ax_force, "force"),
            (self.ax_reward, "reward"),
        ]:
            a_line, = axis.plot(A["time"], A[key], linestyle="--", linewidth=1.3, label="Experiment A")
            b_line, = axis.plot(B["time"], B[key], linestyle=":", linewidth=1.5, label="Experiment B")
            self.comparison_artists.extend([a_line, b_line])
            axis.legend(fontsize=8)

        self.canvas.draw_idle()

    # ==================================================================
    # Cleanup
    # ==================================================================

    def cleanup_evaluation(self):
        self.running = False
        if self.after_id is not None:
            try:
                self.root.after_cancel(self.after_id)
            except Exception:
                pass
            self.after_id = None

        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass
            self.env = None

        if self.ctrl is not None:
            try:
                self.ctrl.close()
            except Exception:
                pass
            self.ctrl = None

        shutil.rmtree(self.temp_eval_dir, ignore_errors=True)

    def on_close(self):
        self.cleanup_evaluation()
        if self.training_process is not None and self.training_process.poll() is None:
            try:
                self.training_process.terminate()
            except Exception:
                pass
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = CartPoleRLLab(root)
    root.mainloop()
