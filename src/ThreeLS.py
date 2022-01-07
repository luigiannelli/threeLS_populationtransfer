# Optimal Control and Reinforcement Learning tutorial for
# three-level population transfer
# Copyright (C) 2021 Luigi Giannelli

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import pickle

import matplotlib.pyplot as plt
import numpy as np
from qutip import (Options, basis, expect, ket2dm, liouvillian, mesolve,
                   operator_to_vector, vector_to_operator)
from scipy.optimize import minimize
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tqdm.auto import tqdm, trange

opts = Options(atol=1e-11, rtol=1e-9, nsteps=int(1e6))
# opts = Options(atol=1e-13, rtol=1e-11, nsteps=int(1e6))
opts.normalize_output = False  # mesolve is x3 faster if this is False


class ThreeLS_v0_env(py_environment.PyEnvironment):
    """Λ-system environment"""

    def __init__(
        self,
        Ωmax=1.0,
        Ωmin=0.0,
        Δ=0.0,
        δp=0.0,
        T=10.0,
        n_steps=20,
        γ=0.0,
        reward_gain=1.0,
        seed=1,
    ):
        """Initializes a new Λ-system environment.
        Args:
          seed: random seed for the RNG.
        """
        self.qstate = [basis(4, i) for i in range(4)]
        self.sig = [
            [self.qstate[i] * self.qstate[j].dag() for j in range(4)] for i in range(4)
        ]
        self.up = (self.sig[0][1] + self.sig[1][0]) / 2
        self.us = (self.sig[1][2] + self.sig[2][1]) / 2
        self.ψ0 = ket2dm(self.qstate[0])
        self.target_state = self.sig[2][2]

        self.Ωmax = Ωmax
        self.Ωmin = Ωmin
        self.Δ = Δ
        self.δp = δp
        self.T = T
        self.n_steps = n_steps
        self.γ = γ
        self.reward_gain = reward_gain

        self.current_step = 0
        self.current_qstate = self.ψ0
        self._state = self._dm2state(self.ψ0)
        self._episode_ended = False

        self.update()

    def update(self):
        self.H0 = self.Δ * self.sig[1][1] + self.δp * self.sig[2][2]
        self.tlist = np.linspace(0, self.T, self.n_steps + 1)
        self.Δt = self.tlist[1] - self.tlist[0]

    def action_spec(self):
        """Returns the action spec."""
        return array_spec.BoundedArraySpec(
            shape=(2,),
            dtype=np.float32,
            name="pulses",
            minimum=self.Ωmin,
            maximum=self.Ωmax,
        )

    def observation_spec(self):
        """Returns the observation spec."""
        return array_spec.BoundedArraySpec(
            shape=(9,),
            dtype=np.float32,
            name="density matrix",
            minimum=np.append(
                np.zeros(3, dtype=np.float32), -1 * np.ones(6, dtype=np.float32)
            ),
            maximum=np.ones(9, dtype=np.float32),
        )

    def _reset(self):
        """Resets the environment and returns the first `TimeStep` of a new episode."""
        # self._reset_next_step = False
        self.current_step = 0
        self.current_qstate = self.ψ0
        self._state = self._dm2state(self.ψ0)
        self._episode_ended = False
        return ts.restart(self._state)

    def _qstep(self, action, qstate):
        H = self.H0 + action[0] * self.up + action[1] * self.us
        L = (liouvillian(H, [np.sqrt(self.γ) * self.sig[3][1]]) * self.Δt).expm()
        return apply_superoperator(L, qstate)

    def _mesolvestep(self, action, qstate):
        H = self.H0 + action[0] * self.up + action[1] * self.us
        tlist = self.tlist[self.current_step : self.current_step + 2]
        result = mesolve(
            H, qstate, tlist, c_ops=[np.sqrt(self.γ) * self.sig[3][1]], options=opts
        )
        return result.states[-1]

    def _step(self, action):
        """Updates the environment according to the action."""

        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()

        if self.current_step < self.n_steps:
            self.current_qstate = self._qstep(action, self.current_qstate)
            # self.current_qstate = self._mesolvestep(action, self.current_qstate)
            next_state = self._dm2state(self.current_qstate)
            terminal = False
            reward = (
                0.0  # self.reward_gain * expect(self.target_state, self.current_qstate)
            )

            if self.current_step == self.n_steps - 1:
                reward = self.reward_gain * expect(
                    self.target_state, self.current_qstate
                )
                terminal = True
        else:
            terminal = True
            reward = 0
            next_state = 0
        self.current_step += 1

        if terminal:
            self._episode_ended = True
            return ts.termination(next_state, reward)
        else:
            return ts.transition(next_state, reward)

    def _dm2state(self, dm):
        return np.append(
            dm.diag()[:-1],
            np.append(
                dm.full()[([0, 0, 1], [1, 2, 2])].real,
                dm.full()[([0, 0, 1], [1, 2, 2])].imag,
            ),
        ).astype(np.float32)

    def run_evolution(self, amps):
        time_step = self.reset()
        time_step_list = [time_step]

        for i in range(self.n_steps):
            time_step = self.step(amps[i])
            time_step_list.append(time_step)

        assert time_step.is_last() == True

        state_list = np.array([x.observation for x in time_step_list])
        reward_list = np.array([x.reward for x in time_step_list])
        terminal_list = np.array([x.step_type for x in time_step_list])

        return state_list, reward_list, terminal_list

    def run_qstepevolution(self, amps):
        Ωp = amps[:, 0]
        Ωs = amps[:, 1]

        states = [self.ψ0]

        for i in range(self.n_steps):
            states.append(self._qstep([Ωp[i], Ωs[i]], states[-1]))

        return states

    def run_mesolvevolution(self, amps):
        Ωp = amps[:, 0]
        Ωs = amps[:, 1]

        fp = function_from_array(Ωp, self.tlist[:-1])
        fs = function_from_array(Ωs, self.tlist[:-1])

        H = [self.H0, [self.up, fp], [self.us, fs]]

        result = mesolve(
            H,
            self.ψ0,
            self.tlist,
            c_ops=[np.sqrt(self.γ) * self.sig[3][1]],
            options=opts,
        )
        return result

    def final_efficiency(self, amps):
        return self.reward_gain * expect(
            self.target_state, self.run_mesolvevolution(amps).states[-1]
        )

    def inefficiency(self, vals):
        amps = vals2amps(vals)
        return 1 - self.final_efficiency(amps)

    def final_qstepefficiency(self, amps):
        return self.reward_gain * expect(
            self.target_state, self.run_qstepevolution(amps)[-1]
        )

    def qstepinefficiency(self, vals):
        amps = vals2amps(vals)
        return 1 - self.final_qstepefficiency(amps)


def run_training(
    agent,
    train_driver,
    replay_buffer,
    eval_driver,
    eval_replay_buffer,
    avg_return,
    num_iterations,
    eval_interval=1,
    save_episodes=False,
    clear_buffer=False,
):
    return_list = []
    episode_list = []
    iteration_list = []
    with trange(num_iterations, dynamic_ncols=False) as t:
        for i in t:
            # t.set_description(f'episode {i}')

            if clear_buffer:
                replay_buffer.clear()

            final_time_step, policy_state = train_driver.run()
            experience = replay_buffer.gather_all()
            train_loss = agent.train(experience)

            if i % eval_interval == 0 or i == num_iterations - 1:
                avg_return.reset()
                final_time_step, policy_state = eval_driver.run()

                iteration_list.append(agent.train_step_counter.numpy())
                return_list.append(avg_return.result().numpy())

                t.set_postfix({"return": return_list[-1]})

                if save_episodes:
                    episode_list.append(eval_replay_buffer.gather_all())

    return return_list, episode_list, iteration_list


def vals2amps(vals):
    message = "vals must be 1-D array with shape (n,) where n is even"
    assert len(vals.shape) == 1, message
    assert vals.shape[0] % 2 == 0, message
    return vals.reshape(-1, 2, order="F")


def function_from_array(y, x):
    """Return function given an array and time points."""

    if y.shape[0] != x.shape[0]:
        raise ValueError("y and x must have the same first dimension")

    yx = np.column_stack((y, x))
    yx = yx[yx[:, -1].argsort()]

    def func(t, args):
        idx = np.searchsorted(yx[1:, -1], t, side="right")
        return yx[idx, 0]

    return func


def apply_superoperator(L, ρ):
    return vector_to_operator(L * operator_to_vector(ρ))


def plot_episode(episode, tlist, offset=0.05, env_py=None):
    Ωp = episode.action.numpy()[0, :, 0]
    Ωs = episode.action.numpy()[0, :, 1]

    fig, ax = plt.subplots(2, 1, figsize=[6, 6], sharex=True)

    ax[0].set_ylabel("pulses")
    ax[1].set_ylabel("populations")
    ax[1].set_xlabel("time")

    # ax[0].set_ylim(-offset, np.max(np.append(Ωp, Ωs)) + offset)
    ax[0].plot(tlist, Ωp, ds="steps-post", ls="-", label=r"$\Omega_\mathrm{p}$")
    ax[0].plot(tlist, Ωs, ds="steps-post", ls="-", label=r"$\Omega_\mathrm{s}$")

    if isinstance(env_py, ThreeLS_v0_env):
        fp = function_from_array(Ωp, tlist)
        fs = function_from_array(Ωs, tlist)

        H = [env_py.H0, [env_py.up, fp], [env_py.us, fs]]

        result = mesolve(
            H,
            env_py.ψ0,
            tlist,
            c_ops=[np.sqrt(env_py.γ) * env_py.sig[3][1]],
            options=opts,
        )

        for _ in range(4):
            ax[1].plot(
                tlist,
                expect(env_py.sig[_][_], result.states),
                ".",
                label=r"$\vert {} \rangle$".format(_ + 1),
            )

    for _ in range(3):
        ax[1].plot(
            tlist,
            episode.observation.numpy()[0, :, _],
            label=r"$\vert {} \rangle$".format(_ + 1),
        )

    ax[0].legend()
    ax[1].legend()
    fig.tight_layout()

    return fig


def plot_evolution(amps, env_parameters, mesolve_check=False):

    env_py = ThreeLS_v0_env(**env_parameters)

    tlist = env_py.tlist
    Ωp = np.append(amps[:, 0], amps[-1, 0])
    Ωs = np.append(amps[:, 1], amps[-1, 1])

    state_list, reward_list, terminal_list = env_py.run_evolution(amps)

    fig, ax = plt.subplots(2, 1, figsize=[6, 6], sharex=True)

    ax[0].set_ylabel("pulses")
    ax[1].set_ylabel("populations")
    ax[1].set_xlabel("time")

    ax[0].plot(tlist, Ωp, ds="steps-post", ls="-", label=r"$\Omega_\mathrm{p}$")
    ax[0].plot(tlist, Ωs, ds="steps-post", ls="--", label=r"$\Omega_\mathrm{s}$")

    for _ in range(3):
        ax[1].plot(tlist, state_list[:, _], label=r"$\vert {} \rangle$".format(_ + 1))

    if mesolve_check:
        result = env_py.run_mesolvevolution(amps)
        for _ in range(4):
            ax[1].plot(
                tlist,
                expect(env_py.sig[_][_], result.states),
                ".",
                label=r"$\vert {} \rangle$".format(_ + 1),
            )

    ax[0].legend()
    ax[1].legend()
    fig.tight_layout()

    print(f"final efficiency = {state_list[-1,2]}")
    if mesolve_check:
        print(
            f"final efficiency (mesolve) = {expect(env_py.target_state, result.states[-1])}"
        )

    return fig


def optimize_vs_pars(parameters):
    env_py = ThreeLS_v0_env(**parameters["env_parameters"])

    try:
        res = minimize(
            env_py.inefficiency,
            parameters["init_vals"],
            args=(),
            method="L-BFGS-B",
            bounds=parameters["bounds"],
            options=parameters["solver_opts"],
        )
    except:
        res = None

    return res


def save_object(obj, filename):
    try:
        with open(filename, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as ex:
        print("Error during pickling object (Possibly unsupported):", ex)


def load_object(filename):
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except Exception as ex:
        print("Error during unpickling object (Possibly unsupported):", ex)


def gaussian(x, T):
    return np.exp(-((x / T) ** 2))
