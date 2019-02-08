# Copyright (c) 2019 Eric Steinberger


import numpy as np
import torch

from PokerRL.rl import rl_util


class SeatActor:
    _AVG = 1
    _BR = 2

    def __init__(self, seat_id, t_prof, env_bldr, br_memory_savers, avg_buf_savers, br_learner, avg_learner):
        self.seat_id = seat_id

        self._t_prof = t_prof
        self._env_bldr = env_bldr

        self.br_learner = br_learner
        self.avg_learner = avg_learner

        self._avg_buf_savers = avg_buf_savers
        self._br_memory_savers = br_memory_savers

        self._current_policy_tags = None

        self._n_actions_arranged = np.arange(self._env_bldr.N_ACTIONS)

    def init(self, step_wrappers):
        self._current_policy_tags = np.empty(shape=self._t_prof.n_envs, dtype=np.int32)

        for sw in step_wrappers:
            self._current_policy_tags[sw.env_idx] = self._pick_training_policy()
            self._br_memory_savers[sw.env_idx].reset(range_idx=sw.range_idxs[self.seat_id])
            self._avg_buf_savers[sw.env_idx].reset(range_idx=sw.range_idxs[self.seat_id])

    def act(self, step_wrappers):
        # """""""""""""""""""""
        # Act
        # """""""""""""""""""""
        self._insert_actions_mixed(step_wrappers)

        # """""""""""""""""""""
        # Add to memories
        # """""""""""""""""""""
        for sw in step_wrappers:
            e_i = sw.env_idx
            if (self._current_policy_tags[e_i] == SeatActor._BR) and (
                self._t_prof.add_random_actions_to_buffer or (not sw.action_was_random)):
                self._avg_buf_savers[e_i].add_step(pub_obs=sw.obs,
                                                   a=sw.action.item(),
                                                   legal_actions_mask=rl_util.get_legal_action_mask_np(
                                                      n_actions=self._env_bldr.N_ACTIONS,
                                                      legal_actions_list=sw.legal_actions_list)
                                                   )
            self._br_memory_savers[e_i].add_non_terminal_experience(obs_t_before_acted=sw.obs,
                                                                    a_selected_t=sw.action.item(),
                                                                    legal_actions_list_t=sw.legal_actions_list)

    def update_if_terminal(self, step_wrappers):
        for sw in step_wrappers:
            if sw.TERMINAL:
                self._br_memory_savers[sw.env_idx].add_terminal(
                    reward_p=sw.term_rew_all[self.seat_id],
                    terminal_obs=sw.term_obs,
                )
                self._br_memory_savers[sw.env_idx].reset(range_idx=sw.range_idxs[self.seat_id])
                self._avg_buf_savers[sw.env_idx].reset(range_idx=sw.range_idxs[self.seat_id])

                self._current_policy_tags[sw.env_idx] = self._pick_training_policy()

    def choose_a_br(self, pub_obses, range_idxs, legal_actions_lists, explore=True):
        """
        TODO maybe allow some explore some BR

        Returns:
            actions, was_random?:
        """
        # """""""""""""""""""""
        # Perhaps explore
        # """""""""""""""""""""
        if explore and (self.br_learner.eps > np.random.random()):
            actions = np.array([
                l[np.random.randint(low=0, high=len(l))]
                for l in legal_actions_lists
            ])
            return actions, True

        with torch.no_grad():
            # """""""""""""""""""""
            # Play by BR
            # """""""""""""""""""""
            actions = self.br_learner.select_br_a(
                pub_obses=pub_obses,
                range_idxs=range_idxs,
                legal_actions_lists=legal_actions_lists,
            )
            return actions, False

    def _pick_training_policy(self):
        if self._t_prof.anticipatory_parameter < np.random.random():
            return SeatActor._AVG
        return SeatActor._BR

    def _insert_actions_mixed(self, state_wraps):
        """ play with p*eps*rnd + p*(1-eps)*br and (1-p)*avg policy """

        with torch.no_grad():

            # """"""""""""""""""""""""
            # Construct
            # """"""""""""""""""""""""
            pub_obses_AVG = []
            range_idxs_AVG = []
            legal_actions_lists_AVG = []
            _sw_list_idxs_AVG = []

            pub_obses_BR = []
            range_idxs_BR = []
            legal_actions_lists_BR = []
            _sw_list_idxs_BR = []

            for i, sw in enumerate(state_wraps):
                if self._current_policy_tags[sw.env_idx] == SeatActor._AVG:
                    pub_obses_AVG.append(sw.obs)
                    range_idxs_AVG.append(sw.range_idxs[self.seat_id])
                    legal_actions_lists_AVG.append(sw.legal_actions_list)
                    _sw_list_idxs_AVG.append(i)
                elif self._current_policy_tags[sw.env_idx] == SeatActor._BR:
                    pub_obses_BR.append(sw.obs)
                    range_idxs_BR.append(sw.range_idxs[self.seat_id])
                    legal_actions_lists_BR.append(sw.legal_actions_list)
                    _sw_list_idxs_BR.append(i)
                else:
                    raise ValueError(self._current_policy_tags[sw.env_idx])

            range_idxs_AVG = np.array(range_idxs_AVG)
            range_idxs_BR = np.array(range_idxs_BR)

            # """"""""""""""""""""""""
            # AVG actions
            # """"""""""""""""""""""""
            if len(_sw_list_idxs_AVG) > 0:
                a_probs = self.avg_learner.get_a_probs(
                    range_idxs=range_idxs_AVG,
                    pub_obses=pub_obses_AVG,
                    legal_actions_lists=legal_actions_lists_AVG,
                )
                for i, ii in enumerate(_sw_list_idxs_AVG):
                    state_wraps[ii].action = np.random.choice(
                        a=self._n_actions_arranged,
                        p=a_probs[i],
                        replace=True
                    )
                    state_wraps[ii].action_was_random = False

            # """"""""""""""""""""""""
            # Greedy actions
            # """"""""""""""""""""""""
            if len(_sw_list_idxs_BR) > 0:
                actions, was_rnd = self.choose_a_br(
                    pub_obses=pub_obses_BR,
                    range_idxs=range_idxs_BR,
                    legal_actions_lists=legal_actions_lists_BR,
                    explore=True,
                )
                for i, ii in enumerate(_sw_list_idxs_BR):
                    state_wraps[ii].action = actions[i]
                    state_wraps[ii].action_was_random = was_rnd
