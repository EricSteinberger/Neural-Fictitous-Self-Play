# Copyright (c) 2019 Eric Steinberger


import pickle

from NFSP.AvgWrapper import AvgWrapper
from NFSP.workers.la.ParallelEnvs import ParallelEnvs
from NFSP.workers.la.SeatActor import SeatActor
from PokerRL.rl import rl_util
from PokerRL.rl.agent_modules.DDQN import DDQN
from PokerRL.rl.base_cls.workers.WorkerBase import WorkerBase


class LearnerActor(WorkerBase):
    """
    Methods for acting are not included in this base.
    """

    def __init__(self, t_prof, worker_id, chief_handle):
        super().__init__(t_prof=t_prof)

        self._env_bldr = rl_util.get_env_builder(t_prof=t_prof)
        self._id = worker_id
        self._chief_handle = chief_handle

        self._ddqn_args = t_prof.module_args["ddqn"]
        self._avg_args = t_prof.module_args["avg"]

        if t_prof.nn_type == "recurrent":
            from PokerRL.rl.buffers.CircularBufferRNN import CircularBufferRNN
            from PokerRL.rl.buffers.BRMemorySaverRNN import BRMemorySaverRNN
            from NFSP.workers.la.action_buffer.ActionBufferRNN import ActionBufferRNN, AvgMemorySaverRNN

            BR_BUF_CLS = CircularBufferRNN
            BR_MEM_SAVER = BRMemorySaverRNN
            AVG_BUF_CLS = ActionBufferRNN
            AVG_MEM_SAVER = AvgMemorySaverRNN

        elif t_prof.nn_type == "feedforward":
            from PokerRL.rl.buffers.CircularBufferFLAT import CircularBufferFLAT
            from PokerRL.rl.buffers.BRMemorySaverFLAT import BRMemorySaverFLAT
            from NFSP.workers.la.action_buffer.ActionBufferFLAT import ActionBufferFLAT, AvgMemorySaverFLAT

            BR_BUF_CLS = CircularBufferFLAT
            BR_MEM_SAVER = BRMemorySaverFLAT
            AVG_BUF_CLS = ActionBufferFLAT
            AVG_MEM_SAVER = AvgMemorySaverFLAT
        else:
            raise ValueError(t_prof.nn_type)

        self._avg_bufs = [
            AVG_BUF_CLS(env_bldr=self._env_bldr, max_size=self._avg_args.res_buf_size,
                        min_prob=self._avg_args.min_prob_res_buf)
            for p in range(self._env_bldr.N_SEATS)
        ]
        self._br_bufs = [
            BR_BUF_CLS(env_bldr=self._env_bldr, max_size=self._ddqn_args.cir_buf_size)
            for p in range(self._env_bldr.N_SEATS)
        ]
        self._avg_memory_savers = [
            [
                AVG_MEM_SAVER(env_bldr=self._env_bldr, buffer=self._avg_bufs[p])
                for _ in range(self._t_prof.n_envs)
            ]
            for p in range(self._env_bldr.N_SEATS)
        ]
        self._br_memory_savers = [
            [
                BR_MEM_SAVER(env_bldr=self._env_bldr, buffer=self._br_bufs[p])
                for _ in range(self._t_prof.n_envs)
            ]
            for p in range(self._env_bldr.N_SEATS)
        ]
        self._br_learner = [
            DDQN(owner=p, ddqn_args=self._ddqn_args, env_bldr=self._env_bldr)
            for p in range(self._env_bldr.N_SEATS)
        ]
        self._avg_learner = [
            AvgWrapper(owner=p, env_bldr=self._env_bldr, avg_training_args=self._avg_args)
            for p in range(self._env_bldr.N_SEATS)
        ]

        self._seat_actors = [
            SeatActor(t_prof=t_prof, env_bldr=self._env_bldr, seat_id=p,
                      br_memory_savers=self._br_memory_savers[p], avg_buf_savers=self._avg_memory_savers[p],
                      br_learner=self._br_learner[p], avg_learner=self._avg_learner[p])
            for p in range(self._env_bldr.N_SEATS)
        ]

        self._parallel_env = ParallelEnvs(t_prof=t_prof, env_bldr=self._env_bldr, n_envs=self._t_prof.n_envs)

        self._last_step_wrappers = self._parallel_env.reset()
        for p in range(self._env_bldr.N_SEATS):
            self._seat_actors[p].init([sw for plyr_sws in self._last_step_wrappers for sw in plyr_sws])

    # ____________________________________________________ Playing _____________________________________________________
    def play(self, n_steps):
        self._all_eval()
        assert n_steps % self._parallel_env.n_envs == 0
        for n in range(n_steps // self._parallel_env.n_envs):
            # merge player's lists
            sws = [sw for plyr_sws in self._last_step_wrappers for sw in plyr_sws]

            # Both players must see all envs here
            for s in self._seat_actors:
                s.update_if_terminal(sws)

            # Let players act on the envs
            for s in self._seat_actors:
                s.act(self._last_step_wrappers[s.seat_id])

            # Step envs
            self._last_step_wrappers = self._parallel_env.step(step_wraps=sws)

        return self._id

    # ____________________________________________________ Learning ____________________________________________________
    def get_br_grads(self, p_id):
        self._br_learner[p_id].train()
        g = self._br_learner[p_id].get_grads_one_batch_from_buffer(buffer=self._br_bufs[p_id])
        if g is None:
            return None
        return self._ray.grads_to_numpy(g)

    def get_avg_grads(self, p_id):
        self._avg_learner[p_id].train()
        g = self._avg_learner[p_id].get_grads_one_batch_from_buffer(buffer=self._avg_bufs[p_id])
        if g is None:
            return None
        return self._ray.grads_to_numpy(g)

    def update(self,
               p_id,
               q1_state_dict,
               avg_state_dict,
               eps,
               ):
        if q1_state_dict is not None:
            dict_torch = self._ray.state_dict_to_torch(q1_state_dict, device=self._br_learner[p_id].device)
            self._br_learner[p_id].load_net_state_dict(dict_torch)

        if avg_state_dict is not None:
            dict_torch = self._ray.state_dict_to_torch(avg_state_dict, device=self._avg_learner[p_id].device)
            self._avg_learner[p_id].load_net_state_dict(dict_torch)

        if eps is not None:
            self._br_learner[p_id].eps = eps

    def update_q2(self, p_id):
        self._br_learner[p_id].update_target_net()

    def empty_cir_bufs(self):
        for b in self._br_bufs:
            b.reset()

    # __________________________________________________________________________________________________________________
    def checkpoint(self, curr_step):
        for p_id in range(self._env_bldr.N_SEATS):
            state = {
                "pi": self._avg_learner[p_id].state_dict(),
                "br": self._br_learner[p_id].state_dict(),
                "cir": self._br_bufs[p_id].state_dict(),
                "res": self._avg_bufs[p_id].state_dict(),
                "p_id": p_id,
            }
            with open(self._get_checkpoint_file_path(name=self._t_prof.name, step=curr_step,
                                                     cls=self.__class__, worker_id=str(self._id) + "_P" + str(p_id)),
                      "wb") as pkl_file:
                pickle.dump(obj=state, file=pkl_file, protocol=pickle.HIGHEST_PROTOCOL)

        with open(self._get_checkpoint_file_path(name=self._t_prof.name, step=curr_step,
                                                 cls=self.__class__, worker_id=str(self._id) + "_General"),
                  "wb") as pkl_file:
            state = {
                "env": self._parallel_env.state_dict()
            }
            pickle.dump(obj=state, file=pkl_file, protocol=pickle.HIGHEST_PROTOCOL)

    def load_checkpoint(self, name_to_load, step):
        for p_id in range(self._env_bldr.N_SEATS):
            with open(self._get_checkpoint_file_path(name=name_to_load, step=step,
                                                     cls=self.__class__, worker_id=str(self._id) + "_P" + str(p_id)),
                      "rb") as pkl_file:
                state = pickle.load(pkl_file)

                assert state["p_id"] == p_id

                self._avg_learner[p_id].load_state_dict(state["avg"])
                self._br_learner[p_id].load_state_dict(state["br"])
                self._br_bufs[p_id].load_state_dict(state["cir"])
                self._avg_bufs[p_id].load_state_dict(state["res"])

        with open(self._get_checkpoint_file_path(name=name_to_load, step=step,
                                                 cls=self.__class__, worker_id=str(self._id) + "_General"),
                  "rb") as pkl_file:
            state = pickle.load(pkl_file)
            self._parallel_env.load_state_dict(state["env"])
            self._last_step_wrappers = self._parallel_env.reset()

    def _all_eval(self):
        for q in self._br_learner:
            q.eval()
        for a_l in self._avg_learner:
            a_l.eval()

    def _all_train(self):
        for q in self._br_learner:
            q.train()
        for a_l in self._avg_learner:
            a_l.train()
