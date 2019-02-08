from NFSP.TrainingProfile import TrainingProfile
from NFSP.workers.driver.Driver import Driver
from PokerRL import DiscretizedNLHoldem, Poker
from PokerRL.eval.lbr import LBRArgs
from PokerRL.game import bet_sets

if __name__ == '__main__':
    # Agent processes: 1 Chief, 2 Parameter-servers, 11 LAs
    # Eval processes: 1 Master, 8 Workers
    # Leave 1 Docker etc.
    # ==> 24 cores needed.
    # You can run this on e.g. a m5.12xlarge machine with hyper-threading disabled (effectively 24 cores and threads).
    # You can also parallelized further since only 5% of the execution time was spent syncing.

    N_WORKERS = 16
    N_LBR_WORKERS = 3

    ctrl = Driver(t_prof=TrainingProfile(name="NFSP_DISTRIBUTED_LH_RNN",

                                         DISTRIBUTED=True,
                                         n_learner_actor_workers=N_WORKERS,

                                         nn_type="recurrent",

                                         game_cls=DiscretizedNLHoldem,
                                         agent_bet_set=bet_sets.B_5,

                                         use_pre_layers_br=True,
                                         use_pre_layers_avg=True,
                                         n_units_final_br=64,
                                         n_units_final_avg=64,
                                         n_merge_and_table_layer_units_br=64,
                                         n_merge_and_table_layer_units_avg=64,
                                         rnn_units_br=64,
                                         rnn_units_avg=64,
                                         n_cards_state_units_br=128,
                                         n_cards_state_units_avg=128,

                                         cir_buf_size_each_la=6e5 / N_WORKERS,
                                         res_buf_size_each_la=2e6,
                                         n_envs=128,
                                         n_steps_per_iter_per_la=128,

                                         lr_br=0.1,
                                         lr_avg=0.01,

                                         mini_batch_size_br_per_la=64,
                                         mini_batch_size_avg_per_la=64,
                                         n_br_updates_per_iter=1,
                                         n_avg_updates_per_iter=1,

                                         eps_start=0.08,
                                         eps_const=0.007,
                                         eps_exponent=0.5,
                                         eps_min=0.0,

                                         lbr_args=LBRArgs(
                                             lbr_bet_set=bet_sets.B_5,
                                             n_lbr_hands_per_seat=15000,
                                             lbr_check_to_round=Poker.TURN,
                                             n_parallel_lbr_workers=N_LBR_WORKERS,
                                             use_gpu_for_batch_eval=False,
                                             DISTRIBUTED=True,
                                         )
                                         ),

                  eval_methods={"lbr": 25000},
                  n_iterations=None)
    ctrl.run()
