from NFSP.TrainingProfile import TrainingProfile
from NFSP.workers.driver.Driver import Driver
from PokerRL import StandardLeduc

if __name__ == '__main__':
    ctrl = Driver(t_prof=TrainingProfile(name="nfsp_leduc_paper",
                                         nn_type="feedforward",
                                         n_envs=128,
                                         n_steps_pretrain_per_la=0,
                                         n_steps_per_iter_per_la=128,

                                         game_cls=StandardLeduc,

                                         DISTRIBUTED=False,
                                         target_net_update_freq=300,
                                         use_pre_layers_br=False,
                                         use_pre_layers_avg=False,
                                         n_units_final_br=128,
                                         n_units_final_avg=128,
                                         n_merge_and_table_layer_units_br=128,
                                         n_merge_and_table_layer_units_avg=128,

                                         mini_batch_size_br_per_la=128,
                                         mini_batch_size_avg_per_la=128,
                                         ),
                  eval_methods={"br": 2000},
                  n_iterations=None)
    ctrl.run()
