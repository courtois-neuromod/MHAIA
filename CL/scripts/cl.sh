SEED = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
SEQUENCE = [CD4, CD8, CO4, CO8, CD16, CO16, COC]

python CL/run_cl.py --sequence [SEQUENCE] --seed [SEED] --cl_method packnet --clipnorm 2e-05
python CL/run_cl.py --sequence [SEQUENCE] --seed [SEED] --cl_method mas --cl_reg_coef=10000
python CL/run_cl.py --sequence [SEQUENCE] --seed [SEED] --cl_method agem
python CL/run_cl.py --sequence [SEQUENCE] --seed [SEED] --cl_method l2 --cl_reg_coef=100000
python CL/run_cl.py --sequence [SEQUENCE] --seed [SEED] --cl_method ewc --cl_reg_coef=250
python CL/run_cl.py --sequence [SEQUENCE] --seed [SEED] --cl_method vcl --cl_reg_coef=1 --vcl_first_task_kl False
python CL/run_cl.py --sequence [SEQUENCE] --seed [SEED]  # Fine-tuning
python CL/run_cl.py --sequence [SEQUENCE] --seed [SEED] --batch_size 512 --buffer_type reservoir --reset_buffer_on_task_change False --replay_size 8e5  # Perfect Memory
python CL/run_cl.py --sequence [SEQUENCE] --seed [SEED] --cl_method clonex --cl_reg_coef=100 --exploration_kind 'best_return'

# COC
python CL/run_cl.py --sequence COC --sparse_rewards --seed [SEED] --cl_method packnet --clipnorm 2e-05

# Network plasticity
python CL/run_cl.py --sequence CO8 --seed [SEED] --repeat_sequence 10 --no_test --steps_per_env 100000

# Method Variations
python CL/run_cl.py --sequence CO8 --cl_method [METHOD] --seed [SEED] --augment --augmentation conv
python CL/run_cl.py --sequence CO8 --cl_method [METHOD] --seed [SEED] --augment --augmentation shift
python CL/run_cl.py --sequence CO8 --cl_method [METHOD] --seed [SEED] --augment --augmentation noise
python CL/run_cl.py --sequence CO8 --cl_method [METHOD] --seed [SEED] --buffer_type prioritized
python CL/run_cl.py --sequence CO8 --cl_method [METHOD] --seed [SEED] --use_lstm
python CL/run_cl.py --sequence CO8 --cl_method [METHOD] --seed [SEED] --regularize_critic