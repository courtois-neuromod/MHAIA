## Result Module
This module provides scripts for downloading and plotting the results of the experiments in our paper.
The results are stored in [Weights | Biases](https://wandb.ai/) and can be downloaded using the scripts in the [download](download) folder.
The plotting scripts are located in the [plotting](plotting) folder.
Calculating the core results in the paper can be done using the [cl_metrics.py](tables/cl_metrics.py) script.

## Installation
To install the results module, run the following command:
```bash 
$ pip install COOM[results]
```

### Running experiments
For running the experiments in our paper please follow the instructions in the continual learning (CL) module [README.md](../CL/README.md).

### Downloading results
We recommend using [Weights | Biases](https://wandb.ai/) to log your experiments. 
Having done so, you can use the following scripts to download the results:
1. Continual Learning Data - [cl_data.py](download/cl_data.py)  
`python cl_data.py --project <WANDB_PROJECT> --sequence <SEQUENCE>`
2. Single Run Data
   1. Single Run Data - [single_data.py](download/single_task_data.py)  
   `python single_data.py --project <WANDB_PROJECT> --sequence <SEQUENCE>`
   2. Evaluation data on given tasks  
   `python single_data.py --project <WANDB_PROJECT> --sequence <SEQUENCE> --test_envs <TEST_ENV_1> <TEST_ENV_2> ...`
3. Action Distribution Data - [action_data.py](download/action_data.py)
4. Runtime Data - [runtime_data.py](download/runtime_data.py)  
   1. For memory usage data run  
   `python runtime_data.py --project <WANDB_PROJECT> --sequence <SEQUENCE> --metric system.proc.memory.rssMB`  
   2. For walltime data run  
   `python runtime_data.py --project <WANDB_PROJECT> --sequence <SEQUENCE> --metric walltime`  

### Plotting figures

Figures from the paper can be drawn using the [plotting scripts](https://github.com/TTomilin/COOM/tree/main/results/plotting).
1. Ablation study bar plots  
Compare ablation results to the default setting - [ablations.py](results/plotting/ablations.py)  
`python plotting/ablations.py --sequence CO8 --tags default noise conv shift reg_critic single_head no_task_id --methods packnet l2 mas ewc clonex agem`
2. Stackplots of action distributions 
   1. Single method during evaluation of all environments of a sequence - [actions_all_envs.py](results/plotting/actions_all_envs.py)  
   `python plotting/actions_all_envs.py --method packnet --sequence CO8 --episode_length 1000 n_actions 12`
   2. Selected methods during training on a given sequence - [actions_by_method.py](results/plotting/actions_by_method.py)  
   `python plotting/actions_all_envs.py --methods packnet vcl l2 agem --sequence CO8 --episode_length 1000 n_actions 12`
   3. Single method during training on the given sequences - [actions_by_sequence.py](results/plotting/actions_by_sequence.py)  
   `python plotting/actions_all_envs.py --method packnet --sequences CO8 COC --episode_length 1000 n_actions 12`
   4. Includes a histogram of training actions - [actions_histogram.py](results/plotting/actions_histogram.py)  
   `python plotting/actions_all_envs.py --method packnet --sequences CO8 COC --episode_length 1000 n_actions 12`
3. Line plots of cumulative success during evaluation
   1. Compare ablations on a given sequence and methods - [avg_success_ablations.py](results/plotting/avg_success_ablations.py)  
   `python plotting/avg_success_methods.py --sequence CO8 --methods packnet ewc mas`
   2. Compare method performance across sequences - [avg_success_sequence.py](results/plotting/avg_success_sequence.py)  
   `python plotting/avg_success_sequence.py --method packnet --sequences CD8 CO8`
4. Resource consumption histograms
   1. Memory consumption during training - [consumption.py](results/plotting/consumption.py)  
   `python plotting/consumption.py --metric memory --sequence CO4 --methods clonex ewc vcl agem`
   2. Walltime per training step - [consumption.py](results/plotting/consumption.py)  
   `python plotting/consumption.py --metric walltime --sequence CO8 --methods packnet mas l2 perfect_memory`
5. Line plots of average success during evaluation on individual envs (useful to visualizer forgetting)
   1. Compare methods - [perf_per_env.py](results/plotting/perf_per_env.py)  
   `python plotting/perf_per_env.py --sequence CO8 --methods fine_tuning mas clonex packnet`
   2. Compare methods across multiple sequences - [perf_per_env_n_seq.py](results/plotting/perf_per_env_n_seq.py)  
      `python plotting/perf_per_env_n_seq.py --sequences CO8 COC --methods packnet l2`
   3. Compare methods on envs - [perf_per_method.py](results/plotting/perf_per_method.py)  
   `python plotting/perf_per_method.py --sequence CO8 --methods mas vcl packnet`
6. Plasticity line plots  
Visualize loss of plasticity across training repetitions of a sequence - [plasticity.py](results/plotting/plasticity.py)  
`python plotting/plasticity.py --sequence CO8 --method fine_tuning --n_repeats 10`
7. Training performance line plots
   1. Compare method across sequences on individual envs - [train_comparison_per_env.py](results/plotting/train_comparison_per_env.py)  
      `python plotting/train_comparison_per_env.py --sequences CO8 COC --methods packnet mas l2`
   2. Compare methods across sequences - [train_comparison_per_method.py](results/plotting/train_comparison_per_method.py)  
      `python plotting/train_comparison_per_method.py --sequences CO8 COC --methods packnet mas l2`
8. Visualize forward transfer with shaded areas between the training curves of the RL baseline and the CL method
   1. Per environment - [transfer_per_env.py](results/plotting/transfer_per_env.py)  
      `python plotting/transfer_per_env.py --sequence CO8 --method packnet`
   2. Compare methods on the full sequence - [transfer_per_method.py](results/plotting/transfer_per_method.py)  
      `python plotting/transfer_per_method.py --sequence CO8 --methods packnet clonex mas fine_tuning vcl l2 ewc agem`

### Calculating metrics
The results tables displayed in our paper can be obtained using the [scripts](https://github.com/TTomilin/COOM/tree/main/results/tables) for drawing tables.
1. Ablation study results - [ablations.py](tables/ablations.py)  
`python tables/ablations.py --sequence CO8 --tags default noise conv shift reg_critic single_head no_task_id --methods packnet l2 mas ewc clonex agem`
2. Continual learning metrics across sequences and methods - [cl_metrics.py](tables/cl_metrics.py)  
`python tables/cl_metrics.py --sequences CD4 CO4 CD8 CO8 COC --methods packnet mas agem l2 ewc vcl fine_tuning clonex perfect_memory`
