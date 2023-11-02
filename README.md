# AccountableBatchedController

The following code reproduces the results reported in the paper. 

1. To generate batched dataset, run

```
python3 generate_data.py --env ENV_NAME
```

- Change the --inv_forv = 1.0 or -1.0 will generate offline dataset for both normal dynamics and a reversed dynamics. (to namifest the multiple potential outcome settings/ POMDP setting)
- Change the --env = 'Pendulum-v1', 'LunarLanderContinuous-v2' to generate data for different tasks.

2. To reproduce our main results, run

```
python3 reproduce_xxx.py
```
