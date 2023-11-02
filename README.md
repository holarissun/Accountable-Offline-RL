# [NeurIPS'2023] Accountability in Offline Reinforcement Learning: Explaining Decisions with a Corpus of Examples 

## [Paper](https://arxiv.org/pdf/2310.07747.pdf) 


### Requirement
```
gym==0.26.2
pytorch==2.1.0
shapely==2.0.2
scipy==1.11.3
medkit-learn==0.1.0
tqdm
```

The following code reproduces the results reported in the paper. 

1. To generate a batched dataset, run

```
python3 generate_data.py --env ENV_NAME
```

- Change the --inv_forv = 1.0 or -1.0 will generate an offline dataset for normal or reversed dynamics. (to manifest the multiple potential outcome settings/ POMDP setting)
- Change the --env = 'Pendulum-v1', 'LunarLanderContinuous-v2' to generate data for different tasks. The Pendulum-v1 dataset is provided in the dataset folder.


2. To reproduce our main results, run

- Pendulum-v1 experiments (quantitative)
```
python3 reproduce_pendulum-het.py --dsize 600000
```

- Maze experiments (qualitative)
```
python3 reproduce_maze_conservation.py
```

```
python3 reproduce_maze_accountability.py
```

- Reproduce Ward (qualitative and quantitative)
```
python3 reproduce_ward.py
```
