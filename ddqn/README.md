# Double Deep Q-Networks (DDQN)

This is a DDQN implementation for EC400 final project. The code from https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html was implemented, and only the necessary modificaitons and parameter changes were done to work with homework 4 files. 

1. create python environment by

```
python3 -m venv env
```

then 

```
source env/bin/activate
pip install -r requirements.txt
```

2. run utils in the homework directory by

```
cd homework
python3 -m utils [track_name]
```

It only works with one track at a time

The plots for q value, length, loss, and rewards will be stored in the save_checkpoints under the homework directory.


please contact ryu74@bu.edu for any question. 