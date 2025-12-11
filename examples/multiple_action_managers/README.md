# Multiple Action Managers

Demonstrates using multliple action managers for a single robot.

## Training

This will be trained using the [rsl_rl](https://github.com/leggedrobotics/rsl_rl) training library. So first, we need to install that and tensorboard:

```bash
pip install tensorboard rsl-rl-lib>=2.2.4
```

Now you can run the training with:

```bash
python ./train.py
```

You can view the training progress with:

```bash
tensorboard --logdir ./logs/
```

The Genesis Forge training environment will also save videos while training that can be viewed in `./logs/go2-walking/videos`.

https://github.com/user-attachments/assets/be46df1b-35e5-4b5b-9bbc-f543210dd463

## Evaluation

Now you can view the trained policy:

```bash
python ./eval.py ./logs/go2-walking/
```
