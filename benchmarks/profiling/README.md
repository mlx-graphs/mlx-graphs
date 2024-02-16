# Profiling mlx-graphs

Here we profile parts of our code to identify any bottlenecks.

To visualize the results, you can execute the selected python file, e.g.
```
python profile_training.py
```
which generates a `program.prof`. You can then visualize the results with `snakeviz`
```
snakeviz program.prof
```
