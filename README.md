# toy_environment
Toy 2D environment for Julius


# Quick start:

```
cd ./src
mkdir ../environments
python3 env_conf_maker.py
python3 test_env.py
```

# Explanations:

`src` contains the source code
`models` contains json files describing the robots
`environments` contains pickle files, associating json files with environment parameters. See env_conf_maker.


The `viewer.py` file contains various code snippets that I've been using for displaying the agent while performing. This is very convenient for debuging / seeing what the agent does. I'll tell you more about that when you'll need it.

The `Renderer` class in `environment.py` generates the data of the vision sensor of the agent.
The `Skin` and `TactileSensor` classes generate the data for the tactile sensor of the agent.


I can also provide you the skeleton of the asynchronous setup if needed.

Cheers,
Charles
