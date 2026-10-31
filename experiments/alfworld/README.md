# ALFWorld Experiment Archive

The scripts and outputs here belong to our ALFWorld experiments. The runtime/config source is the external [trace2env / Word2World repository](https://github.com/ruyue0001/trace2env), specifically its `baseline/Word2World/AgentGym` directory. The local `AgentGym` symlink is intentionally not tracked because it points outside this repository. Clone that dependency and create a local symlink named `AgentGym` here before running the archived scripts. Use the existing `word2world-alfworld-py310` environment so project-relative paths resolve here.

The environment-wrapper adjustment used during earlier runs is preserved in `patches/alfworld_env_wrapper.patch`. The data-description collector in `scripts/collect_init_context/` is the project-local copy used by the experiment scripts.
