import d3rlpy

# prepare dataset
dataset, env = d3rlpy.datasets.get_minari("door-cloned-v1")

# prepare algorithm
cql = d3rlpy.algos.CQLConfig().create(device="cpu")

# train
cql.fit(
    dataset,
    n_steps=100,
    evaluators={"environment": d3rlpy.metrics.EnvironmentEvaluator(env)},
)
