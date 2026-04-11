from src.envs.matrix_games import make_matrix_env


def test_matrix_episode_reward_and_obs_shape():
    env = make_matrix_env("matrix_coordination", seed=0)
    obs, _ = env.reset(seed=0)
    assert set(obs.keys()) == {"agent_0", "agent_1", "agent_2"}
    assert obs["agent_0"].shape == (6,)
    _, rewards, terminateds, _, infos = env.step({"agent_0": 0, "agent_1": 0, "agent_2": 0})
    assert rewards["agent_0"] == 1.0
    assert all(terminateds.values())
    assert infos["agent_0"]["success"] == 1
