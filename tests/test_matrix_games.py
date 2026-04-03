from src.envs.matrix_games import play_episode

def test_matrix_episode_reward_range():
    *_,r=play_episode([0.1,0.2])
    assert -1.0 <= r <= 1.0
