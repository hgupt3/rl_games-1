import types

import pytest
import torch

from rl_games.common.a2c_common import A2CBase, ContinuousA2CBase


def test_true():
    assert True


def _bare_agent(config_lr):
    agent = A2CBase.__new__(A2CBase)
    agent.model = torch.nn.Linear(1, 1)
    agent.optimizer = torch.optim.Adam(agent.model.parameters(), lr=config_lr)
    agent.epoch_num = 5
    agent.frame = 1000
    agent.has_central_value = False
    agent.last_mean_rewards = 1.0
    agent.vec_env = None
    agent.mixed_precision = False
    agent.normalize_input = False
    agent.normalize_value = False
    agent.normalize_rms_advantage = False
    agent.last_lr = config_lr
    agent.entropy_coef = 0.0
    return agent


def test_adaptive_kl_lr_survives_full_state_round_trip():
    # Pin: the adaptive-KL adapted LR (and entropy_coef) must survive
    # save -> warm resume instead of restarting at the config value.
    saver = _bare_agent(config_lr=3e-4)
    saver.last_lr = 1e-5
    saver.entropy_coef = 0.007
    for group in saver.optimizer.param_groups:
        group['lr'] = 1e-5
    state = saver.get_full_state_weights()
    assert state['last_lr'] == 1e-5
    assert state['entropy_coef'] == 0.007

    fresh = _bare_agent(config_lr=3e-4)
    fresh.set_full_state_weights(state, set_epoch=True)
    assert fresh.last_lr == 1e-5
    assert fresh.entropy_coef == 0.007
    for group in fresh.optimizer.param_groups:
        assert group['lr'] == pytest.approx(1e-5)
    # Every restore marks the first post-resume rollout for discard.
    assert fresh._discard_post_resume_rollout is True

    # Pre-fix checkpoints (no last_lr/entropy_coef keys) load unchanged:
    # last_lr and entropy_coef keep their init (config) values.
    legacy = {k: v for k, v in state.items() if k not in ('last_lr', 'entropy_coef')}
    old = _bare_agent(config_lr=3e-4)
    old.set_full_state_weights(legacy, set_epoch=True)
    assert old.last_lr == 3e-4
    assert old.entropy_coef == 0.0


def test_resume_overrides_apply_only_without_persisted_keys(monkeypatch):
    # Rollbacks onto pre-fix checkpoints inject the wandb-history values;
    # checkpoints that persisted their own keys must ignore the override.
    saver = _bare_agent(config_lr=3e-4)
    saver.last_lr = 1e-5
    saver.entropy_coef = 0.007
    state = saver.get_full_state_weights()
    legacy = {k: v for k, v in state.items() if k not in ('last_lr', 'entropy_coef')}

    monkeypatch.setenv('RLG_RESUME_LAST_LR', '2e-5')
    monkeypatch.setenv('RLG_RESUME_ENTROPY_COEF', '0.004')
    old = _bare_agent(config_lr=3e-4)
    old.set_full_state_weights(legacy, set_epoch=True)
    assert old.last_lr == 2e-5
    assert old.entropy_coef == 0.004
    for group in old.optimizer.param_groups:
        assert group['lr'] == pytest.approx(2e-5)

    fresh = _bare_agent(config_lr=3e-4)
    fresh.set_full_state_weights(state, set_epoch=True)
    assert fresh.last_lr == 1e-5
    assert fresh.entropy_coef == 0.007


def _bare_continuous_agent():
    agent = ContinuousA2CBase.__new__(ContinuousA2CBase)
    agent.model = torch.nn.Linear(1, 1)
    agent.vec_env = types.SimpleNamespace(set_train_info=lambda *a, **k: None)
    agent.frame = 0
    agent.normalize_rms_advantage = False
    agent.epochs_between_resets = 0
    agent.is_rnn = False
    agent.ppo_device = 'cpu'
    agent.last_lr = 1e-5
    agent.play_steps = lambda: {'step_time': 0.0, 'played_frames': 8}

    def _reached_update(*args, **kwargs):
        raise RuntimeError('reached update path')

    agent.prepare_dataset = _reached_update
    return agent


def test_first_post_resume_rollout_skips_ppo_update():
    agent = _bare_continuous_agent()
    agent._discard_post_resume_rollout = True
    ret = agent.train_epoch()
    # Rollout collected and counted, update skipped, flag consumed.
    assert agent.curr_frames == 8
    assert agent._discard_post_resume_rollout is False
    assert len(ret) == 11
    assert ret[9] == 1e-5  # last_lr passes through untouched
    # The second epoch proceeds into the update path.
    with pytest.raises(RuntimeError, match='reached update path'):
        agent.train_epoch()