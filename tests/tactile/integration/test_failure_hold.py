"""Failure targets must resist observation drift and reset per episode (CPU)."""

import copy
import json

import numpy as np
import pytest

from contact_policy_server import ContactPolicy


def reference():
    return {
        "estimated_width_on_closing_axis": .02,
        "closed_gel_gap": .0008,
        "contact_ee_pose": [1., 0., .8, 1., 0., 0., 0.],
    }


def observation(step, *, offset=0., episode="episode-A"):
    angle = .08 * step
    quaternion = [np.cos(angle / 2), 0., 0., np.sin(angle / 2)]
    return {
        "episode_id": episode,
        "physics_step": step,
        "sim_time": .01 * step,
        "state": {
            "left_ee_pose": np.array([.1 + offset + .01 * step, .2, .9, *quaternion]),
            "right_ee_pose": np.array([-.2 + offset - .02 * step, -.3, .95, *quaternion]),
            "left_arm_joint_state": np.zeros(6),
            "right_arm_joint_state": np.zeros(6),
            "left_ee_joint_state": [.35 + .01 * step],
            "right_ee_joint_state": [.42 + .02 * step],
        },
    }


def expected_hold(obs):
    return {
        key: np.asarray(obs["state"][key], dtype=float).tolist()
        for arm in ("left", "right")
        for key in (f"{arm}_ee_pose", f"{arm}_ee_joint_state")
    }


def fail_by_timeout(policy, *, offset=0., episode="episode-A"):
    policy.update_obs(observation(0, offset=offset, episode=episode))
    policy.get_action()
    assert policy.phase == "approach"
    failed_observation = observation(1, offset=offset, episode=episode)
    policy.update_obs(failed_observation)
    failed_action = policy.get_action()[0]
    assert policy.phase == "failed"
    assert failed_action == expected_hold(failed_observation)
    return failed_action


@pytest.fixture
def policy(tmp_path):
    value = ContactPolicy(tmp_path / "policy", "disabled", reference(),
                          max_phase_actions=1, gripper_settle_observations=1)
    value.reset()
    try:
        yield value
    finally:
        value.close()


def test_failed_target_stays_fixed_across_pose_rotation_and_gripper_drift(policy):
    frozen = fail_by_timeout(policy)
    for step in range(2, 12):
        obs = observation(step)
        assert expected_hold(obs) != frozen
        policy.update_obs(obs)
        assert policy.get_action()[0] == frozen
        assert policy.get_action()[0] == frozen  # Duplicate action reads also hold.
    events = [json.loads(line) for line in (policy.output / "policy_trace.jsonl").read_text().splitlines()]
    snapshots = [event for event in events if event["event"] == "failure_hold_snapshot"]
    assert len(snapshots) == 1
    assert snapshots[0]["action"] == frozen
    assert snapshots[0]["episode_id"] == "episode-A"
    assert snapshots[0]["physics_step"] == 1
    assert snapshots[0]["following_action_index"] == 2


def test_mutating_returned_action_or_observation_cannot_change_failed_hold(policy):
    action = fail_by_timeout(policy)
    frozen = copy.deepcopy(action)
    action["left_ee_pose"][0] += 5
    action["right_ee_joint_state"][0] = 0
    policy.current_observation["state"]["right_ee_pose"][0] += 3
    assert policy.get_action()[0] == frozen
    policy.update_obs(observation(5))
    assert policy.get_action()[0] == frozen


def test_reset_captures_new_failure_state_and_trace_once_per_episode(policy):
    previous = fail_by_timeout(policy)
    policy.update_obs(observation(5))
    assert policy.get_action()[0] == previous
    policy.reset()
    current = fail_by_timeout(policy, offset=.04, episode="episode-B")
    assert current != previous
    policy.update_obs(observation(8, offset=.09, episode="episode-B"))
    assert policy.get_action()[0] == current
    events = [json.loads(line) for line in (policy.output / "policy_trace.jsonl").read_text().splitlines()]
    snapshots = [event for event in events if event["event"] == "failure_hold_snapshot"]
    assert [event["episode_id"] for event in snapshots] == ["episode-A", "episode-B"]
    assert snapshots[0]["action"] == previous
    assert snapshots[1]["action"] == current
