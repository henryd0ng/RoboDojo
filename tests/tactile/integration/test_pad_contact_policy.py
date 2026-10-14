"""CPU controls for source-pad geometry, travel mapping, and timed press dwell."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from contact_policy_server import (
    PHASES,
    RELEASE_FIRST_PHASES,
    ROOT,
    ContactPolicy,
    build_parser,
    reference_geometry,
    rotation,
    trajectory_options,
)
from test_contact_policy import observation, reference


@pytest.fixture(scope="module")
def pad_audit(tmp_path_factory):
    source = ROOT / "Assets/Robots/x5_pad/ARX.usda"
    if not source.is_file():
        pytest.skip("Source-pad assets are required for the CPU geometry audit")
    site = Path(sys.prefix) / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
    if importlib.util.find_spec("pxr") is None and not list((site / "isaacsim/extscache").glob("omni.usd.libs-*")):
        pytest.skip("USD Python bindings are required for the CPU geometry audit")
    output = tmp_path_factory.mktemp("pad-source-audit") / "geometry.json"
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("check_mount_geometry.py")),
            "--repo",
            str(ROOT),
            "--config",
            str(ROOT / "env_cfg/arx_x5_pad_tactile.yml"),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return output


def pad_reference(audit, task="insert_tubes"):
    report = json.loads(audit.read_text())
    return reference_geometry(
        task,
        ROOT / "Assets/Eval_Layout/RoboDojo/arx_x5/0" / f"{task}_0.json",
        report["gap_m"]["current"]["tcp_link6"],
        audit,
        env_cfg="arx_x5_pad_tactile",
    )


def test_source_geometry_uses_actual_pad_chain_and_original_objects(pad_audit):
    report = json.loads(pad_audit.read_text())
    assert report["robot"] == "dual_x5_pad"
    assert report["mounting"] == "source_fingertip_pads"
    assert report["case_root_bounds"] is None
    assert not any(Path(path).name == "Case.usd" for path in report["sources_sha256"])
    np.testing.assert_allclose(report["gripper_range_m"], [0.001, 0.044], atol=1e-9)
    gap = report["gap_m"]["current"]
    assert gap["closed"]["gel"] == pytest.approx(0.00060199245, abs=1e-8)
    assert gap["open"]["gel"] - gap["closed"]["gel"] == pytest.approx(0.086, abs=1e-8)
    # The source pad's ~63.5 mm finger offset is not the link6-frame TCP.
    assert gap["tcp_link6"][0] == pytest.approx(0.150002106, abs=1e-7)
    assert set(report["objects"]) == {"plug_in_charger", "insert_tubes", "deposit_coin"}


@pytest.mark.parametrize("task", ["plug_in_charger", "insert_tubes", "deposit_coin"])
def test_pad_normalized_command_preserves_thin_gel_compression_limit(tmp_path, pad_audit, task):
    ref = pad_reference(pad_audit, task)
    policy = ContactPolicy(tmp_path, "enabled", ref)
    try:
        q_min, q_max = ref["gripper_range_m"]
        q = q_min + policy.contact_gripper * (q_max - q_min)
        gap = ref["closed_gel_gap"] + 2 * (q - ref["closed_gap_joint_position_m"])
        assert ref["estimated_width_on_closing_axis"] - gap == pytest.approx(0.0002)
        assert policy.reference["trajectory_config"]["requested_total_gel_compression"] == 0.0002
        assert ref["robot"] == "dual_x5_pad"
    finally:
        policy.close()


def test_pad_reference_rejects_audit_from_other_configuration(tmp_path, pad_audit):
    report = json.loads(pad_audit.read_text())
    report["env_cfg_path"] = "/another/config.yml"
    invalid = tmp_path / "wrong-config.json"
    invalid.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="different environment configuration"):
        pad_reference(invalid)


def test_cli_defaults_to_cad_pads_and_three_second_contact():
    args = build_parser().parse_args(
        ["--task", "insert_tubes", "--layout", "/tmp/unused.json", "--output", "/tmp/unused-policy"]
    )
    assert args.env_cfg == "arx_x5_pad_tactile"
    assert args.contact_dwell_seconds == 3.2
    assert trajectory_options(args) == {
        "max_translation": .001, "max_rotation": .005,
        "max_gripper_step": .025, "max_phase_actions": 1000,
    }


def test_pad_reference_rejects_stale_source_hash(tmp_path, pad_audit):
    report = json.loads(pad_audit.read_text())
    report["sources_sha256"][str(ROOT / "Assets/Robots/x5_pad/ARX.usda")] = "0" * 64
    stale = tmp_path / "stale.json"
    stale.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="stale"):
        pad_reference(stale)


def test_contact_dwell_uses_sim_time_and_does_not_consume_convergence_budget(tmp_path):
    policy = ContactPolicy(
        tmp_path,
        "enabled",
        reference(),
        contact_dwell_seconds=3.2,
        settle_observations=1,
        gripper_settle_observations=1,
        max_phase_actions=4,
    )
    try:
        policy.reset()
        policy.phase = "contact"
        pose, command = policy._goal()
        policy.update_obs(observation(pose, command, step=0))
        policy.get_action()
        assert policy.contact_dwell_start_sim_time == 0.0
        # Many repeated actions and new observations at a paused timestamp cannot
        # finish the dwell, and must not exhaust the pose-convergence budget.
        for _ in range(8):
            policy.update_obs(observation(pose, command, step=0))
            policy.get_action()
            assert policy.phase == "contact"
        policy.update_obs(observation(pose, command, step=319))
        policy.get_action()
        assert policy.phase == "contact"
        policy.update_obs(observation(pose, command, step=320))
        policy.get_action()
        assert policy.phase == "retract"
        assert policy.failed_reason is None
    finally:
        policy.close()
    summary = json.loads((tmp_path / "policy_summary.json").read_text())
    assert not summary["tactile_contact_verified"]
    assert not summary["task_success_claimed"]


def test_contact_dwell_restarts_when_pose_leaves_tolerance(tmp_path):
    policy = ContactPolicy(
        tmp_path,
        "enabled",
        reference(),
        contact_dwell_seconds=3.2,
        settle_observations=1,
        gripper_settle_observations=1,
    )
    try:
        policy.reset()
        policy.phase = "contact"
        pose, command = policy._goal()
        policy.update_obs(observation(pose, command))
        policy.get_action()
        shifted = pose.copy()
        shifted[2] += 0.02
        policy.update_obs(observation(shifted, command, step=200))
        policy.get_action()
        assert policy.contact_dwell_start_sim_time is None
        policy.update_obs(observation(pose, command, step=300))
        policy.get_action()
        assert policy.contact_dwell_start_sim_time is None  # Gripper reopened during the descent.
        policy.update_obs(observation(pose, command, step=301))
        policy.get_action()
        assert policy.contact_dwell_start_sim_time == pytest.approx(3.01)
        policy.update_obs(observation(pose, command, step=600))
        policy.get_action()
        assert policy.phase == "contact"
    finally:
        policy.close()


def test_timed_contact_requires_monotonic_sim_time(tmp_path):
    policy = ContactPolicy(tmp_path, "enabled", reference(), contact_dwell_seconds=3.2)
    try:
        policy.reset()
        obs = observation(policy.contact_pose)
        obs.pop("sim_time")
        with pytest.raises(ValueError, match="finite sim_time"):
            policy.update_obs(obs)
        policy.update_obs(observation(policy.contact_pose, step=10))
        with pytest.raises(ValueError, match="backward"):
            policy.update_obs(observation(policy.contact_pose, step=9))
    finally:
        policy.close()


def cli_arguments(env_cfg, *options):
    return build_parser().parse_args(
        [
            "--task",
            "insert_tubes",
            "--env-cfg",
            env_cfg,
            "--layout",
            "/tmp/layout.json",
            "--output",
            "/tmp/policy",
            *options,
        ]
    )




def test_cli_limits_accept_explicit_overrides():
    args = cli_arguments(
        "arx_x5_pad_tactile",
        "--max-translation",
        "0.0005",
        "--max-rotation",
        "0.003",
        "--max-gripper-step",
        "0.01",
        "--max-phase-actions",
        "1200",
    )
    assert trajectory_options(args) == {
        "max_translation": 0.0005,
        "max_rotation": 0.003,
        "max_gripper_step": 0.01,
        "max_phase_actions": 1200,
    }


@pytest.mark.parametrize("flag", ["--max-translation", "--max-rotation", "--max-gripper-step"])
@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_cli_step_limits_reject_nonpositive_and_nonfinite_values(flag, value):
    with pytest.raises(SystemExit):
        cli_arguments("arx_x5_pad_tactile", flag, value)




def test_source_pad_cli_pose_limits_bound_actual_requested_motion(tmp_path):
    limits = trajectory_options(cli_arguments("arx_x5_pad_tactile"))
    policy = ContactPolicy(tmp_path, "enabled", reference(), **limits)
    try:
        policy.reset()
        pose = np.array([0, 0, 1.3, 1, 0, 0, 0], dtype=float)
        policy.update_obs(observation(pose))
        action = policy.get_action()[0]
        commanded = np.asarray(action["left_ee_pose"])
        observed = np.asarray(policy.current_observation["state"]["left_ee_pose"], dtype=float)
        assert np.linalg.norm(commanded[:3] - observed[:3]) == pytest.approx(0.001)
        assert (rotation(commanded) * rotation(observed).inv()).magnitude() == pytest.approx(0.005)
    finally:
        policy.close()


def test_descend_stays_open_until_pose_settles_on_new_observations(tmp_path):
    policy = ContactPolicy(tmp_path, "enabled", reference(), max_translation=0.002, max_gripper_step=0.025)
    try:
        policy.reset()
        policy.phase = "open"
        # Opening settles at clearance before any descent starts.
        for step in range(3):
            policy.update_obs(observation(policy.approach_pose, 1.0, step=step))
            action = policy.get_action()[0]
            assert action["left_ee_joint_state"] == [1.0]
        assert policy.phase == "descend"
        assert action["left_ee_pose"][2] < policy.approach_pose[2]

        near = policy.contact_pose.copy()
        near[2] += 0.002
        policy.update_obs(observation(near, 1.0, step=3))
        # A pose within tolerance still needs three fresh settled observations.
        for _ in range(5):
            action = policy.get_action()[0]
            assert policy.phase == "descend"
            assert action["left_ee_joint_state"] == [1.0]
        policy.update_obs(observation(policy.contact_pose, 1.0, step=4))
        assert policy.get_action()[0]["left_ee_joint_state"] == [1.0]
        assert policy.phase == "descend"
        policy.update_obs(observation(policy.contact_pose, 1.0, step=5))
        action = policy.get_action()[0]
        assert policy.phase == "contact"
        assert action["left_ee_joint_state"][0] == pytest.approx(0.975)
        assert policy.contact_dwell_start_sim_time is None
    finally:
        policy.close()
    events = [json.loads(line) for line in (tmp_path / "policy_trace.jsonl").read_text().splitlines()]
    assert [(event["from"], event["to"]) for event in events if event["event"] == "phase_transition"] == [
        ("open", "descend"),
        ("descend", "contact"),
    ]


def test_gripper_ramp_uses_emitted_commands_and_waits_after_final_goal(tmp_path, pad_audit):
    policy = ContactPolicy(
        tmp_path,
        "enabled",
        pad_reference(pad_audit),
        max_gripper_step=0.025,
        settle_observations=1,
        max_phase_actions=1000,
    )
    try:
        policy.reset()
        policy.phase = "contact"
        policy.update_obs(observation(policy.contact_pose, 1.0))
        previous = 1.0
        step = 0
        while previous > policy.contact_gripper + 1e-12:
            action = policy.get_action()[0]
            command = action["left_ee_joint_state"][0]
            assert 0 < previous - command <= 0.025 + 1e-12
            assert command >= policy.contact_gripper - 1e-12
            assert policy.phase == "contact"
            previous = command
            step += 1
            # An immediate final-goal echo cannot skip the emitted-command ramp.
            policy.update_obs(observation(policy.contact_pose, policy.contact_gripper, step=step))
        assert previous == pytest.approx(policy.contact_gripper)
        final_command_observation = policy.gripper_commands["left"][1]
        while policy.observation_count - final_command_observation < 15:
            assert policy.get_action()[0]["left_ee_joint_state"][0] == pytest.approx(policy.contact_gripper)
            assert policy.phase == "contact"
            step += 1
            policy.update_obs(observation(policy.contact_pose, policy.contact_gripper, step=step))
        policy.get_action()
        assert policy.phase == "retract"
    finally:
        policy.close()


@pytest.mark.parametrize("field", ["max_translation", "max_rotation", "max_gripper_step"])
@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_constructor_step_limits_reject_invalid_values(tmp_path, field, value):
    with pytest.raises(ValueError, match="positive and finite"):
        ContactPolicy(tmp_path, "enabled", reference(), **{field: value})


def test_release_before_retract_is_explicit_in_cli():
    assert not cli_arguments("arx_x5_pad_tactile").release_before_retract
    assert cli_arguments("arx_x5_pad_tactile", "--release-before-retract").release_before_retract


@pytest.mark.parametrize("value", [1, "true", None])
def test_release_before_retract_rejects_nonboolean_constructor_flag(tmp_path, value):
    with pytest.raises(ValueError, match="must be boolean"):
        ContactPolicy(tmp_path, "enabled", reference(), release_before_retract=value)


def test_release_before_retract_opens_at_contact_pose_and_waits_before_lifting(tmp_path):
    policy = ContactPolicy(
        tmp_path,
        "enabled",
        reference(),
        release_before_retract=True,
        max_gripper_step=0.025,
        settle_observations=3,
        gripper_settle_observations=15,
        contact_dwell_seconds=3.2,
        max_phase_actions=1000,
    )
    try:
        assert policy.phases == RELEASE_FIRST_PHASES
        config = policy.reference["trajectory_config"]
        assert config["release_before_retract"] is True
        assert config["phase_sequence"] == list(RELEASE_FIRST_PHASES)
        policy.reset()
        policy.phase = "contact"
        command = policy.contact_gripper
        for step in (0, 1, 2):
            policy.update_obs(observation(policy.contact_pose, command, step=step))
            policy.get_action()
        assert policy.contact_dwell_start_sim_time == pytest.approx(0.02)
        policy.update_obs(observation(policy.contact_pose, command, step=321))
        assert policy.get_action()[0]["left_ee_joint_state"] == [command]
        assert policy.phase == "contact"
        policy.update_obs(observation(policy.contact_pose, command, step=322))
        action = policy.get_action()[0]
        assert policy.phase == "release"
        np.testing.assert_allclose(action["left_ee_pose"], policy.contact_pose, atol=1e-7)
        assert command < action["left_ee_joint_state"][0] <= command + 0.025 + 1e-12
        step = 323
        while policy.phase == "release":
            previous = action["left_ee_joint_state"][0]
            policy.update_obs(observation(policy.contact_pose, previous, step=step))
            action = policy.get_action()[0]
            assert previous <= action["left_ee_joint_state"][0] <= previous + 0.025 + 1e-12
            if policy.phase == "release":
                np.testing.assert_allclose(action["left_ee_pose"], policy.contact_pose, atol=1e-7)
            step += 1
            assert step < 450
        assert policy.phase == "retract"
        assert policy.observation_count - policy.gripper_commands["left"][1] >= 17
        assert action["left_ee_joint_state"] == [1.0]
        assert action["left_ee_pose"][2] > policy.contact_pose[2]
        pose, command = policy._goal()
        np.testing.assert_array_equal(pose, policy.approach_pose)
        assert command == 1.0
    finally:
        policy.close()
    events = [json.loads(line) for line in (tmp_path / "policy_trace.jsonl").read_text().splitlines()]
    transitions = [(event["from"], event["to"]) for event in events if event["event"] == "phase_transition"]
    assert transitions == [("contact", "release"), ("release", "retract")]


@pytest.mark.parametrize("max_gripper_step", [None, 0.025])
@pytest.mark.parametrize("release_first", [False, True])
@pytest.mark.parametrize("explicit_none", [False, True])
def test_release_step_none_preserves_legacy_actions_on_same_observations(
    tmp_path, monkeypatch, max_gripper_step, release_first, explicit_none
):
    options = {"max_gripper_step": max_gripper_step, "release_before_retract": release_first,
               "settle_observations": 100}
    legacy = ContactPolicy(tmp_path / "legacy", "enabled", reference(), **options)
    current = ContactPolicy(
        tmp_path / "current", "enabled", reference(), **options,
        **({"max_release_gripper_step": None} if explicit_none else {}),
    )

    def legacy_gripper_command(arm, goal):
        if legacy.max_gripper_step is None:
            return goal
        previous = legacy.gripper_commands[arm][0]
        return float(previous + np.clip(goal - previous, -legacy.max_gripper_step, legacy.max_gripper_step))

    monkeypatch.setattr(legacy, "_bounded_gripper_command", legacy_gripper_command)
    try:
        legacy.reset()
        current.reset()
        for index, phase in enumerate(legacy.phases):
            legacy.phase = current.phase = phase
            pose, _ = legacy._goal()
            obs = observation(pose, 0.55, step=index)
            legacy.update_obs(obs)
            current.update_obs(obs)
            assert legacy.get_action() == current.get_action()
            assert legacy.phase == current.phase
            assert legacy.gripper_commands == current.gripper_commands
        assert current.max_release_gripper_step is None
    finally:
        legacy.close()
        current.close()


@pytest.mark.parametrize("release_step,expected", [(0.0125, 0.0125), (0.1, 0.025)])
def test_release_opening_uses_smaller_step_without_enlarging_original_bound(tmp_path, release_step, expected):
    policies = [
        ContactPolicy(tmp_path / name, "enabled", reference(), max_gripper_step=0.025,
                      max_release_gripper_step=step, release_before_retract=True)
        for name, step in (("original", None), ("override", release_step))
    ]
    try:
        actions = []
        for policy in policies:
            policy.reset()
            policy.phase = "release"
            policy.update_obs(observation(policy.contact_pose, 0.5))
            actions.append(policy.get_action()[0])
        original, overridden = actions
        assert original["left_ee_joint_state"] == pytest.approx([0.525])
        assert overridden["left_ee_joint_state"] == pytest.approx([0.5 + expected])
        assert overridden["left_ee_pose"] == original["left_ee_pose"]
        assert overridden["right_ee_pose"] == original["right_ee_pose"]
        assert overridden["right_ee_joint_state"] == original["right_ee_joint_state"] == [1.0]
        assert policies[1].reference["trajectory_config"]["effective_release_gripper_step"] == expected
    finally:
        for policy in policies:
            policy.close()


@pytest.mark.parametrize("phase", [phase for phase in PHASES if phase != "release"])
@pytest.mark.parametrize("release_first", [False, True])
def test_release_step_preserves_closing_and_other_phase_actions(tmp_path, phase, release_first):
    policies = [
        ContactPolicy(tmp_path / name, "enabled", reference(), max_gripper_step=0.025,
                      max_release_gripper_step=step, release_before_retract=release_first)
        for name, step in (("original", None), ("override", 0.0125))
    ]
    try:
        actions = []
        for policy in policies:
            policy.reset()
            policy.phase = phase
            pose, _ = policy._goal()
            policy.update_obs(observation(pose, 0.8))
            actions.append(policy.get_action())
        assert actions[0] == actions[1]
    finally:
        for policy in policies:
            policy.close()


@pytest.mark.parametrize("max_gripper_step", [None, 0.025])
def test_release_only_bound_does_not_limit_a_closing_command(tmp_path, max_gripper_step):
    policy = ContactPolicy(tmp_path, "enabled", reference(), max_gripper_step=max_gripper_step,
                           max_release_gripper_step=0.0125)
    try:
        policy.phase = "release"
        policy.gripper_commands["left"] = (0.8, 0)
        assert policy._bounded_gripper_command("left", 0.1) == pytest.approx(
            0.1 if max_gripper_step is None else 0.775
        )
        assert policy._bounded_gripper_command("left", 1.0) == pytest.approx(0.8125)
    finally:
        policy.close()


@pytest.mark.parametrize("value", [True, False, np.bool_(True), 0, -1, np.nan, np.inf, -np.inf,
                                    [], [0.01], {}, "invalid", 1 + 1j])
def test_release_step_constructor_rejects_invalid_values(tmp_path, value):
    with pytest.raises(ValueError, match="max_release_gripper_step.*positive and finite"):
        ContactPolicy(tmp_path, "enabled", reference(), max_release_gripper_step=value)


def test_release_step_cli_default_and_override_preserve_other_limits():
    default = cli_arguments("arx_x5_pad_tactile")
    overridden = cli_arguments("arx_x5_pad_tactile", "--max-release-gripper-step", "0.0125")
    assert default.max_release_gripper_step is None
    assert overridden.max_release_gripper_step == 0.0125
    assert trajectory_options(default) == trajectory_options(overridden)


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "-inf", "invalid", "true"])
def test_release_step_cli_rejects_invalid_values(value):
    with pytest.raises(SystemExit):
        cli_arguments("arx_x5_pad_tactile", "--max-release-gripper-step", value)
