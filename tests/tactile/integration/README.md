# CAD X5: three contact demos

The simulator always runs through `scripts/robodojo.sh client`. The small reference
server sends ordinary EE actions through the native `demo_policy` adapter. It
uses known asset geometry and demonstrates contact/release, not task success.

Run from the RoboDojo root. In **both terminals**, activate the existing environment
and set the same task and output directory. Use a new `demo-01` identifier for each rerun:

```bash
conda activate robodojo-tactile
TASK=plug_in_charger
RUN="$PWD/.cache/tactile/demo-01/$TASK"
```

| TASK | Target | Action cap | Gripper increment | Contact Z offset |
| --- | --- | ---: | ---: | ---: |
| `plug_in_charger` | charger | 320 | 0.025 | 0 m |
| `insert_tubes` | tube1 | 350 | 0.025 | 0 m |
| `deposit_coin` | coin0 | 300 | 0.05 | 0.012 m |

## Terminal 1: reference controller

```bash
GRIPPER_STEP=0.025
CONTACT_Z=0
if [ "$TASK" = deposit_coin ]; then
  GRIPPER_STEP=0.05
  CONTACT_Z=0.012
fi
python tests/tactile/integration/contact_policy_server.py \
  --task "$TASK" --env-cfg arx_x5_pad_tactile \
  --layout "Assets/Eval_Layout/RoboDojo/arx_x5/0/${TASK}_0.json" \
  --contact-offset 0 0 "$CONTACT_Z" \
  --max-translation 0.004 --max-rotation 0.04 --max-gripper-step "$GRIPPER_STEP" \
  --contact-dwell-seconds 3.2 --release-before-retract \
  --port 19091 --output "$RUN/policy"
```

Wait for the server to listen. It measures the CAD geometry on CPU first.
The sequence is approach → open → descend → contact → release → retract → hold.
The **contact dwell is 3.2 simulated seconds**; `hold` is the final empty-gripper phase.

## Terminal 2: native simulator

```bash
case "$TASK" in
  plug_in_charger) MAX_ACTIONS=320 ;;
  insert_tubes) MAX_ACTIONS=350 ;;
  deposit_coin) MAX_ACTIONS=300 ;;
esac
ROBODOJO_RUN_ID="demo-01-$TASK" \
ROBODOJO_EVAL_OUTPUT_ROOT="$RUN/native" \
ROBODOJO_EVAL_MAX_ACTIONS="$MAX_ACTIONS" \
ROBODOJO_MAX_BASH_RETRIES=1 \
  bash scripts/robodojo.sh client \
  --task "$TASK" --env-cfg arx_x5_pad_tactile \
  --policy-name demo_policy --policy-host 127.0.0.1 --policy-port 19091 \
  --eval-num 1 --action-type ee --num-envs 1 --seed 0 --ckpt contact_demo
```

Add `--gui` to the client command for a window. After the client finishes, stop
the server with Ctrl-C. Repeat with the other tasks. The action caps preserve
the accepted trajectories and allow dwell/release within the native task budget.
No checkpoint or model API is required.

## Videos and results

The evaluator writes a `*_visuotactile_*.mp4` containing the three cameras and
two tactile streams. A `fail` filename can be normal: these trajectories do not
attempt the complete insertion task. Inspect contact and release in the video.
The server keeps a compact JSON summary/JSONL trace; it does not save image NPZ.

To export one run, after successful client shutdown:

```bash
DEST="$PWD/outputs/tactile/demo-01"
mkdir -p "$DEST"
VIDEO=$(rg --files "$RUN/native" | rg '_visuotactile_(success|fail)\.mp4$')
RESULT=$(rg --files "$RUN/native" | rg '/_result\.json$')
cp "$VIDEO" "$DEST/$TASK.mp4"
cp "$RESULT" "$DEST/$TASK.json"
```

After exporting all three runs, concatenate them with ffmpeg (no simulation rerun):

```bash
printf "file 'file:%s'\n" \
  "$DEST/plug_in_charger.mp4" "$DEST/insert_tubes.mp4" "$DEST/deposit_coin.mp4" \
  | ffmpeg -n -protocol_whitelist file,pipe,fd -f concat -safe 0 -i - \
      -c copy "$DEST/3task.mp4"
```

Keep logs and geometry records in `.cache`; only MP4 and corresponding result
JSON need to be copied to `outputs`. The two existing accepted/reproduced video
directories are preserved and are not destinations for new runs.
