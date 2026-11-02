# Workbench candidate API (MuJoCo / Panda)

The platform applies JSON `files` replacements and runs all checks. The coding
model does not need shell access, filesystem write permission or direct rollout
tools. Upstream instructions to edit files, inspect a repository, run shell
commands or use LIBERO entrypoints are translated by the host into this contract:
return a complete replacement of the allowed `program.py`, with `def run(ctx)`.
Never claim that checks have run; the host runs them after applying the response.
Read-only diagnosis means return `files: []`; edit requests must return changed code.
If genuinely blocked, explain the missing capability; the host stops rather than
rerunning unchanged code. In automatic mode the host retrieves or generates a
program for the selected task. Advanced fixed-program mode executes the supplied
source unchanged, so changing its task selector alone does not rewrite that code.

## Configured models and tools

`ctx.capabilities` describes the configured action model and named tools (without
credentials). Configuration is optional; missing services are not installed or
silently substituted. Use the host methods instead of implementing HTTP clients:

- `ctx.policy(instruction)` calls the configured VLA/WAM with the current public
  observation and returns an `embodied-policy-v1` output. It does not move the robot.
- `ctx.execute_policy(output)` validates and executes that output through the same
  bounded native control as `move_tcp`. Supported action types are `tcp_pose`
  (absolute world pose) and `actuator_targets` (explicit `joint_names`, `actions`,
  and `dt_s`). Model `done`/`blocked` is not an executable action or proof of success.
- `ctx.tool(name, instruction, payload=None)` invokes a configured perception or
  decision service and returns its `result`. The caller chooses when to re-observe.

If an action model is selected, each successful episode must have called it and
executed a policy output; a pure IK program cannot count as a VLA/WAM experiment.
Exceptions preserve service errors for the evolution loop. Calls share the episode
wall-clock budget and can be cancelled. API keys stay in the parent server process.

The action service uses the existing [policy contract](../server/policy.py).
Tool services receive a POST with an optional Bearer authorization header:

```json
{"protocol":"embodied-tool-v1","model":"my-vision-model","task":"Locate the red block",
 "observation":{"cameras":[],"tcp":{},"joints":[]},"input":{"target":"red block"}}
```

`observation` is the full current public `ctx.observe()` output, including images
and calibration; the abbreviated object above only shows its shape. Return:

```json
{"protocol":"embodied-tool-v1","result":{"pixel":[320,240],"confidence":0.9},
 "usage":{"input_tokens":100,"output_tokens":20}}
```

The configured description must document the tool's result schema and coordinate
frame. The host records returned output and available usage; it cannot invent
unreported usage or local GPU costs. Evaluator-only truth is never sent to services.

## Observations

`ctx.budget()` returns `actions_used`, `actions_remaining`, `sim_time_s`,
`sim_remaining_s`, and `wall_remaining_s`. Check it when planning phases and retries.
The configured budget is shared by grasp, transport, release, retreat and settling.
`move_tcp` may include extra physical settling after its requested `duration_s`;
use the measured simulation time, not a sum of requested durations. Reserve time
and actions for release/retreat and the native success hold. Budget exhaustion is
failure feedback for the next program revision, not a reason to assert completion.

`ctx.observe()` returns a dict at the current simulation time. It does not advance
physics and does not contain object positions, segmentation, depth or task scores.
The scene starts paused at time zero. Loose objects can settle under gravity during
the first movement. Re-localize from current images before grasping; an old visual
estimate may no longer be valid, even if the robot has not touched the object.
Use Python standard library, NumPy and Pillow (installed in the episode runtime)
to process RGB pixels, or connect an actual external perception service explicitly.
Do not assume pi0.5, SAM3 or other model services are installed.

- `time`: simulation seconds; `generation`: scene generation.
- `tcp`: `frame: 'world'`, `position_m: [x,y,z]`,
  `quaternion_wxyz: [w,x,y,z]`. Coordinates are world Z-up, in metres.
  For Panda the TCP is the centre between the fingertips (`studio_tcp` site), not
  the wrist link origin. Gripper local +Z points along the fingers; a downward
  grasp points local +Z towards world -Z. Use gradual rotations, not a sudden flip.
- `gripper`: `position` in metres per finger, `target_open_fraction` in [0,1].
- `joints`: list of `name`, `position_rad`, `velocity_rad_s`, `limits_rad`.
- `limits`: `translation_per_round_m`, `rotation_per_round_deg`,
  `position_tolerance_m`, `orientation_tolerance_rad`.
- `cameras`: three dicts named `front`, `side`, `wrist`, each with
  `image_base64` (JPEG bytes encoded as base64), `width`, `height`, `time`,
  `position_world_m`, `rotation_camera_to_world` (3x3), `fovy_deg`.
  Decode with `PIL.Image.open(io.BytesIO(base64.b64decode(camera['image_base64'])))`.
  Camera local axes: +X right, +Y up, view direction -Z; image pixels: X right,
  Y down. For a pixel (u,v), local ray is approximately
  `[(u+0.5-width/2)/f, -(v+0.5-height/2)/f, -1]`, where
  `f = height / (2*tan(radians(fovy_deg)/2))`. Multiply by
  `rotation_camera_to_world` to get a world ray from `position_world_m`.
  Two fixed calibrated views can triangulate a visible point; reject inconsistent
  rays and occlusions. RGB alone does not provide a depth map. Avoid hardcoded
  object coordinates or seed-specific solutions.

## Actions and feedback

`ctx.move_tcp(pose)` executes one bounded absolute TCP target and advances physics.
Required pose fields: `status: 'move'`, `frame: 'world'`, `position_m`,
`quaternion_wxyz`, `gripper_open` (0 closed, 1 open), `duration_s` (0.5–4.0),
`explanation` (string). Read current feedback before the next target.
Translation is limited to 0.12 m and rotation to 35 degrees per call, with an
additional IK joint-change limit of 0.65 rad; split movements into smaller targets.
TCP Z must be at least 0.008 m. Rejected IK targets raise ValueError without moving.
The returned dict has `execution`, `motion`, `tcp`; `motion.status` may be
`complete` or `failed`, with position/orientation errors and a message.
Failed motion is not task success. Use smaller/reoriented moves when needed.
Gripper closure requires simulation time; observe actual finger position and RGB.
To hold a grasp, command the current TCP with the gripper still closed and a
positive duration. `time.sleep()` and `ctx.observe()` do not advance physics.

`ctx.note(text)` saves a diagnostic message. `ctx.seed` and `ctx.task` identify the
episode. Budgeted move calls include failed IK attempts. Do not access private
attributes, simulator internals, checkpoint files or evaluator state. The host
evaluates completion after running the program; candidate returns do not set it.
