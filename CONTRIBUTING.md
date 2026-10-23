# Contributing

Issues and pull requests in Chinese or English are welcome. Include the robot,
scene, task, layout seed, method configuration, expected behavior and observed
behavior when reporting a simulation issue. Remove API keys and private experiment
content before sharing logs or videos.

Follow the platform instructions in [docs/INSTALL.md](docs/INSTALL.md), then run:

```sh
npm run test:launcher
npm run test:ui
npm test
npm run build
```

The regression suite uses real MuJoCo and local HTTP fixtures. It does not call
paid models. FFmpeg and ffprobe are needed for video checks. The optional
`tests/live_agent_smoke.py` consumes your Codex account quota when explicitly run.

Keep simulation state, model output, control execution and evaluator results
distinct. Missing provider usage or reasoning data must remain missing rather
than being estimated as zero. Keep regression tests tied to observable behavior.

Project source is MIT licensed. Robot assets and policy weights retain their
upstream licenses; see [MODEL_SOURCES.md](MODEL_SOURCES.md). Do not submit vendor
assets, weights, credentials, local method snapshots, or private experiment data.
