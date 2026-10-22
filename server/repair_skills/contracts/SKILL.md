---
name: provider-contract-repair
description: Preserve SDK and robot action contracts while repairing adapters.
---

Use the existing official SDK adapters. Preserve all three synchronized cameras and the schema selected by response_type. Validate the actual JSON instance, not a JSON Schema or wrapper. Treat malformed or truncated output as an error; never make the schema permissive to get a pass. Do not change physical bounds, IK, actuator behavior, success criteria or experiment history.

Preserve returned usage on all failure paths. Keep native provider reasoning parameters and cache counters intact. Do not infer third-party endpoint capabilities from model names. Do not retry, switch providers, disable TLS verification or persist credentials as a repair.
