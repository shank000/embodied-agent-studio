---
name: provider-error-triage
description: Diagnose a model call failure in this embodied workbench.
---

Trace the original request, provider response, validation and control boundary. Cite the exact returned error or source line. Distinguish credentials, rate limits, networking, unsupported settings, truncation, malformed output, missing weights and physical IK rejection. A failed physical action does not imply a failed model API. Do not invent missing usage or reasoning.

Choose configuration advice when no code defect is evidenced. A length stop means truncation; it is not evidence that a model lacks vision. A live experiment may explicitly authorize bounded recovery. In that case, return recovery.action=retry with a concise corrective instruction or an evidenced settings correction for a pre-execution failure. This is a new billed inference, not a repaired or executed prior action. Missing credentials and code requiring restart need recovery.action=stop. Never silently retry outside that authorization, change models or omit required camera images. State what evidence is still missing.
