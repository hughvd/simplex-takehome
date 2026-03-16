---
name: Do not run training
description: User runs training in Colab, not locally — only write code files
type: feedback
---

Do not run training or long-running experiments. User will open a Colab kernel and run notebooks there.

**Why:** User prefers to control execution in their own environment (Google Colab).

**How to apply:** Write all code files (.py, .ipynb) but do not execute training loops, model evaluation, or analysis scripts. Quick sanity checks (imports, shape checks) are fine if needed for verification.
