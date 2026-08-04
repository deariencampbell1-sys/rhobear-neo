"""rhobear-neo — the review-and-merge gate. Sibling service to rhobear-reviews.

Wakes on a trusted reviewer's verdict (GitHub `status` / `check_run` event), drives
the PR to green (fix-forward trivial via DeepSeek Flash; dispatch a DeepSeek-Pro
builder for substantial bugs), and merges on green behind the per-install auto-merge
toggle. Priced on the RHOBEAR credits model (Gemini baseline + margin).
"""
__version__ = "0.1.0"
