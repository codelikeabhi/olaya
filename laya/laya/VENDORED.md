# Vendored Laya

This directory is Laya 0.3.5 (https://github.com/NandhaKishorM/laya, commit 573e5b6,
tracked via the fork https://github.com/codelikeabhi/laya), Apache License 2.0
(`LICENSES/laya-Apache-2.0.txt`). Copyright Convai Innovations and Laya contributors.

It is vendored because Olaya modifies the decision model itself: calibration (defect D1),
training on Apple Silicon, and architecture changes (see `docs/` in the planning workspace).
Olaya's changes are recorded in git history on top of the verbatim import. The import
commit contains the upstream files unmodified.
