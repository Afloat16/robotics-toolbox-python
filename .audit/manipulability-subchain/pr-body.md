## Summary

Correct the requested start/end chain in `manipulability()`, apply the configured Robot tool once, and express Asada inertia in the selected Jacobian's joint coordinates. The present code reverses start/end, omits the ordinary Robot tool, and combines a subchain Jacobian with full-robot inertia.

This changes results for previously misselected chains and omitted tool transforms, and removes the subchain inertia dimension error. Metric formulas and public signatures stay the same. DH sequences already include their tool and retain their existing behavior.

A supplied full-robot Jacobian retains global joint order and its existing override behavior, independently of start/end arguments. A supplied shorter Jacobian uses the selected chain's joint order. Computed Jacobians always use the selected chain order, including reversed paths. The docstring states the distinction.

Contribution owner: @Afloat16.

## Related issue

Fixes #943. The issue was claimed before implementation.

## Validation

[Native baseline/candidate validation](https://github.com/Afloat16/robotics-toolbox-python/actions/runs/43961512696) compares unchanged base `cee0023535988e4ae9490c81d639a46f2311b8fd` with exact final source `c4e6e28d0fd748b46b4059faf52ed19f2b0e0048`. Both compiled extensions are verified present.

- The complete new 194-case regression file reports **103 failed, 91 passed** on the unchanged base and **194 passed** on the final source.
- The repository's canonical core command passes on both: base **2,479 passed, 19 skipped, 11 deselected, 254 subtests passed**; candidate **2,673 passed, 19 skipped, 11 deselected, 254 subtests passed**. The command retains the project's blocks exclusion and `not slow` selection. No additional dynamics or notebook exclusions were needed on these runners.
- Geometric metrics are checked against the public Jacobian and determinant/SVD references. Asada tests use inverse dynamics for unit task accelerations to obtain an independent kinetic-energy reference, including reverse joint order, trajectories and nonidentity tools.
- Compatibility cases cover DH tools and explicit full-width Jacobians with start/end selections. A separate final source review also checked the full-J global-order contract.
- Ruff and formatting pass for the new test file.
- Advisory mypy reports the same **8 existing errors in 2 block modules**, checking 165 source files, on both base and candidate. Error messages match after normalizing source line numbers; there are no new type errors.

The successful core run is for the final source, including the supplied-J compatibility correction. It does not claim an unrestricted slow/block suite or a full operating-system matrix.

## Checklist

- [x] Checked for competing work and claimed the issue.
- [x] Conventional Commit title.
- [x] The canonical core test command passes on the final source; its standard exclusions are stated above.
- [x] Added regression and compatibility tests.
- [x] New test code uses modern type annotations.
- [x] Updated reST docstrings.
- [x] One focused selected-chain computation fix.
- [x] Required versus advisory checks reviewed.
- [x] Numerical behavior changes described.
- [x] Diff reviewed and required assistance disclosed.
- [x] No project-specific data or notebooks added.

AI assistance: ChatGPT (GPT-6 Astra Pro).
