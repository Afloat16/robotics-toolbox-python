## Summary

Correct the requested start/end chain in `manipulability()`, apply the configured Robot tool once, and express Asada inertia in the selected Jacobian's joint coordinates. The present code reverses start/end, omits the ordinary Robot tool, and combines a subchain Jacobian with full-robot inertia.

This changes results for previously misselected chains and omitted tool transforms, and removes the subchain inertia dimension error. Metric formulas and public signatures stay the same. DH sequences already include their tool and retain their existing behavior.

A supplied full-robot Jacobian retains global joint order and its existing override behavior, independently of start/end arguments. A supplied shorter Jacobian uses the selected chain's joint order. Computed Jacobians always use the selected chain order, including reversed paths. The docstring states the distinction.

Contribution owner: @Afloat16. This is an AI-assisted contribution; the required tool attribution is at the end.

## Related issue

Fixes #943. The issue was claimed before implementation.

## Validation

Python 3.12.14, NumPy 2.5.5 and SpatialMath 1.1.19 on Linux, with native fknm/frne extensions.

- The initial 116 selected-chain/tool cases produce **93 failures and 23 passes** on unchanged base `cee00235`. Separately, all **48 DH compatibility controls** and **18 supplied-J compatibility controls** pass on that base.
- Final commit `c4e6e28d0fd748b46b4059faf52ed19f2b0e0048`: the regression file has **194 cases**; together with existing Robot, DHRobot and PoERobot suites, **332 passed, 2 skipped**.
- Geometric metrics are checked against the public Jacobian and determinant/SVD references. Asada tests use inverse dynamics for unit task accelerations to obtain an independent kinetic-energy reference, including reverse joint order, trajectories and nonidentity tools.
- Independent final review verifies **12 full-robot supplied-J selections** exactly match the base and **12 selected-chain/axis combinations** match the inverse-dynamics reference. The preceding DH correction separately passes **24/24** independent tool cases.
- The preceding commit `ad46a757` completes the core run with **2,642 passed, 15 skipped, 12 deselected and 254 subtests passed**. Observation of the final supplied-J revision's complete core rerun was interrupted by loss of the execution environment; its final outcome is unconfirmed. The 332-test result above is for the final code.
- The new test file passes Ruff and formatting. At `ad46a757`, mypy reports exactly the same eight existing block-annotation errors as the base; its final rerun is unconfirmed.

The canonical full invocation was attempted. Both base and candidate hit the existing 50-second dynamics integration timeout. Baseline checks also reproduce the slow symbolic-IK timeout and four notebook animation setup errors caused by unavailable sandbox network interfaces. The core run excludes that integration case and notebook animation file, uses `not slow`, and retains the documented blocks exclusion. No unrestricted full-suite success is claimed.

## Checklist

- [x] Checked for competing work and claimed the issue.
- [x] Conventional Commit title.
- [ ] Unrestricted local test suite passes; limitations and exact tested revisions are stated above.
- [x] Added regression and compatibility tests.
- [x] New test code uses modern type annotations.
- [x] Updated reST docstrings.
- [x] One focused selected-chain computation fix.
- [x] Required versus advisory checks reviewed.
- [x] Numerical behavior changes described.
- [x] Diff reviewed and required assistance disclosed.
- [x] No project-specific data or notebooks added.

AI assistance: ChatGPT (GPT-6 Astra Pro).
