# Runtime source

- `entry`: dispatches the requested instrument to `backends`.
- `recorder`: shared expert controller, feedback gates, robot reset, scene spawn integration,
  coverage accounting, storage transactions and logs.
- `ui`: panel state/widgets, commands and previews.

Root modules are compatibility shims loaded by `_p4_compat.py`; import names stay stable.
They execute these sources in the original module namespace, so patching a module variable
still affects the functions that use it. `__file__` retains the legacy resource root while
tracebacks refer to the real implementation file. Use `scripts/p4.py` to run commands.

Dependency flow:

```text
panel / scripts CLI -> entry runner -> instrument backend
                                      |
                                      +-> env + assets
                                      +-> shared recorder hooks and physical QC
                                      +-> storage transaction -> raw diagnostic H5

completed-case manifest -> training export -> sensor-only policy files + separate labels
                         -> training/model -> runtime RGB/proprio API
```

Expert/QC dependencies may access simulator object geometry. Neural-network input code is
under `training` and must not import the expert geometry/state builders. Its live sensor
reader uses only robot FK and camera RGB.
