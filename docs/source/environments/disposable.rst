Disposable environment
======================

Use this component when you, the operator, give every chain a target that no
earlier chain has touched, and you want a probe to run repeated chains or the
baseline of a paired run.

Select it with ``plugins.environment_type: disposable``.

The component declares isolation, and the declaration is yours. Garak never
verifies it: it cannot see your target. You make the declaration true by
replacing the target before each chain, outside garak, for example with a new
container or a freshly seeded service. If you declare isolation and do not
provide it, a repeated chain or a baseline meets the state that an earlier
chain left. A comparison that you draw from such chains is then not valid, and
garak cannot tell you so.

The component has no reset capability, because a replaced target needs no
reset, and it refuses a reset. It records no observations. Configuration cannot
change what it declares. The ``unmanaged`` component still refuses any declared
isolation, so it still refuses repeated chains and baselines.

.. automodule:: garak.environments.disposable
   :members:
