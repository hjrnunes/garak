MCP interception
================

The optional ``midojo-mim`` service speaks ``midojo-binding-v1`` over one
WebSocket per chain. Configure both ``plugins.injectors.mim.service_url`` and
``plugins.injectors.mim.bind_credential`` for a separately managed service,
or neither to use a local loopback service. The matching environment
variables are ``MIDOJO_INTERCEPTOR_URL`` and
``MIDOJO_INTERCEPTOR_CREDENTIAL``. The service URL accepts a base URL or
an already-suffixed ``/v1/bindings`` URL, with an optional proxy path prefix.
You can also set ``bind_timeout``,
``release_timeout`` and ``loop_timeout`` in the same component namespace.
``loop_timeout`` bounds how long one operation may take on the run's event
loop. The loop cancels a longer operation and waits for its cleanup, and the
caller then sees a timeout, or a capability error while a binding is being
made. An operation that finishes in spite of the cancellation keeps its own
result. A caller stops waiting ``release_timeout`` after ``loop_timeout`` if the
cleanup, or the loop's thread, is still blocked, and cleanup that has not
finished by then is abandoned. Closing the run reports a loop thread that does
not stop within ``loop_timeout``. Keep ``loop_timeout`` above ``bind_timeout``
and ``release_timeout`` so that those bounds apply first. A chain accepts only
tool-output instructions, one MCP face and no host tools. Keep the tool
block's bearer token private; only the target receives it.
Keep the configuration file private.

.. automodule:: garak.injectors.mim
   :members:
