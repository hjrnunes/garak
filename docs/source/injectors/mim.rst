MCP interception
================

The optional ``midojo-mim`` service speaks ``midojo-binding-v1`` over one
WebSocket per chain. Configure both ``plugins.injectors.mim.service_url`` and
``plugins.injectors.mim.bind_credential`` for a separately managed service,
or neither to use a local loopback service. The matching environment
variables are ``MIDOJO_INTERCEPTOR_URL`` and
``MIDOJO_INTERCEPTOR_CREDENTIAL``. The service URL accepts a base URL or
an already-suffixed ``/v1/bindings`` URL, with an optional proxy path prefix.
You can also set ``bind_timeout`` and
``release_timeout`` in the same component namespace. A chain accepts only
tool-output instructions, one MCP face and no host tools. Keep the tool
block's bearer token private; only the target receives it.
Keep the configuration file private.

.. automodule:: garak.injectors.mim
   :members:
