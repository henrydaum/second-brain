"""The native face of the five plugin families.

Nothing in here is a plugin, and nothing subclasses these by hand. Every
plugin is sandboxed code; ``sandbox.bridge`` reads the file and builds a
subclass of one of these classes whose ``run`` forwards into a box. To the
tool registry, the orchestrator, the command registry and the frontend
manager, the adapter is an ordinary plugin — these classes are what makes
that true.

They are not the retired plugin contract. They were called
``plugins/BaseTool.py`` and friends while a plugin could still be written
against them directly, and the filename went on saying "this is how you write
a tool" long after that stopped being possible. What each file holds is the
half the *kernel* needs: the result types it reads, the declarations it
schedules and routes from, and — in :mod:`~plugins.native.frontend` — the
host-side bus routing the guest deliberately does not own.

**Import from the submodule, not from here**, and note this package
deliberately re-exports nothing. The five differ enormously in what they drag
in: ``tool`` and ``task`` are standalone, ``command`` needs
``state_machine.conversation`` for ``FormStep``, and ``frontend`` reaches the
bus and the runtime. Re-exporting them together means importing the lightest
costs you the heaviest — which is not merely wasteful, it is a cycle:
``agent.tool_registry`` wants ``BaseTool``, and pulling ``command`` alongside
it routes back through ``state_machine`` into ``agent.tool_registry`` before
it has finished defining ``ToolRegistry``.
"""
