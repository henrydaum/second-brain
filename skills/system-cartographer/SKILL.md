---
name: system-cartographer
description: Explain or diagram how a codebase or system fits together: trace entry points, map dependencies, produce Mermaid diagrams. Use for 'how does X work', 'diagram the architecture', onboarding docs.
dependencies_files: [tools/tool_use_skill.py, tools/tool_read_file.py, tools/tool_run_command.py, tools/tool_edit_file.py]
---

# System cartographer

Maps are claims about structure. Make them from traced evidence, render
them small, and say what you left out.

## Procedure

1. **Find the spine**: entry points (main, CLI, service start), then trace
   one real request/flow end to end with `read_file` before generalizing.
   `run_command` grep for imports/symbols beats guessing relationships.
2. **Choose altitude deliberately**: a component map (5-10 boxes) OR a
   sequence for one flow OR a data-model sketch. One diagram per question;
   resist the everything-map.
3. **Mermaid in markdown** via `edit_file`:
   - flowchart TD for components/dependencies
   - sequenceDiagram for a request flow
   - erDiagram for tables/relations
   Keep under ~12 nodes; label edges with the verb ("emits", "imports",
   "writes"). Cluster with subgraphs rather than adding nodes.
4. **Annotate the map** with the 3-5 sentences a newcomer needs: where
   things start, the one rule of the architecture, where the bodies are
   buried.

## Pitfalls

- Folder structure is not architecture; map the runtime relationships.
- Mark uncertainty on the diagram (dashed edge, "?") instead of drawing
  confident fiction.
- Diagrams rot: date them and name the commit/version they describe.
