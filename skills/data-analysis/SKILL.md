---
name: data-analysis
description: Analyze tabular data (CSV/TSV/spreadsheets): profile, aggregate, chart with matplotlib, present findings. Use for 'analyze this data', 'plot X', or questions over a data file.
dependencies_files: [tools/tool_use_skill.py, tools/tool_run_command.py, tools/tool_read_file.py, tools/tool_render_files.py, tools/tool_edit_file.py]
---

# Data analysis

Numbers come from executed code, never from eyeballing rows in context.

## Procedure

1. **Profile first.** `read_file` the first ~30 lines for delimiter,
   header, and types; then run a profiling snippet via `run_command`
   (python: row count, columns, nulls, min/max of numerics). Pandas if
   installed; csv + statistics from the stdlib otherwise.
2. **One question, one script.** Write the analysis as a small script via
   `edit_file` and execute it — don't pipe long one-liners; scripts are
   re-runnable and debuggable. Print results as small aligned tables.
3. **Charts:** matplotlib with savefig to a file in the same directory
   (no plt.show() — there is no display). Show it with `render_files`.
   Label axes and units; one chart per claim.
4. **Findings before mechanics** in the reply: lead with the answer
   ("March revenue dropped 18%"), then the supporting table/chart, then
   caveats (nulls dropped, outliers, sample size).

## Pitfalls

- Encoding and locale bite first: try utf-8 then latin-1; watch for
  comma-as-decimal in European data.
- Sanity-check magnitudes against the profile (a sum smaller than the max
  means the parse went wrong).
- Never overwrite the source data file; derived outputs get new names.
