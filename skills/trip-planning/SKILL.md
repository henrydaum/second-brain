---
name: trip-planning
description: Plan a trip: destination research, day-by-day itinerary, budget table, packing notes — saved as a document. Use for 'plan a trip to X', 'weekend in Y', or travel comparisons.
dependencies_files: [tools/tool_use_skill.py, tools/tool_web_search.py, tools/tool_edit_file.py, tools/tool_ask_user_question.py]
---

# Trip planning

A good itinerary balances must-sees against the human need to not be
scheduled at 30-minute granularity on vacation.

## Procedure

1. **Constraints first** (`ask_user_question`, one round): dates, origin,
   budget level, travel party, pace (packed vs relaxed), interests, any
   fixed points (a wedding, a conference).
2. **Research with recency**: `web_search` for the destination's current
   state — opening hours, seasonal closures, safety advisories, local
   events during the dates. Fetch official sites for anything
   reservation-critical (URL as query fetches the page).
3. **Itinerary shape**: one anchor activity per day + clustered nearby
   options, geographically grouped to minimize transit. Mark which need
   advance booking and when booking opens. Mornings for popular sites,
   reserve a free half-day per 3 days.
4. **Budget table**: lodging / transport / food / activities per day with
   a realistic range, currency stated. Mark estimates as estimates.
5. Save as `trips/<destination>-<dates>.md` via `edit_file`; reply with
   the day-by-day skeleton and the booking-deadline list.

## Pitfalls

- Prices and hours from search snippets are stale by default — verify the
  ones that can ruin a day (closures, last entry times).
- Check day-of-week reality: many museums close Mondays.
