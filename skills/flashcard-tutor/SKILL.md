---
name: flashcard-tutor
description: Create spaced-repetition flashcards from the user's notes or a topic, store decks as files, and quiz on request or schedule. Use for 'make flashcards', 'quiz me', 'help me memorize X'.
dependencies_files: [tools/tool_use_skill.py, tools/tool_edit_file.py, tools/tool_read_file.py, tools/tool_hybrid_search.py, tools/tool_schedule_subagent.py]
---

# Flashcard tutor

Cards live as markdown files; the agent is the scheduler and the quizmaster.

## Deck format

`decks/<topic>.md` in a sync directory, one card per block:

    ## Q: <question>
    A: <answer>
    <!-- box:2 last:2026-06-10 -->

box is the Leitner box (1=daily, 2=every 3 days, 3=weekly); new cards
start at box 1.

## Creating decks

1. Source material: the user's own notes via `hybrid_search`/`read_file`
   beats generated trivia — cards should test what THEY need to know.
2. Write atomic cards: one fact per card, question unambiguous without
   context. Cloze-style ("The kernel hard-imports exactly ___ plugin
   modules") works well for definitions.
3. 15-30 cards per deck; offer to split larger topics.

## Quizzing

1. Read the deck, select due cards (today >= last + box interval),
   shuffle, ask ONE at a time. Wait for the answer before showing yours.
2. Grade honestly: correct -> box+1, wrong -> box 1. Update the comment
   line via `edit_file` targeted replace after the session.
3. End with the session score and the next due date.

## Scheduling

Offer a recurring quiz via `schedule_subagent` — but note the subagent
runs unattended, so a scheduled session should DELIVER the due-card count
as a message ("5 cards due today — open the chat and say 'quiz me'"),
not try to hold an interactive quiz.
