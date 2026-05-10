from state_machine.serialization import latest_state, messages_to_history, save_compaction_marker, save_state_marker


class FakeDb:
    def __init__(self):
        self.rows = []

    def save_message(self, conversation_id, role, content, tool_call_id=None, tool_name=None):
        self.rows.append({"role": role, "content": content, "tool_call_id": tool_call_id, "tool_name": tool_name})


def test_compaction_marker_preserves_db_rows_but_hides_pre_checkpoint_replay():
    db = FakeDb()
    db.save_message(1, "user", "old user")
    db.save_message(1, "assistant", "old assistant")
    save_compaction_marker(db, 1, "Earlier summary.")
    db.save_message(1, "user", "after")
    save_state_marker(db, 1, {"active_agent_profile": "builder"})

    history = messages_to_history(db.rows)

    assert [r["content"] for r in db.rows if r["role"] != "system"] == ["old user", "old assistant", "after"]
    assert history == [
        {"role": "user", "content": "[Conversation summary from earlier]\nEarlier summary."},
        {"role": "assistant", "content": "Understood - I have the earlier context."},
        {"role": "user", "content": "after"},
    ]
    assert latest_state(db.rows)["active_agent_profile"] == "builder"
