from state_machine.forms import schema_to_form_steps


def test_schema_form_prompts_include_action_and_description():
    steps = schema_to_form_steps({
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Short title for the scheduled conversation."},
            "run_immediately": {"type": "boolean", "description": "If true, run one turn immediately.", "default": False},
        },
        "required": ["title"],
    }, prompt_optional=True)

    assert steps[0].prompt == "Enter a title.\nShort title for the scheduled conversation."
    assert steps[1].prompt == "Choose run immediately.\nIf true, run one turn immediately."
    assert steps[1].prompt_when_missing is True
