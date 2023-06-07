import pytest
import sqlite3
import pandas as pd
from delphi import (
    extract,
    transform,
    load,
    etl_step,
    Environment,
    Actor,
    Message,
    Dialog,
    create_new_dialog,
    insert_dialog_step,
)

# Set up environment
env = Environment()


def test_extract():
    actor = Actor(("Belief 1", "Belief 2"), ("Description 1", "Description 2"))
    dialog = Dialog([{"id": 0, "role": "user", "content": env.topic}])
    message = extract(actor=actor, dialog=dialog)
    assert message.role == "user"
    assert "LLM Instructions" in message.content
    assert "Your Primary Belief: Belief 1." in message.content
    assert "Your Secondary Belief: Belief 2." in message.content


def test_transform():
    message = Message(role="user", content="Hello, how are you?")
    response = transform(
        openai_api_key=env.openai_api_key, model=env.model, message=message
    )
    assert response.role == "assistant"
    assert isinstance(response.content, str)


def test_insert_dialog_step():
    dialog_id = create_new_dialog()
    prompt = Message(
        id=1, role="user", content="Prompt content", dialog_id=dialog_id, round_n=1
    )
    response = Message(
        id=2,
        role="assistant",
        content="Response content",
        parent_id=1,
        dialog_id=dialog_id,
        round_n=1,
    )
    assert insert_dialog_step(
        prompt=prompt, response=response, parent_id=prompt.id, dialog_id=dialog_id
    )


def test_load():
    dialog = Dialog([{"id": 0, "role": "user", "content": env.topic}])
    prompt_message = Message(role="user", content="This is a test prompt")
    response_message = Message(role="assistant", content="This is a test response")
    new_dialog = load(
        dialog=dialog, prompt_message=prompt_message, response_message=response_message
    )
    assert isinstance(new_dialog, Dialog)


def test_etl_step():
    actor = Actor(("Belief 1", "Belief 2"), ("Description 1", "Description 2"))
    dialog = Dialog([{"id": 0, "role": "user", "content": env.topic}])
    new_dialog = etl_step(actor=actor, dialog=dialog)
    assert isinstance(new_dialog, Dialog)
