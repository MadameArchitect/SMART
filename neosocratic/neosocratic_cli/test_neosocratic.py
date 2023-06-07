from sqlite3 import Connection, Cursor
import random  # noqa F401

import pandas as pd
import pytest

from neosocratic import (  # noqa F401
    Environment,
    connect_sqlite,
    read_names_weights_beliefs,
    call_openai_api,
    branch,
    merge,
    run_delphi_method,
)


@pytest.fixture
def env():
    environ = Environment(
        sqlite_db_path="test_neosocratic.db",
        beliefs_path="data_framework_beliefs.csv",
    )
    print(environ)
    return environ


@pytest.fixture
def names_weights_beliefs(env):
    return read_names_weights_beliefs(env.beliefs_path)


@pytest.fixture
def topic():
    return "How do you design a democratic software platform?"


@pytest.fixture
def rounds():
    return 3


@pytest.fixture
def actor_instructions():
    return "Respond to the topic according to your weighted beliefs."


@pytest.fixture
def facilitator_instructions():
    return "Summarize the responses without revealing who said what."


def test_environment(env):
    assert isinstance(env, Environment)
    assert env.openai_api_key is not None
    assert isinstance(env.sqlite_db_path, str)
    assert isinstance(env.beliefs_path, str)


def test_connect_sqlite(env):
    db_connection, db_cursor = connect_sqlite(env.sqlite_db_path)
    assert isinstance(db_connection, Connection)
    assert isinstance(db_cursor, Cursor)


def test_read_names_weights_beliefs(env):
    names_weights_beliefs = read_names_weights_beliefs(env.beliefs_path)
    assert isinstance(names_weights_beliefs, pd.DataFrame)
    assert names_weights_beliefs.index.name == "Name"


# def test_call_openai_api():
#     query = random.choice(
#         [
#             "What do you think about thinking?",
#             "What's the laplace transform of E=mc^2?",
#             "What's the meaning of life?",
#             "What's the best way to learn?",
#         ]
#     )
#     openai_response = call_openai_api(query)
#     assert openai_response is not None


def test_branch_and_merge(actor_instructions, names_weights_beliefs, topic):
    # test branch
    print("Begin Branch")
    agent_prompts = branch(
        db_cursor=None,
        actor_instructions=actor_instructions,
        names_weights_beliefs=names_weights_beliefs,
        round_number=1,
        topic=topic,
        calling_openai=False,
    )
    assert isinstance(agent_prompts, frozenset)
    assert len(agent_prompts) == len(names_weights_beliefs)

    # test merge
    print("Begin Merge")
    facilitator_prompt = merge(
        db_cursor=None,
        facilitator_instructions=facilitator_instructions,
        round_number=1,
        responses=agent_prompts,
        calling_openai=False,
    )
    assert isinstance(facilitator_prompt, str)
    assert len(facilitator_prompt) > 0


def test_run_delphi_method(
    actor_instructions, facilitator_instructions, names_weights_beliefs, rounds, topic
):
    response_frozensets, topics = run_delphi_method(
        db_cursor=None,
        actor_instructions=actor_instructions,
        facilitator_instructions=facilitator_instructions,
        names_weights_beliefs=names_weights_beliefs,
        topic=topic,
        rounds=rounds,
        calling_openai=False,
    )
    assert isinstance(response_frozensets, tuple)
    assert len(response_frozensets) == rounds
    assert all(isinstance(x, frozenset) for x in response_frozensets)
    assert len(response_frozensets[0]) == len(names_weights_beliefs)
    print(f"{response_frozensets=}")
    assert isinstance(topics, tuple)
    print(f"{topics=}")
