"a cli for researchers to design and run dialog[ue]s and estimate treatment effects"
# TODO: integrate optuna to optimize dialogs, prompts, agents, beliefs, etc.
# TODO: evaluate conditional/heterogeneous average treatment effects
# TODO: add agents, beliefs, templates, instructions to SQLite
# TODO: add `pipe` for sequential dialog designs
# TODO: add `product` fully connected dialogs
from sqlite3 import Connection, Cursor
from typing import Optional
import requests
import sqlite3
import json

from pydantic import BaseSettings
from rich.markdown import Markdown
from rich.console import Console
from rich import print
import pandas as pd
import click


class Environment(BaseSettings):
    """Environment variables."""

    # general use
    openai_api_key: str
    calling_openai: bool = False  # Switch OpenAI on and off here
    temperature: float = 0.0
    max_tokens: int = 420
    model: str = "gpt-3.5-turbo"

    # task specific use
    sqlite_db_path: str = "delphi.db"
    beliefs_path: str = "data_framework_beliefs.csv"
    topic: str = "What is the meaning of life?"
    rounds: int = 3
    dialog_design: str = "delphi"

    class Config:
        """Load environment variables from .env file."""

        env_file = ".env"


env = Environment()


def connect_sqlite(
    database_path: str = env.sqlite_db_path,
) -> tuple[Connection, Cursor]:
    "Connect to the SQLite database."
    db_connection: Connection = sqlite3.connect(database_path)
    db_cursor: Cursor = db_connection.cursor()
    # Create SQL tables if they don't exist
    db_cursor.execute(
        """
CREATE TABLE IF NOT EXISTS dialogs (
    id integer PRIMARY KEY,
    created_at text DEFAULT CURRENT_TIMESTAMP,
    topic text,
    design text
)"""
    )
    db_cursor.execute(
        """
CREATE TABLE IF NOT EXISTS messages (
    id integer PRIMARY KEY,
    dialog_id integer,
    prompt text,
    response text,
    FOREIGN KEY(dialog_id) REFERENCES dialogs(dialog_id)
)"""
    )
    return db_connection, db_cursor


@click.command()
@click.option(
    "--sqlite-db-path",
    "-s",
    default=env.sqlite_db_path,
    help="Path to the SQLite database.",
)
@click.option(
    "--topic",
    "-t",
    default=env.topic,
    help="Topic to discuss.",
)
@click.option("--rounds", "-r", default=env.rounds, help="Number of rounds.")
@click.option("--beliefs", "-b", default=env.beliefs_path, help="Name of beliefs.")
@click.option(
    "--design", "-d", default=env.dialog_design, help="Name of dialog design."
)
def main(sqlite_db_path: str, topic: str, rounds: int, beliefs: str, design: str):
    "The entry point for the script."
    # Connect to the SQLite database
    db_connection, db_cursor = connect_sqlite(sqlite_db_path)
    # Load beliefs from CSV
    names_weights_beliefs: pd.DataFrame = read_names_weights_beliefs(env.beliefs_path)
    # Decide which dialog design to run
    if design == "delphi":
        # Run Delphi method
        run_delphi_method(db_cursor, names_weights_beliefs, topic, rounds)
    else:
        raise NotImplementedError(f"design {design} not implemented")


def read_names_weights_beliefs(file_path: str) -> pd.DataFrame:
    "read agent names, belief weights, and beliefs from a CSV file into a pd.DataFrame."
    with open(file_path, newline="") as f:
        names_weights_beliefs = pd.read_csv(f, sep=";", index_col="Name")
    return names_weights_beliefs


def run_delphi_method(
    db_cursor: Cursor,
    actor_instructions: str,
    facilitator_instructions: str,
    names_weights_beliefs: pd.DataFrame,
    topic: str,
    rounds: int,
    calling_openai: bool = False,
) -> tuple[tuple[frozenset, ...], tuple[str]]:
    "Orchestrate the Delphi Method about a topic for a given number of rounds."
    response_frozensets: list[frozenset] = []
    topics: list = [topic]
    for round_number in range(rounds):
        # Branch: each expert responds to the topic
        responses: frozenset[str, ...] = branch(
            db_cursor,
            actor_instructions,
            names_weights_beliefs,
            round_number,
            topic,
            calling_openai,
        )
        response_frozensets.append(responses)
        # Merge: the facilitator summarizes responses to make the next topic
        topic: str = merge(
            db_cursor,
            facilitator_instructions,
            round_number,
            responses,
            calling_openai,
        )
        if not calling_openai:
            # hash the topic to shorten it so we don't blow up the stdout
            topic = str(hash(topic))
        topics.append(topic)
    return tuple(response_frozensets), tuple(topics)


default_console = Console()


def render_markdown(content: str, console: Console = default_console) -> None:
    "display a Message with role & content in the console"
    markdown = Markdown(content)
    console.print(markdown)


def call_openai_api(
    prompt: str,
    model: str = env.model,
    max_tokens: int = env.max_tokens,
    temperature: float = env.temperature,
    openai_api_key: str = env.openai_api_key,
) -> Optional[str]:
    "Request an OpenAI Chat Completion for a prompt"
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {openai_api_key}",
            },
            data=json.dumps(
                {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }
            ),
        )
        response_json = response.json()
        response_message_content = response_json["choices"][0]["message"]["content"]
        return response_message_content
    except Exception as e:
        print("Exception calling OpenAI:", e)
        return None


def branch(
    db_cursor: Cursor,
    actor_instructions: str,
    names_weights_beliefs: pd.DataFrame,
    round_number: int,
    topic: str,
    calling_openai: bool = False,
) -> frozenset[str, ...]:
    "Generate N responses to the topic, one per pair of beliefs."
    prompts = []
    responses = []
    for name, weights_beliefs in names_weights_beliefs.iterrows():
        # handle variadic weighted beliefs (w1, b1, w2, b2, ...)
        # weights_beliefs is a dataframe with columns (w1, b1, w2, b2, ...)
        weights = weights_beliefs[::2]
        beliefs = weights_beliefs[1::2]
        # handle edge cases where weights sum to 0, are nan, or are negative
        if (
            sum(weights) <= 0
            or any(pd.isna(weights))
            or any(weight < 0 for weight in weights)
        ):
            weight_belief_bullets = "- N/A (Think for yourself!)"
        # handle edge case where the number of weights and beliefs don't match
        elif len(weights) != len(beliefs):
            print(f"{weights=}")
            print(f"{beliefs=}")
            raise ValueError("Number of weights and beliefs did not match.")
        # handle edge cases where weights and beliefs are not the right types
        elif not all(isinstance(belief, str) for belief in beliefs):
            print(f"{beliefs=}")
            raise ValueError("Beliefs must be strings.")
        elif not all(isinstance(weight, float) for weight in weights):
            print(f"{weights=}")
            raise ValueError("Weights must be floats.")
        # otherwise, we have a valid set of weighted beliefs
        else:
            # normalize weights to sum to 1
            normalized_weights = weights / sum(weights)
            # group them in pairs (w1, b1), (w2, b2), ...
            weight_belief_pairs = tuple(zip(normalized_weights, beliefs))
            weight_belief_bullets = "\n".join(
                f"""
- Belief ***{i}*** - Weight = ***{weight}***:
    - ***{belief}***
"""
                for i, (weight, belief) in enumerate(weight_belief_pairs)
            )
        if name is None or name == "":
            name = "OpenAI ChatGPT"
        prompt = f"""
# Dialog Round {round_number}: {name}
## LLM Instructions:
- ***{actor_instructions}***
## Your Weighted Beliefs:
{weight_belief_bullets}
## Respond to this Topic:
- ***{topic}***
## Begin Response:
"""
        prompts.append(prompt)
        render_markdown(prompt)
        if calling_openai:
            # Call OpenAI API
            response = call_openai_api(prompt)
            responses.append(response)
            # Store prompt and response in the database
            db_cursor.execute(
                "INSERT INTO messages (prompt, response) VALUES (?, ?)",
                (prompt, response.choices[0].text.strip()),
            )
            db_cursor.commit()
            render_markdown(response)
    return frozenset(responses) if calling_openai else frozenset(prompts)


def merge(
    db_cursor: Cursor,
    facilitator_instructions: str,
    round_number: int,
    responses: frozenset[str],
    calling_openai: bool = False,
) -> str:
    "Generate a summary prompt based on a set of responses."
    if len(responses) == 0:
        raise ValueError("No responses to summarize.")
    response_bullets = "\n".join(
        f"""
### Response ***{i}***:
```
{response}
```
"""
        for i, response in enumerate(responses)
    )
    summary_prompt = f"""
# Dialog Round {round_number} - Facilitator
## LLM Instructions:
- ***{facilitator_instructions}***
## Responses:
{response_bullets}
## Summarize Responses without revealing who said what:
"""
    render_markdown(summary_prompt)
    if calling_openai:
        summary_response = call_openai_api(summary_prompt)
        db_cursor.execute(
            "INSERT INTO discussions (prompt, response) VALUES (?, ?)",
            (summary_prompt, summary_response),
        )
        db_cursor.commit()
        render_markdown(summary_response)
        return summary_response
    return summary_prompt


if __name__ == "__main__":
    main()
