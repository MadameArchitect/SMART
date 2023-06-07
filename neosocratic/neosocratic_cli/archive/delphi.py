"a cli for researchers to design dialogs, engineer prompts, science(responses)"
from typing import Optional
import itertools
import sqlite3

from pydantic.dataclasses import dataclass
from rich.markdown import Markdown
from rich.console import Console
from pydantic import BaseSettings
import pandas as pd
import optuna
import openai
import click

FACILITATOR_INSTRUCTIONS_FILENAME = "facilitator_instructions.md"
ACTOR_INSTRUCTIONS_FILENAME = "actor_instructions.md"
BELIEFS_FILENAME = "beliefs.csv"


def extract_instructions(filename: str) -> str:
    "read the instructions from the instructions.md file"
    with open(filename, "r") as file:
        instructions: str = file.read()
    return instructions


def extract_beliefs(filename: str = BELIEFS_FILENAME) -> dict[str, tuple[str, ...]]:
    "read the beliefs from the beliefs.csv file"
    return {
        row[0]: " ".join(row[1].values)
        for row in pd.read_csv("beliefs.csv", sep=";", index_col=0).iterrows()
    }


class Environment(BaseSettings):
    openai_api_key: str
    db_name: str = "roundtable.db"
    model: str = "gpt-3.5-turbo"
    actor_instructions: str = extract_instructions(ACTOR_INSTRUCTIONS_FILENAME)
    facilitator_instructions: str = extract_instructions(
        FACILITATOR_INSTRUCTIONS_FILENAME
    )
    topic: str = "How can we construct a genuine democratic platform?"
    beliefs: dict[str, str] = extract_beliefs()


env = Environment()


@dataclass(frozen=True)
class Actor:
    belief_names: tuple[str, str] = (
        "Summarize Actors' Responses & Reasoning",
        "Conceal Actors' Identities",
    )
    belief_descriptions: tuple[str, str] = (
        "Summarize the responses and reasoning of the actors in the dialog.",
        "Conceal the identities of the actors in the dialog.",
    )


@dataclass(frozen=True)
class Message:
    id: int = 0
    role: str = "user"
    content: str = env.topic
    parent_id: Optional[int] = None
    dialog_id: Optional[int] = None
    round_n: int = 0


default_console = Console()
facilitator = Actor()
default_message = Message()


class Dialog(pd.DataFrame):
    "a DAG of prompts and responses rooted on a topic"
    # retain the topic property
    _metadata = ["_topic", "_dialog"]

    @property
    def _constructor(self):
        return Dialog

    @property
    def topic(self):
        return self._topic

    @property
    def dialog_id(self):
        return self._dialog_id

    def __init__(self, data=None, index=None, columns=None, dtype=None, copy=False):
        if data is None:
            data = [
                {
                    "id": 0,
                    "role": "user",
                    "content": env.topic,
                    "parent_id": None,
                    "dialog_id": 0,
                }
            ]
        if columns is None:
            columns = ["id", "role", "content", "parent_id", "timestamp"]
        if dtype is None:
            dtype = {"id": int, "role": str, "content": str, "parent_id": Optional[int]}
        super().__init__(data, index, columns, dtype, copy)
        self._topic = self["content"].values[0] if not self.empty else None


default_dialog = Dialog()


def extract(
    trial: Optional[optuna.Trial] = None,
    include_descriptions: bool = True,
    instructions: str = env.actor_instructions,
    actor: Optional[Actor] = None,
    dialog: Dialog = default_dialog,
) -> Message:
    "construct messages to prompt an actor given their beliefs and the dialog state"
    assert actor is not None
    assert (
        instructions is not None and len(instructions) > 0
    ), "LLM requires instructions"
    assert (
        len(actor.belief_names) == len(actor.belief_descriptions) == 2
    ), "Actor needs primary & secondary beliefs"
    # use Optuna to A/B test belief description ablation
    # (as an example of optuna for prompt engineering)
    include_descriptions = (
        trial.suggest_categorical("include_descriptions", [True, False])
        if trial is not None
        else include_descriptions
    )
    primary_desc = " " + actor.belief_descriptions[0] if include_descriptions else ""
    secondary_desc = " " + actor.belief_descriptions[1] if include_descriptions else ""
    prompt = f"""
LLM Instructions: {instructions}
Your Primary Belief: {actor.belief_names[0]}.{primary_desc}
Your Secondary Belief: {actor.belief_names[1]}.{secondary_desc}
Topic: {dialog.topic}
"""
    message = Message(role="user", content=prompt)
    return message


def transform(
    openai_api_key: str = env.openai_api_key,
    model: str = env.model,
    message: Message = default_message,
) -> str:
    "invoke a Transformer to generate a response to a prompt"
    assert (
        openai_api_key is not None and len(openai_api_key) > 0
    ), "Transformer requires an OpenAI API key"
    assert model is not None and len(model) > 0, "Transformer requires a model"
    assert (
        isinstance(message, Message)
        and len(message.role) > 0
        and len(message.content) > 0
    ), "Transformer requires a message"
    openai.api_key = openai_api_key
    completion = openai.ChatCompletion.create(
        model=model, messages=[{"role": message.role, "content": message.content}]
    )
    content = completion["choices"][0]["message"].content
    response = Message(role="assistant", content=content)
    return response


def render_message(
    console: Console = default_console, message: Message = default_message
) -> bool:
    "display a Message with role & content in the console"
    assert isinstance(message, Message), "Message must have role & content"
    markdown_str = f"# {message['role']}:\n{message['content']}"
    markdown = Markdown(markdown_str)
    console.print(markdown)
    return True


def insert_dialog_step(
    db_name: str = env.db_name,
    prompt: Message = default_message,
    response: Message = default_message,
    parent_id: Optional[int] = None,
    dialog_id: Optional[int] = None,
) -> bool:
    if not prompt or not response:
        raise ValueError("Prompt and response must be non-empty strings")
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    c.execute(
        """
CREATE TABLE IF NOT EXISTS messages
    (id INTEGER PRIMARY KEY,
    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
    role TEXT,
    content TEXT,
    parent_id INTEGER,
    dialog_id INTEGER,
    round_n INTEGER,
    FOREIGN KEY(dialog_id) REFERENCES dialogs(dialog_id))
"""
    )
    INSERT_MESSAGE_QUERY = (
        """
    INSERT INTO messages
        (role, content, parent_id, dialog_id, round_n)
        VALUES (?, ?, ?, ?, ?)
    """,
    )
    c.execute(
        INSERT_MESSAGE_QUERY,
        (prompt.role, prompt.content, parent_id, dialog_id, prompt.round_n),
    )
    last_id = c.lastrowid
    c.execute(
        INSERT_MESSAGE_QUERY,
        (response.role, response.content, last_id, dialog_id, response.round_n),
    )
    conn.commit()
    conn.close()
    return True


def load(
    dialog: Dialog = default_dialog,
    prompt_message: Message = default_message,
    response_message: Message = default_message,
) -> Dialog:
    if prompt_message is None or response_message is None:
        raise ValueError("prompt_message and response_message must be Messages")
    try:
        prompt_rendered = render_message(prompt_message)
        response_rendered = render_message(response_message)
        step_inserted = insert_dialog_step(
            prompt=prompt_rendered, response=response_rendered
        )
    except Exception as e:
        print(e)
    success = prompt_rendered and response_rendered and step_inserted
    return success


def etl_step(
    trial: Optional[optuna.Trial] = None,
    include_descriptions: bool = env.include_descriptions,
    instructions: str = env.instructions,
    openai_api_key: str = env.openai_api_key,
    model: str = env.model,
    actor: Optional[Actor] = None,
    dialog: Dialog = default_dialog,
) -> Dialog:
    assert actor is not None
    prompt_message: Message = extract(
        trial=trial,
        include_descriptions=include_descriptions,
        instructions=instructions,
        actor=actor,
        dialog=dialog,
    )
    response_message: Message = transform(
        openai_api_key=openai_api_key,
        model=model,
        message=prompt_message,
    )
    new_dialog = load(prompt_message, response_message)
    return new_dialog


def make_actors(environ: Environment = env) -> tuple[Actor, ...]:
    actors = []
    beliefs = tuple(env.beliefs.items())
    for primary, secondary in itertools.combinations(beliefs, 2):
        belief_names = (primary[0], secondary[0])
        belief_descriptions = (primary[1], secondary[1])
        actor = Actor(belief_names, belief_descriptions)
        actors.append(actor)
    return tuple(actors)


def create_new_dialog():
    conn = sqlite3.connect(env.db_name)
    c = conn.cursor()
    c.execute(
        """
    CREATE TABLE IF NOT EXISTS dialogs
        (dialog_id INTEGER PRIMARY KEY,
        start_time TEXT DEFAULT CURRENT_TIMESTAMP,
        end_time TEXT,
        final_state TEXT)
    """
    )
    c.execute("INSERT INTO dialogs (start_time) VALUES (CURRENT_TIMESTAMP)")
    new_dialog_id = c.last_insert_rowid()
    conn.commit()
    conn.close()
    return new_dialog_id


@click.command()
@click.option(
    "--message",
    "-m",
    default=env.default_message,
    help="The message to which the AI LLM will respond.",
    show_default=True,
)
@click.option(
    "--n_rounds",
    "-n",
    default=env.n_rounds,
    help="The number of rounds of branching and merging dialog.",
    show_default=True,
)
def main(message: str, n_rounds: int) -> Dialog:
    "run the CLI (command line interface) and orchestrate a dialog tree"
    if message is None or len(message) == 0:
        message = env.default_message

    dialog = Dialog([{"id": 0, "role": "user", "content": message}])
    actors = make_actors()

    for round_n in range(n_rounds):
        for actor in actors:
            try:
                dialog = etl_step(
                    actor=actor,
                    dialog=dialog,
                )
            except Exception as e:
                print(f"Error occurred: {e}")


if __name__ == "__main__":
    main()
