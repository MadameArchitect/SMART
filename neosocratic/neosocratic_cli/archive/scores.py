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


def request_scores(prompt: Message, response: Message):
    render_message(prompt)
    render_message(response)
    coherence = int(input("How relevant is the response to the prompt? int(1-99): "))
    comprehensiveness = int(input("How comprehensive is the response? int(1-99): "))
    process_correctness = int(
        input("How correct is the process followed in the response? int(1-99): ")
    )
    outcome_correctness = int(
        input("How correct is the outcome of the response? int(1-99): ")
    )
    flexibility = int(input("How flexible is the response? int(1-99):"))
    concision = int(input("How concise is the response? (1-99): "))
    scores = tuple(
        [
            min(99, max(1, score))
            for score in (
                coherence,
                comprehensiveness,
                process_correctness,
                outcome_correctness,
                flexibility,
                concision,
            )
        ]
    )
    return scores


# def insert_dialog_scores(db_name: str = env.db_name, response_id: int, scores: tuple) -> bool:
#     "load scores for a response into the sqlite scores table"
#     conn = sqlite3.connect(db_name)
#     c = conn.cursor()
#     c.execute("""
# CREATE TABLE IF NOT EXISTS scores
#     (id INTEGER PRIMARY KEY,
#     message_id INTEGER,
#     coherence_score INTEGER,
#     comprehensiveness_score INTEGER,
#     process_correctness_score INTEGER,
#     outcome_correctness_score INTEGER,
#     flexibility_score INTEGER,
#     concision_score INTEGER,
#     FOREIGN KEY (message_id) REFERENCES messages(id))
# """)
#     c.execute("INSERT INTO scores (message_id, coherence_score, comprehensiveness_score, process_correctness_score, outcome_correctness_score, flexibility_score, concision_score) VALUES (?, ?, ?, ?, ?, ?, ?)",
#               (response_id, *scores))
#     conn.commit()
#     conn.close()
#     return True
