"""
FastAPI backend for the Adaptive Tutor demo.

Single endpoint: given a policy name and a seed, run one simulated
30-question tutoring session and return the full step-by-step trace.

Each step reports TWO mastery views:
  - "true_ability": ground-truth learning (only knowable in simulation).
    This is the metric used for "final_avg_mastery" and all comparisons,
    since it cannot be gamed.
  - "est_mastery": the noisy observable estimate the agent/heuristic
    actually see and act on -- shown for the per-topic bars/table so
    the viewer can see what information the policy was working with.
"""
import datetime
import json
import numpy as np
import os
import re
import uuid
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from stable_baselines3 import PPO, DQN, A2C
from openai import OpenAI, APIError

from env.tutor_env import AdaptiveTutorEnv, N_DIFFICULTY_LEVELS, MAX_TOPICS, decode_action
from env.student import SyntheticStudent, estimate_update

app = FastAPI(title="Adaptive Tutor RL API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the frontend from this same app/origin -- avoids CORS and
# "wrong API_BASE" issues entirely once deployed, since the page and
# the API it calls are then always the same host. Local dev can still
# open frontend/index.html directly if preferred (see API_BASE
# auto-detection in that file).
_FRONTEND_PATH = os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html")


@app.get("/", include_in_schema=False)
def serve_frontend():
    return FileResponse(_FRONTEND_PATH)


N_DEMO_TOPICS = 5  # the curated "Compare All 6" / "Single Strategy" scenario
SESSION_LENGTH = 30
TOPIC_NAMES = ["Algebra", "Geometry", "Probability", "Calculus Basics", "Word Problems"]

# ---------------------------------------------------------------------
# Student roster (persistence)
#
# Everything above this used to live only in LIVE_SESSIONS (an
# in-memory dict) -- gone the moment the server restarts. That's fine
# for a quick demo but not for a teacher actually using this week to
# week. SQLite is deliberately the simplest thing that persists: a
# single local file, no server process to run, no extra service to
# deploy. Swap it for Postgres later if this ever needs multiple
# teachers hitting the same backend concurrently.
#
# NOTE: there's no login/auth here yet -- every student created is
# visible to whoever can reach this API. Fine for one teacher running
# this locally; add auth before exposing it beyond that.
# ---------------------------------------------------------------------
import sqlite3

DB_PATH = os.environ.get("TUTOR_DB_PATH", os.path.join(os.path.dirname(__file__), "..", "tutor.db"))


def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db():
    conn = _get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            topic_names TEXT NOT NULL,   -- JSON array, real topics only (not padded)
            est_mastery TEXT NOT NULL,   -- JSON array, same length as topic_names
            history TEXT NOT NULL,       -- JSON array of {topic, difficulty, correct}
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


_init_db()


def _student_row_to_dict(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "name": row["name"],
        "topic_names": json.loads(row["topic_names"]),
        "est_mastery": json.loads(row["est_mastery"]),
        "history": json.loads(row["history"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _now_iso() -> str:
    return datetime.datetime.utcnow().isoformat()


class CreateStudentRequest(BaseModel):
    name: str
    topic_names: list[str] | None = None  # optional at creation -- set later via POST /students/{id}/topics if omitted


class SetStudentTopicsRequest(BaseModel):
    topic_names: list[str]


class StudentSummary(BaseModel):
    id: str
    name: str
    n_topics: int
    avg_mastery: float
    questions_answered: int
    updated_at: str


class StudentDetail(BaseModel):
    id: str
    name: str
    topic_names: list[str]
    est_mastery: list[float]
    history: list[dict]
    created_at: str
    updated_at: str


def _validate_topic_names(topic_names: list[str]) -> list[str]:
    """No auto-generated names anywhere -- every entry must be typed by
    the teacher. Rejects blanks and duplicates instead of silently
    filling them in with placeholders like "Topic 3" or a default
    curriculum."""
    if not (1 <= len(topic_names) <= MAX_TOPICS):
        raise HTTPException(
            status_code=400,
            detail=f"topic_names must have between 1 and {MAX_TOPICS} entries -- got {len(topic_names)}.",
        )
    cleaned = [t.strip() for t in topic_names]
    if any(not t for t in cleaned):
        raise HTTPException(status_code=400, detail="Every topic name must be filled in -- no blanks allowed.")
    if len(set(t.lower() for t in cleaned)) != len(cleaned):
        raise HTTPException(status_code=400, detail="Topic names must be unique.")
    return cleaned


@app.post("/students", response_model=StudentDetail)
def create_student(req: CreateStudentRequest):
    if not req.name.strip():
        raise HTTPException(status_code=400, detail="name cannot be empty.")
    # Topics are optional here -- a student can be added by name alone and
    # have their subjects set later (first time they're "used"), via
    # POST /students/{id}/topics. There is no default topic list.
    topic_names = _validate_topic_names(req.topic_names) if req.topic_names is not None else []

    student_id = str(uuid.uuid4())
    est_mastery = [0.3] * len(topic_names)  # real topics only -- padded to MAX_TOPICS only inside a live session
    now = _now_iso()

    conn = _get_db()
    conn.execute(
        "INSERT INTO students (id, name, topic_names, est_mastery, history, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (student_id, req.name.strip(), json.dumps(topic_names), json.dumps(est_mastery), json.dumps([]), now, now),
    )
    conn.commit()
    conn.close()

    return StudentDetail(
        id=student_id, name=req.name.strip(), topic_names=topic_names,
        est_mastery=est_mastery, history=[], created_at=now, updated_at=now,
    )


@app.get("/students", response_model=list[StudentSummary])
def list_students():
    conn = _get_db()
    rows = conn.execute("SELECT * FROM students ORDER BY updated_at DESC").fetchall()
    conn.close()
    summaries = []
    for row in rows:
        s = _student_row_to_dict(row)
        avg_mastery = round(sum(s["est_mastery"]) / len(s["est_mastery"]), 4) if s["est_mastery"] else 0.0
        summaries.append(StudentSummary(
            id=s["id"], name=s["name"], n_topics=len(s["topic_names"]),
            avg_mastery=avg_mastery, questions_answered=len(s["history"]), updated_at=s["updated_at"],
        ))
    return summaries


@app.get("/students/{student_id}", response_model=StudentDetail)
def get_student(student_id: str):
    conn = _get_db()
    row = conn.execute("SELECT * FROM students WHERE id = ?", (student_id,)).fetchone()
    conn.close()
    if row is None:
        raise HTTPException(status_code=404, detail="Student not found.")
    return StudentDetail(**_student_row_to_dict(row))


@app.post("/students/{student_id}/topics", response_model=StudentDetail)
def set_student_topics(student_id: str, req: SetStudentTopicsRequest):
    """Set a student's subject list -- allowed exactly once, the first
    time they're set up (their saved mastery/history is keyed to this
    list, so it can't be silently changed out from under them later)."""
    conn = _get_db()
    row = conn.execute("SELECT * FROM students WHERE id = ?", (student_id,)).fetchone()
    if row is None:
        conn.close()
        raise HTTPException(status_code=404, detail="Student not found.")
    student = _student_row_to_dict(row)
    if student["topic_names"]:
        conn.close()
        raise HTTPException(
            status_code=400,
            detail="This student's topics are already set and can't be changed here -- their saved progress is tied to them.",
        )

    cleaned = _validate_topic_names(req.topic_names)
    est_mastery = [0.3] * len(cleaned)
    now = _now_iso()
    conn.execute(
        "UPDATE students SET topic_names = ?, est_mastery = ?, updated_at = ? WHERE id = ?",
        (json.dumps(cleaned), json.dumps(est_mastery), now, student_id),
    )
    conn.commit()
    updated_row = conn.execute("SELECT * FROM students WHERE id = ?", (student_id,)).fetchone()
    conn.close()
    return StudentDetail(**_student_row_to_dict(updated_row))


@app.delete("/students/{student_id}")
def delete_student(student_id: str):
    conn = _get_db()
    conn.execute("DELETE FROM students WHERE id = ?", (student_id,))
    conn.commit()
    conn.close()
    return {"status": "deleted"}

# ---------------------------------------------------------------------
# AI teacher summary -- turns a live session's raw numbers into a short
# natural-language note for the teacher, via Gemma 4 31B on Cerebras.
# This is plain prompting over data already in this app, NOT retrieval
# (RAG) -- there's no external document corpus involved.
# ---------------------------------------------------------------------
CEREBRAS_MODEL = "gemma-4-31b"
_cerebras_client = None


def _get_cerebras_client() -> OpenAI:
    global _cerebras_client
    if _cerebras_client is None:
        api_key = os.environ.get("CEREBRAS_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=500,
                detail="CEREBRAS_API_KEY environment variable is not set on the server.",
            )
        _cerebras_client = OpenAI(base_url="https://api.cerebras.ai/v1", api_key=api_key)
    return _cerebras_client

# Load all trained models once at startup, not per-request.
MODELS = {
    "ppo": PPO.load("training/ppo_tutor_agent"),
    "dqn": DQN.load("training/dqn_tutor_agent"),
    "a2c": A2C.load("training/a2c_tutor_agent"),
}


def _step_record(env, topic=None, difficulty=None, correct=None):
    # The environment always has MAX_TOPICS=15 slots internally (10 of
    # them pinned to 1.0 "fully known" for this 5-topic demo) -- slice
    # down to just the real N_DEMO_TOPICS, or the padded 1.0s would
    # wildly distort the numbers shown here.
    return {
        "true_ability": env.student.true_ability[:N_DEMO_TOPICS].tolist(),
        "est_mastery": env.student.est_mastery[:N_DEMO_TOPICS].tolist(),
        "topic": TOPIC_NAMES[topic] if topic is not None else None,
        "difficulty": round(float(difficulty), 2) if difficulty is not None else None,
        "correct": correct,
    }


def run_random(seed: int):
    env = AdaptiveTutorEnv(session_length=SESSION_LENGTH, n_active_topics=N_DEMO_TOPICS)
    rng = np.random.default_rng(seed)
    valid_actions = [t * N_DIFFICULTY_LEVELS + d for t in range(N_DEMO_TOPICS) for d in range(N_DIFFICULTY_LEVELS)]
    obs, info = env.reset(seed=seed)
    steps = [_step_record(env)]
    for _ in range(SESSION_LENGTH):
        action = int(rng.choice(valid_actions))
        obs, reward, terminated, truncated, info = env.step(action)
        steps.append(_step_record(env, info["topic"], info["difficulty"], bool(info["correct"])))
    return steps


def run_heuristic(seed: int):
    env = AdaptiveTutorEnv(session_length=SESSION_LENGTH, n_active_topics=N_DEMO_TOPICS)
    obs, info = env.reset(seed=seed)
    steps = [_step_record(env)]
    for _ in range(SESSION_LENGTH):
        topic = int(np.argmin(obs[:N_DEMO_TOPICS]))
        difficulty_idx = N_DIFFICULTY_LEVELS // 2
        action = topic * N_DIFFICULTY_LEVELS + difficulty_idx
        obs, reward, terminated, truncated, info = env.step(action)
        steps.append(_step_record(env, info["topic"], info["difficulty"], bool(info["correct"])))
    return steps


def make_model_runner(model):
    def _run(seed: int):
        env = AdaptiveTutorEnv(session_length=SESSION_LENGTH, n_active_topics=N_DEMO_TOPICS)
        obs, info = env.reset(seed=seed)
        steps = [_step_record(env)]
        for _ in range(SESSION_LENGTH):
            action, _ = model.predict(obs, deterministic=True)  # full 15-dim obs, as trained
            obs, reward, terminated, truncated, info = env.step(int(action))
            steps.append(_step_record(env, info["topic"], info["difficulty"], bool(info["correct"])))
        return steps
    return _run


def run_oracle(seed: int):
    student = SyntheticStudent(N_DEMO_TOPICS, seed=seed)  # no padding needed, oracle bypasses the trained model

    class _Wrap:
        pass
    env = _Wrap()
    env.student = student

    steps = [_step_record(env)]
    for _ in range(SESSION_LENGTH):
        topic = int(np.argmin(student.true_ability))
        difficulty = float(student.true_ability[topic])
        correct, true_gain, est_gain = student.attempt(topic, difficulty)
        steps.append(_step_record(env, topic, difficulty, bool(correct)))
    return steps


POLICY_RUNNERS = {
    "random": run_random,
    "heuristic": run_heuristic,
    "ppo": make_model_runner(MODELS["ppo"]),
    "dqn": make_model_runner(MODELS["dqn"]),
    "a2c": make_model_runner(MODELS["a2c"]),
    "oracle": run_oracle,
}


@app.get("/simulate")
def simulate(policy: str, seed: int = 0):
    if policy not in POLICY_RUNNERS:
        raise HTTPException(status_code=400, detail=f"Unknown policy '{policy}'. Choose from {list(POLICY_RUNNERS)}")
    steps = POLICY_RUNNERS[policy](seed)
    return {
        "policy": policy,
        "seed": seed,
        "topic_names": TOPIC_NAMES,
        "steps": steps,
        "final_avg_mastery": round(float(np.mean(steps[-1]["true_ability"])), 4),
    }


@app.get("/health")
def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------
# LIVE SESSION MODE -- for use with a REAL student, not a simulated one.
#
# Unlike /simulate (which replays a full pre-scripted 30-question
# session against a fake student), this drives the trained agent one
# question at a time: it recommends a topic+difficulty, you go ask that
# question to an actual person, tell the API whether they got it right,
# and it recommends the next one -- open-ended, not a fixed length.
#
# There's no "true_ability" here (no simulator, no hidden ground truth
# to peek at) -- only the observable estimate, updated from real
# answers, which is exactly what the agent was trained to work with
# anyway (it never saw true_ability during training either).
# ---------------------------------------------------------------------

LIVE_SESSIONS = {}  # session_id -> {policy, topic_names (real, len 1-MAX_TOPICS), est_mastery (padded len MAX_TOPICS), history, pending_topic, pending_difficulty}

DIFFICULTY_VALUES_LIST = np.linspace(0.1, 0.9, N_DIFFICULTY_LEVELS).tolist()

# Friendly 3-level labels for manual difficulty choice, mapped to 3 of
# the 5 internal levels the agent was actually trained on. We don't
# expose all 5 (0.1/0.3/0.5/0.7/0.9) to keep the UI simple -- the two
# in-between levels remain available to the agent's own auto-choice,
# just not to manual override.
DIFFICULTY_LABELS = {"easy": 0.1, "medium": 0.5, "hard": 0.9}


def _nearest_difficulty(value: float) -> float:
    """Snap an arbitrary difficulty value to the nearest of the 5 discrete
    levels the agent was trained on (0.1, 0.3, 0.5, 0.7, 0.9)."""
    return float(min(DIFFICULTY_VALUES_LIST, key=lambda d: abs(d - value)))


class StartSessionRequest(BaseModel):
    policy: str = "dqn"
    topic_names: list[str] | None = None  # 1 to MAX_TOPICS entries, all required (no default) -- ignored if student_id is set
    student_id: str | None = None  # attach this session to a saved student -- loads their prior mastery/history and writes answers back as they happen. Omit for a throwaway/anonymous session.


class AnswerRequest(BaseModel):
    correct: bool


class ChooseTopicRequest(BaseModel):
    topic_index: int | None = None  # index into the REAL topics, or null/omitted to let the agent choose
    difficulty_label: str | None = None  # "easy" | "medium" | "hard" | null. Only meaningful when topic_index is set.


def _recommend(session, forced_topic: int | None = None, forced_difficulty_label: str | None = None):
    """
    Returns (topic, difficulty) where topic is an index into the REAL
    topics (0..n_real-1), never into the padded/phantom slots.

    If forced_topic is given but no difficulty label, the difficulty is
    chosen by matching it to the current estimate for that topic (the
    theoretically ideal difficulty, per our own IRT-style logic) rather
    than consulting the trained agent -- asking the agent to evaluate one
    specific topic in isolation would require reaching into
    algorithm-specific internals (DQN's Q-network vs PPO/A2C's policy
    logits), which adds real complexity for a case where the
    "match difficulty to current estimate" heuristic is already close to
    optimal (see our oracle baseline, which uses exactly this rule and
    sits near the top of every comparison in this project).

    If forced_difficulty_label is also given, it overrides that
    heuristic entirely with the user's explicit Easy/Medium/Hard choice.
    """
    n_real = len(session["topic_names"])

    if forced_topic is not None:
        if not (0 <= forced_topic < n_real):
            raise HTTPException(status_code=400, detail=f"topic_index must be between 0 and {n_real - 1}.")
        if forced_difficulty_label is not None:
            if forced_difficulty_label not in DIFFICULTY_LABELS:
                raise HTTPException(
                    status_code=400,
                    detail=f"difficulty_label must be one of {list(DIFFICULTY_LABELS)} or null -- "
                           f"got '{forced_difficulty_label}'.",
                )
            difficulty = DIFFICULTY_LABELS[forced_difficulty_label]
        else:
            difficulty = _nearest_difficulty(session["est_mastery"][forced_topic])
        return forced_topic, difficulty

    model = MODELS[session["policy"]]
    obs = np.array(session["est_mastery"], dtype=np.float32)  # full padded length-MAX_TOPICS vector
    action, _ = model.predict(obs, deterministic=True)
    topic, difficulty = decode_action(int(action))

    if topic >= n_real:
        # Agent picked one of the padded "phantom" slots (padded to 1.0
        # mastery specifically so a trained agent should almost never
        # want to, since it learned to avoid already-mastered topics --
        # this is just a safety net for the rare edge case). Fall back
        # to the weakest real topic, using the same difficulty-matching
        # rule as the manual-topic-choice path above.
        topic = int(np.argmin(session["est_mastery"][:n_real]))
        difficulty = _nearest_difficulty(session["est_mastery"][topic])

    return topic, difficulty


def _difficulty_description(value: float) -> str:
    """Turn a raw 0-1 difficulty value into a short phrase for prompts."""
    if value <= 0.2:
        return "easy, foundational"
    elif value <= 0.4:
        return "easy-to-medium"
    elif value <= 0.6:
        return "medium"
    elif value <= 0.8:
        return "medium-to-hard"
    else:
        return "hard, challenging"


def _session_summary(session, session_id):
    return {
        "session_id": session_id,
        "policy": session["policy"],
        "student_id": session.get("student_id"),
        "student_name": session.get("student_name"),
        "topic_names": session["topic_names"],
        "est_mastery": session["est_mastery"][: len(session["topic_names"])],  # hide padded phantom slots
        "avg_mastery": round(float(np.mean(session["est_mastery"][: len(session["topic_names"])])), 4),
        "history": session["history"],
        "recommended_topic": session["topic_names"][session["pending_topic"]],
        "recommended_topic_index": session["pending_topic"],
        "recommended_difficulty": round(session["pending_difficulty"], 2),
        "question_number": len(session["history"]) + 1,
    }


@app.post("/session/start")
def start_session(req: StartSessionRequest):
    if req.policy not in MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown policy '{req.policy}'. Choose from {list(MODELS)}")

    if req.student_id is not None:
        conn = _get_db()
        row = conn.execute("SELECT * FROM students WHERE id = ?", (req.student_id,)).fetchone()
        conn.close()
        if row is None:
            raise HTTPException(status_code=404, detail="Student not found.")
        student = _student_row_to_dict(row)
        if not student["topic_names"]:
            raise HTTPException(
                status_code=400,
                detail="This student doesn't have topics set yet -- set them first with POST /students/{id}/topics.",
            )
        topic_names = student["topic_names"]
        real_mastery = student["est_mastery"]
        history = student["history"]
    else:
        if not req.topic_names:
            raise HTTPException(
                status_code=400,
                detail="topic_names is required for a session with no student_id -- there is no default topic list.",
            )
        topic_names = _validate_topic_names(req.topic_names)
        real_mastery = [0.3] * len(topic_names)
        history = []

    n_real = len(topic_names)
    # Pad unused slots to 1.0 ("fully known") so the trained agent, which
    # learned to avoid already-mastered topics, naturally steers clear of
    # them -- see _recommend()'s fallback for the rare case it doesn't.
    est_mastery = real_mastery + [1.0] * (MAX_TOPICS - n_real)

    session_id = str(uuid.uuid4())
    session = {
        "policy": req.policy,
        "student_id": req.student_id,
        "student_name": student["name"] if req.student_id is not None else None,
        "topic_names": topic_names,
        "est_mastery": est_mastery,
        "history": history,
        "pending_topic": None,
        "pending_difficulty": None,
    }
    topic, difficulty = _recommend(session)
    session["pending_topic"] = topic
    session["pending_difficulty"] = difficulty
    LIVE_SESSIONS[session_id] = session
    return _session_summary(session, session_id)


@app.post("/session/{session_id}/choose_topic")
def choose_topic(session_id: str, req: ChooseTopicRequest):
    """Override which topic (and optionally difficulty) the NEXT question
    will be about. Pass topic_index=null to let the agent choose the
    topic again; difficulty_label is only used when topic_index is set.
    Does not affect mastery -- only changes what's about to be asked."""
    session = LIVE_SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found. Start a new one with /session/start.")

    topic, difficulty = _recommend(session, forced_topic=req.topic_index, forced_difficulty_label=req.difficulty_label)
    session["pending_topic"] = topic
    session["pending_difficulty"] = difficulty
    return _session_summary(session, session_id)


@app.post("/session/{session_id}/answer")
def answer_session(session_id: str, req: AnswerRequest):
    session = LIVE_SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found. Start a new one with /session/start.")

    topic = session["pending_topic"]
    difficulty = session["pending_difficulty"]

    old_est = session["est_mastery"][topic]
    session["est_mastery"][topic] = estimate_update(old_est, difficulty, req.correct)

    session["history"].append({
        "topic": session["topic_names"][topic],
        "difficulty": round(difficulty, 2),
        "correct": req.correct,
    })

    # Write straight through to the student's saved record after every
    # single answer (not just at end-of-session) so nothing is lost if
    # the browser closes, the server restarts, or the teacher just ends
    # the session without a formal "save" step.
    if session.get("student_id") is not None:
        n_real = len(session["topic_names"])
        conn = _get_db()
        conn.execute(
            "UPDATE students SET est_mastery = ?, history = ?, updated_at = ? WHERE id = ?",
            (
                json.dumps(session["est_mastery"][:n_real]),
                json.dumps(session["history"]),
                _now_iso(),
                session["student_id"],
            ),
        )
        conn.commit()
        conn.close()

    # Default the next question back to "let the agent choose" -- the
    # frontend can immediately call /choose_topic again if the user wants
    # to override this specific upcoming turn too.
    next_topic, next_difficulty = _recommend(session)
    session["pending_topic"] = next_topic
    session["pending_difficulty"] = next_difficulty

    return _session_summary(session, session_id)


@app.get("/session/{session_id}")
def get_session(session_id: str):
    session = LIVE_SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return _session_summary(session, session_id)


@app.delete("/session/{session_id}")
def end_session(session_id: str):
    LIVE_SESSIONS.pop(session_id, None)
    return {"status": "ended"}


def _topic_accuracy(topic_names: list[str], history: list[dict]) -> dict[str, dict]:
    """Plain right/wrong accuracy per topic, matching what the frontend
    mastery bars show (not the agent's internal IRT-style est_mastery)."""
    stats = {name: {"correct": 0, "total": 0} for name in topic_names}
    for h in history:
        if h["topic"] in stats:
            stats[h["topic"]]["total"] += 1
            if h["correct"]:
                stats[h["topic"]]["correct"] += 1
    return stats


class SummaryResponse(BaseModel):
    summary: str


@app.post("/session/{session_id}/summary", response_model=SummaryResponse)
def session_summary_ai(session_id: str):
    session = LIVE_SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found. Start a new one with /session/start.")
    if not session["history"]:
        raise HTTPException(status_code=400, detail="No answers recorded yet -- answer at least one question first.")

    stats = _topic_accuracy(session["topic_names"], session["history"])
    lines = []
    for name, s in stats.items():
        if s["total"] == 0:
            lines.append(f"- {name}: no questions asked yet")
        else:
            pct = round(100 * s["correct"] / s["total"])
            lines.append(f"- {name}: {pct}% correct ({s['correct']}/{s['total']})")
    recent = session["history"][-8:]  # last few questions for trend context
    recent_lines = [
        f"  {i+1}. {h['topic']} (difficulty {h['difficulty']}): {'correct' if h['correct'] else 'wrong'}"
        for i, h in enumerate(recent)
    ]

    prompt = (
        "You are helping a teacher who just ran a live tutoring session with one student. "
        "Below is that student's per-topic accuracy and their most recent answers. "
        "Write a short, plain-language summary (2-4 sentences) for the teacher: "
        "call out which topics look strong vs weak, and suggest what to focus on next. "
        "Be concrete and specific to the numbers given -- no generic filler.\n\n"
        "Per-topic accuracy:\n" + "\n".join(lines) + "\n\n"
        "Most recent answers:\n" + "\n".join(recent_lines)
    )

    client = _get_cerebras_client()
    try:
        completion = client.chat.completions.create(
            model=CEREBRAS_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=200,
        )
    except APIError as e:
        raise HTTPException(status_code=502, detail=f"Cerebras API error: {e}")

    summary = completion.choices[0].message.content.strip()
    return SummaryResponse(summary=summary)


# ---------------------------------------------------------------------
# Questions generated FROM a teacher's own material (RAG)
#
# The live-session "generate a question for this topic" idea (tried
# earlier) only ever knew a bare topic label like "Algebra" -- it had
# no idea what the teacher was actually teaching, so it just guessed at
# generic textbook content. This is grounded instead: the teacher
# uploads their own notes/chapter, we chunk it, retrieve the chunks
# most relevant to whatever they ask about, and only let the LLM write
# questions from those retrieved excerpts (not general knowledge).
#
# Retrieval uses real sentence embeddings (sentence-transformers,
# all-MiniLM-L6-v2) + cosine similarity when that package is installed
# -- it captures meaning, not just shared words, so "how do plants make
# food" retrieves a photosynthesis chunk even with zero words in
# common. If sentence-transformers isn't installed, this falls back to
# plain keyword overlap (same pattern as the pypdf fallback below) so
# the feature still works, just less precisely.
# ---------------------------------------------------------------------

def _init_materials_table():
    conn = _get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS materials (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            chunks TEXT NOT NULL,           -- JSON array of strings
            chunk_embeddings TEXT,          -- JSON array of arrays, or NULL if embeddings weren't available
            preview TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


_init_materials_table()


def _get_material(material_id: str) -> dict | None:
    conn = _get_db()
    row = conn.execute("SELECT * FROM materials WHERE id = ?", (material_id,)).fetchone()
    conn.close()
    if row is None:
        return None
    return {
        "id": row["id"],
        "name": row["name"],
        "chunks": json.loads(row["chunks"]),
        "chunk_embeddings": json.loads(row["chunk_embeddings"]) if row["chunk_embeddings"] else None,
        "preview": row["preview"],
        "created_at": row["created_at"],
    }


_STOPWORDS = {
    "the", "a", "an", "of", "and", "or", "to", "in", "on", "for", "is", "are",
    "was", "were", "be", "this", "that", "it", "as", "with", "by", "at", "from",
    "which", "how", "what", "when", "where", "why", "who", "will", "can", "do",
}

_EMBEDDER = None
_EMBEDDER_LOAD_FAILED = False


def _get_embedder():
    """Lazily load the sentence-transformers embedding model. Returns
    None (once, cheaply) if the package isn't installed, so callers can
    fall back to keyword retrieval without repeatedly retrying."""
    global _EMBEDDER, _EMBEDDER_LOAD_FAILED
    if _EMBEDDER is not None:
        return _EMBEDDER
    if _EMBEDDER_LOAD_FAILED:
        return None
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        _EMBEDDER_LOAD_FAILED = True
        return None
    _EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBEDDER


def _chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list:
    """Split into overlapping character chunks on whitespace-normalized
    text. Simple and format-agnostic -- good enough for retrieval."""
    text = re.sub(r"\s+", " ", text).strip()
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
    return chunks


def _tokenize(s: str) -> set:
    return {w for w in re.findall(r"[a-zA-Z0-9]+", s.lower()) if w not in _STOPWORDS and len(w) > 1}


def _retrieve_chunks_keyword(chunks: list, query: str, top_k: int = 3) -> list:
    """Fallback retrieval when embeddings aren't available: score each
    chunk by how many query terms it contains, return the top_k
    highest scoring. Falls back further to the first top_k chunks if
    nothing overlaps at all (e.g. a vague query)."""
    query_terms = _tokenize(query)
    if not query_terms:
        return chunks[:top_k]
    scored = [(len(query_terms & _tokenize(c)), c) for c in chunks]
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [c for score, c in scored[:top_k] if score > 0]
    return top if top else chunks[:top_k]


def _retrieve_chunks(material: dict, query: str, top_k: int = 3) -> list:
    """Retrieve the top_k chunks most relevant to `query`. Uses cosine
    similarity over sentence embeddings when the material has them
    (i.e. sentence-transformers was installed at upload time),
    otherwise falls back to keyword overlap."""
    chunks = material["chunks"]
    chunk_embeddings = material.get("chunk_embeddings")
    embedder = _get_embedder()
    if chunk_embeddings is not None and embedder is not None:
        query_embedding = embedder.encode([query])[0]
        emb_matrix = np.array(chunk_embeddings)
        query_vec = np.array(query_embedding)
        norms = np.linalg.norm(emb_matrix, axis=1) * np.linalg.norm(query_vec) + 1e-9
        sims = (emb_matrix @ query_vec) / norms
        top_idx = np.argsort(-sims)[:top_k]
        return [chunks[i] for i in top_idx]
    return _retrieve_chunks_keyword(chunks, query, top_k=top_k)


class MaterialUploadResponse(BaseModel):
    material_id: str
    name: str
    n_chunks: int
    preview: str
    retrieval_mode: str  # "embeddings" or "keyword" -- which retrieval this material will use


class MaterialSummary(BaseModel):
    material_id: str
    name: str
    n_chunks: int
    preview: str
    created_at: str


@app.get("/materials", response_model=list[MaterialSummary])
def list_materials():
    """Previously uploaded materials, most recent first -- lets the
    teacher reuse one across restarts instead of re-uploading."""
    conn = _get_db()
    rows = conn.execute("SELECT * FROM materials ORDER BY created_at DESC").fetchall()
    conn.close()
    return [
        MaterialSummary(
            material_id=row["id"], name=row["name"],
            n_chunks=len(json.loads(row["chunks"])), preview=row["preview"], created_at=row["created_at"],
        )
        for row in rows
    ]


@app.post("/materials/upload", response_model=MaterialUploadResponse)
async def upload_material(file: UploadFile = File(...)):
    """Upload a .txt/.md/.pdf file of teaching material to generate
    questions from later. Saved to the same local database as the
    student roster, so it survives server restarts."""
    MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10MB -- generous for a chapter/worksheet, cheap insurance against abuse
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large -- max {MAX_UPLOAD_BYTES // (1024*1024)}MB.")
    filename = file.filename or "document"

    if filename.lower().endswith(".pdf"):
        try:
            from pypdf import PdfReader
        except ImportError:
            raise HTTPException(
                status_code=500,
                detail="PDF support needs the `pypdf` package installed on the server (pip install pypdf).",
            )
        import io
        reader = PdfReader(io.BytesIO(raw))
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
    else:
        try:
            text = raw.decode("utf-8", errors="ignore")
        except Exception:
            raise HTTPException(status_code=400, detail="Could not read this file as text.")

    text = text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="No extractable text found in the uploaded file.")

    chunks = _chunk_text(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="Uploaded file had no usable content after cleanup.")

    embedder = _get_embedder()
    chunk_embeddings = embedder.encode(chunks).tolist() if embedder is not None else None

    material_id = str(uuid.uuid4())
    preview = text[:300]
    conn = _get_db()
    conn.execute(
        "INSERT INTO materials (id, name, chunks, chunk_embeddings, preview, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (material_id, filename, json.dumps(chunks), json.dumps(chunk_embeddings) if chunk_embeddings else None,
         preview, _now_iso()),
    )
    conn.commit()
    conn.close()

    return MaterialUploadResponse(
        material_id=material_id, name=filename, n_chunks=len(chunks), preview=text[:300],
        retrieval_mode="embeddings" if chunk_embeddings is not None else "keyword",
    )


class MaterialQuestionRequest(BaseModel):
    material_id: str
    topic_query: str  # what to focus on, e.g. "Chapter 3: Newton's Laws" or "fractions"
    difficulty_label: str = "medium"  # easy | medium | hard
    n_questions: int = 3


class MaterialQuestion(BaseModel):
    question: str
    answer: str
    source_excerpt: str


class MaterialQuestionResponse(BaseModel):
    questions: list


@app.post("/materials/generate_questions", response_model=MaterialQuestionResponse)
def generate_material_questions(req: MaterialQuestionRequest):
    material = _get_material(req.material_id)
    if material is None:
        raise HTTPException(status_code=404, detail="Material not found -- upload it again.")
    if req.difficulty_label not in DIFFICULTY_LABELS:
        raise HTTPException(status_code=400, detail=f"difficulty_label must be one of {list(DIFFICULTY_LABELS)}.")
    if not (1 <= req.n_questions <= 10):
        raise HTTPException(status_code=400, detail="n_questions must be between 1 and 10.")
    if not req.topic_query.strip():
        raise HTTPException(status_code=400, detail="topic_query cannot be empty -- say what to focus on.")

    relevant_chunks = _retrieve_chunks(material, req.topic_query, top_k=3)
    context = "\n\n---\n\n".join(relevant_chunks)
    difficulty_desc = _difficulty_description(DIFFICULTY_LABELS[req.difficulty_label])

    prompt = (
        "You are writing practice questions STRICTLY from the study material excerpts below. "
        "Do not use any outside knowledge -- base every question only on what's in the excerpts. "
        f"Focus on: \"{req.topic_query}\". Difficulty: {difficulty_desc}.\n\n"
        f"Study material excerpts:\n{context}\n\n"
        f"Write exactly {req.n_questions} practice question(s). Respond ONLY with a JSON array where each "
        "element is an object with three keys: \"question\", \"answer\" (short -- final answer or "
        "1-2 sentence solution), and \"source_excerpt\" (the short snippet from the material the question "
        "is based on). No markdown, no code fences, no extra text -- just the raw JSON array."
    )

    client = _get_cerebras_client()
    try:
        completion = client.chat.completions.create(
            model=CEREBRAS_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=220 * req.n_questions + 100,
        )
    except APIError as e:
        raise HTTPException(status_code=502, detail=f"Cerebras API error: {e}")

    raw = completion.choices[0].message.content.strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        parsed = json.loads(raw)
        questions = [
            MaterialQuestion(
                question=str(item["question"]).strip(),
                answer=str(item["answer"]).strip(),
                source_excerpt=str(item.get("source_excerpt", "")).strip(),
            )
            for item in parsed
        ]
    except (json.JSONDecodeError, KeyError, TypeError, IndexError):
        raise HTTPException(status_code=502, detail="Could not parse questions from the AI response. Try again.")

    return MaterialQuestionResponse(questions=questions)