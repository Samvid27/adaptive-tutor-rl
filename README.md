# tution_tutor()

**An adaptive tutoring system where a reinforcement-learning agent decides what topic and difficulty to teach a student next — trained on a simulated student, usable live with a real one.**

[Report a bug](https://github.com/Samvid27/adaptive-tutor-rl/issues)

---

## The idea

Every time you give a student a question, you're making two decisions: **which topic**, and **how hard**. Too easy and they learn nothing (they already knew it). Too hard and they also learn nothing (pure frustration). The best question sits just above what they currently know.

This project trains three different reinforcement-learning algorithms (PPO, DQN, A2C) to make that decision automatically — no hand-coded rules, just reward signal from a simulated student's actual learning — and compares them against simpler baselines (random, and a "always ask the weakest topic" heuristic) and an oracle upper bound.

The best-performing agent (DQN) recovers roughly **86% of the theoretical maximum** learning gain, using only noisy estimated mastery, not the ground-truth ability a real teacher never gets to see either.

## What's actually in here

**1. `compare_all_6`** — Runs a 30-question simulated tutoring session against six strategies at once (Random, a fixed-difficulty heuristic, PPO, A2C, DQN, and an Oracle upper bound) and plots real learning-gain-over-time for each, so the difference between "adaptive" and "not adaptive" is visible in one glance.

**2. `real_student_live`** — Use it with an actual student. Add them to a persistent roster (SQLite-backed, survives restarts), and the trained agent recommends one question at a time — pick your own number of subjects (1 to 15), name them whatever you're actually teaching, and either let the agent choose the topic/difficulty or override either one yourself. Mark each answer right or wrong and watch the mastery estimate update in real time. Includes an AI-generated plain-English summary of how the session is going, for the teacher.

**3. `questions_from_material`** — Upload your own teaching material (`.txt`/`.md`/`.pdf`), and generate an actual practice question grounded in that content at a specific difficulty level, instead of an abstract "Topic X, Difficulty 0.7."

## A finding worth mentioning: catching reward hacking

Early in training, one agent (DQN) appeared to *beat* the oracle upper bound — which shouldn't be possible, since the oracle gets to see the student's true hidden ability directly. Digging in, the agent had found a loophole: the reward was originally computed from an *observable estimate* of mastery, and that estimate could be inflated by repeatedly asking easy, guaranteed-correct questions, without the student learning anything real.

The fix: train on the student's true simulated learning gain (only knowable because this is a simulation) while keeping the agent's actual *observation* limited to the noisy estimate — the same partial information a real tutoring system would have. This is a standard technique (privileged reward, restricted observation) and it closed the exploit completely; all reported results reflect the corrected environment.

## Architecture

```
env/student.py       -- IRT-style synthetic student (hidden true ability + observable mastery estimate)
env/tutor_env.py      -- Gymnasium environment, MAX_TOPICS=15 slots, randomized active-topic-count training
training/             -- PPO / DQN / A2C training + evaluation scripts, baseline comparisons
api/main.py            -- FastAPI backend: /simulate, /session/* (live mode + roster), /materials/* (RAG)
frontend/index.html    -- Single-page frontend, all three tabs
```

## Tech stack

- **RL**: `stable-baselines3` (PPO, DQN, A2C), `gymnasium`
- **Backend**: FastAPI, SQLite (student roster + uploaded materials/embeddings)
- **AI features**: Cerebras API (LLM-generated questions from uploaded material + session summaries)
- **Frontend**: Vanilla HTML/JS, Chart.js — no build step

## Running it locally

```bash
pip install -r api/requirements.txt
```

Set your Cerebras API key (needed for the material-upload and AI-summary features):
```bash
# PowerShell
$env:CEREBRAS_API_KEY = "your-key-here"
```

**Terminal 1 — backend:**
```bash
python -m uvicorn api.main:app --reload --port 8000
```

**Terminal 2 — frontend:**
```bash
cd frontend
python -m http.server 5500
```

Open `http://localhost:5500`.

## Deployment

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/Samvid27/adaptive-tutor-rl)

The backend serves the frontend directly from `GET /`, so you only need **one service**:

1. Click the button above (or go to [render.com](https://render.com) → **New** → **Blueprint** → connect this repo)
2. Render reads the included `render.yaml` and sets up the service automatically
3. Enter your `CEREBRAS_API_KEY` when prompted (needed for AI-generated questions and session summaries; the simulation tab works without it)
4. Deploy — you'll get a URL like `https://adaptive-tutor-rl.onrender.com`

**Known limitation:** on Render's free tier, the service spins down after 15 minutes of inactivity (~30-60s cold start), and the SQLite database resets on redeploy/restart (no persistent disk). Fine for a demo; a real deployment would use a managed Postgres instance or a paid persistent volume.

## Roadmap

Things a real tuition-teacher tool would need next, roughly in priority order:
- Multi-student batch mode (quick-switch between several students in one sitting)
- Curriculum templates (e.g. preset topic lists per grade/board)
- Diagnostic starter quiz to calibrate initial mastery instead of a flat guess
- Parent-facing progress reports (PDF/WhatsApp-friendly)
- Teacher accounts/auth (currently single-tenant, no login)
- Mobile-first redesign of the live tab specifically (this is meant to be used mid-session on a phone)

## License

MIT
