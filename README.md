# Competitive Intelligence: DeepAgents vs Prompt Chaining

A side-by-side comparison of two LLM orchestration architectures applied to the same real-world task: researching the top 5 competitors in a product category and producing a structured analyst brief.

---

## Objectives

### The Question

When you need an LLM to do multi-step, multi-entity research, how you architect the pipeline dramatically affects cost. This project makes that difference **measurable and visual** by running the same task two ways and comparing token counts, estimated USD cost, and wall-clock time.

### The Task

Both notebooks:
- Identify the top 5 competitors in a configurable product category
- Search for current pricing, recent news, and user sentiment for each (via Tavily)
- Produce a structured analyst brief with a pricing comparison table, sentiment summary, and recommendation matrix

### The Two Approaches

| | Notebook 01 — DeepAgents | Notebook 02 — Prompt Chaining |
|---|---|---|
| **Architecture** | Orchestrator spawns one isolated sub-agent per competitor | Fixed sequence of LangGraph nodes; state passed forward |
| **Context window** | Each sub-agent has its own isolated context | `research_context` field grows at every step |
| **File I/O** | Sub-agents write `reports/<competitor>.md`; orchestrator reads summaries | Everything lives in-memory in graph state |
| **Cost scaling** | Sub-agent isolation keeps per-call context flat | O(n²) — every step re-pays for all prior steps' tokens |
| **Parallelism** | Sub-agents can be scheduled independently | Fully sequential |

### What You Can Measure

Both notebooks use the **same shared `TokenCostTracker` LangChain callback** (`utils/reporting.py`) so results are directly comparable:

- Total prompt / completion / total tokens
- Estimated USD cost (GPT-4o pricing baked in, easy to update)
- Number of LLM calls and Tavily searches
- Wall-clock execution time

Notebook 02 additionally:
- Prints context size (chars) before every LLM call so the compounding is visible in output
- Generates a **context growth chart** (`context_comparison.png`) showing the divergence between the two approaches

---

## Project Structure

```
competitive-intelligence/
├── README.md
├── .env.example                          # API key template
├── requirements.txt                      # All Python dependencies
├── reports/                              # DeepAgents writes <competitor>.md files here
├── utils/
│   ├── __init__.py
│   └── reporting.py                      # Shared TokenCostTracker + LangGraph reporter
├── 01_deepagents_competitive_intel.ipynb # Approach 1: autonomous sub-agent delegation
└── 02_prompt_chaining_competitive_intel.ipynb  # Approach 2: sequential LangGraph chain
```

---

## Prerequisites

- Python 3.10+
- An **OpenAI API key** with access to `gpt-4o`
- A **Tavily API key** (free tier at [tavily.com](https://tavily.com) covers testing)

---

## Setup

### 1. Clone / open the project

```bash
cd ~/competitive-intelligence
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API keys

```bash
cp .env.example .env
```

Open `.env` and fill in both values:

```
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
```

---

## Running the Notebooks

### For a fair comparison, use the same settings in both notebooks

At the top of each notebook there is a configuration block:

```python
PRODUCT_CATEGORY   = "AI coding assistants"   # ← change to any market
NUM_COMPETITORS    = 5
MODEL_NAME         = "gpt-4o"
MAX_SEARCH_RESULTS = 5
```

Keep these **identical** across both notebooks.

---

### Notebook 01 — DeepAgents

Open `01_deepagents_competitive_intel.ipynb` and run all cells top-to-bottom.

**What happens:**
1. A DeepAgent is created with Tavily as its tool
2. The orchestrator calls `write_todos` to plan the research upfront
3. One sub-agent per competitor is spawned via `task()` — each runs independently with its own isolated context window
4. Each sub-agent searches pricing, news, and sentiment, then writes findings to `reports/<competitor>.md`
5. The orchestrator reads all report files and synthesizes the final brief
6. A token/cost report is printed and a token-distribution chart is saved to `deepagents_tokens.png`

At the end, **copy the printed `deepagents_summary` JSON** — you will paste it into notebook 02.

---

### Notebook 02 — Prompt Chaining

Open `02_prompt_chaining_competitive_intel.ipynb` and run all cells top-to-bottom.

**What happens:**
1. A LangGraph `StateGraph` is compiled with 5 nodes: `discover → research → sentiment → synthesize → matrix`
2. Each node appends its output to `research_context` in the graph state
3. Every LLM call prints the current context size so you can watch it grow
4. The final brief, sentiment analysis, and recommendation matrix are displayed
5. A token/cost report is printed
6. Two charts are generated:
   - `prompt_chaining_tokens.png` — context growth bar chart + token pie
   - `context_comparison.png` — DeepAgents vs Prompt Chaining context divergence line chart (the blog-post hero image)

**In cell 10 (Side-by-Side Comparison):** paste the `deepagents_summary` JSON from notebook 01 to generate the comparison table with winner highlighting.

---

## Output Files

| File | Produced by | Contents |
|---|---|---|
| `reports/<competitor>.md` | Notebook 01 | Per-competitor research written by each sub-agent |
| `deepagents_tokens.png` | Notebook 01 | Token split pie + per-call bar chart |
| `prompt_chaining_tokens.png` | Notebook 02 | Context growth bars + token split pie |
| `context_comparison.png` | Notebook 02 | Divergence line chart — DeepAgents vs Prompt Chaining |

---

## Shared Utilities (`utils/reporting.py`)

| Symbol | What it does |
|---|---|
| `TokenCostTracker` | LangChain callback — attach to any LLM call via `callbacks=[tracker]` |
| `build_report()` | Runs a LangGraph reporter graph and prints a formatted metrics summary |
| `build_reporter()` | Returns the compiled LangGraph reporter graph directly |
| `compare_runs()` | Takes two `tracker.summary()` dicts and renders a styled Pandas comparison table |
| `COST_PER_1K_TOKENS` | Pricing table (USD/1K tokens) — update here when model pricing changes |

---

## Customisation

- **Change the market**: Update `PRODUCT_CATEGORY` in both notebooks (e.g. `"project management SaaS"`, `"vector databases"`)
- **Change the model**: Update `MODEL_NAME` and the corresponding entry in `COST_PER_1K_TOKENS` in `utils/reporting.py`
- **Adjust depth**: `NUM_COMPETITORS` and `MAX_SEARCH_RESULTS` both affect cost — start with `3` and `3` for a quick test run
