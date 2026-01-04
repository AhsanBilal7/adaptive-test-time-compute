# UniversalAgent: Dataset Generation & Training Pipeline

This README defines the **dataset generation pipeline** and the **training pipeline** for the UniversalAgent. Follow exactly to ensure **clean train/dev/test separation** and **contrastive supervision** suitable for top-tier conferences.

---

## Dataset Generation Pipeline

### 1. Freeze Environment
- Build **TRAIN_Q**, **DEV_Q**, **TEST_Q** (disjoint questions).
- Build **TRAIN_CORPUS** and **TEST_CORPUS** (disjoint documents or time-split).
- Configure all tools so dataset generation accesses **only TRAIN_CORPUS**.

---

### 2. Add Trajectory Logging
Log every rollout to **JSONL** with:
- `qid`, `problem`, `plan`
- `selected_tools`, `compute_config` (per tool)
- Step-wise:
  - `tool_name`
  - `tool_input`
  - `tool_output`
  - `reasoning_steps`
- `final_answer_structured`
- `final_answer_unstructured`
- `correct` (from evaluator)
- `cost` = `{ tool_calls, compute_param_sum, reasoning_step_count }`
- `run_id`

---

### 3. Generate Multi-Rollout Trajectories (K per question)
For each `q ∈ TRAIN_Q`, run **K = 8–12** rollouts:

- **2×** `fixed_tool="cot"` with compute `{ best_of_n, N }`, `N ∈ {1, 4}`
- **2×** `fixed_tool="self_reflection"` with compute `{ best_of_n, N }`, `N ∈ {1, 4}`
- **2×** `use_tool_selector=True`, `use_compute_selector=False`, compute `N ∈ {1, 4}`
- **2× Forced negatives**:
  - Restrict tool-set (e.g., disable `web_search` or `verifier`)
  - Or low-budget / truncated execution (early stop)

---

### 4. Evaluate Each Trajectory
- Compute `correct ∈ {0,1}` using:
  - `NumericVerifier` when applicable
  - Otherwise judge/evaluator
- Compute utility:
  ```
  score = correct
          - λ * tool_calls
          - μ * compute_param_sum
          - ν * reasoning_step_count
  ```

---

### 5. Create Contrastive Preference Pairs
For each `qid`:
- `τ+` = highest `score` among `correct = 1` (if any)
- `τ-` =
  - best `score` among `correct = 0` (hard negative), else
  - a `correct = 1` trajectory with much higher cost (efficiency negative)

Store **pairs JSONL**:
```
{ qid, preferred_run_id, rejected_run_id, preference_type }
```

---

### 6. Final Artifacts
- `rollouts_train.jsonl` — all TRAIN trajectories
- `prefs_train.jsonl` — contrastive pairs (TRAIN)
- `rollouts_dev.jsonl`, `prefs_dev.jsonl` — generated from **DEV_Q** using **TRAIN_CORPUS**
- **Never generate anything from TEST_Q** except final evaluation.

---

## Training Pipeline

### 1. Train ToolSelector (Router)
- **Data**: `prefs_train.jsonl` + trajectories from `rollouts_train.jsonl`
- **Objective**: Preference learning (DPO / ranking)
  - Prefer `τ+` tool sequences over `τ-` for the same problem
- **Output**: Trained `ToolSelector`

---

### 2. Train ComputeSelector
- **Data**: `prefs_train.jsonl` + trajectories
- **Objective**: Preference learning on compute configs
  - Prefer `τ+` compute decisions over `τ-`
- **Output**: Trained `ComputeSelector`

---

### 3. Train PRMModel (Optional but Recommended)
- **Data**: `rollouts_train.jsonl` (uses `reasoning_steps`)
- **Labels**:
  - Trajectory-level: `correct`
  - Step-level: derived from verifier outcomes / final correctness
- **Objective**: Step-wise scoring/ranking
  - Assign higher scores to steps/actions from `τ+` than `τ-`
- **Output**: Trained `PRMModel`

---

### 4. End-to-End Validation
- Run full UniversalAgent with:
  - `use_tool_selector=True`
  - `use_compute_selector=True`
  - PRM enabled
- Evaluate on **DEV_Q** (tools restricted to **TRAIN_CORPUS**)
- Select checkpoint by **accuracy vs cost trade-off**

---

### 5. Final Test
- Lock all weights, prompts, and hyperparameters
- Switch tools to **TEST_CORPUS-only**
- Evaluate once on **TEST_Q**
- Report **accuracy vs cost curves**
