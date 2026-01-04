# Controller Policy Training Pipeline - Complete Flow Diagram

## 📊 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CONTROLLER POLICY TRAINING                        │
│                                                                       │
│  Input: Mathematical Problems                                        │
│  Output: Trained Controller that selects optimal Tool + Compute      │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
        ┌───────────────────────────────────────────────────┐
        │         DATA GENERATION (Your Existing Code)       │
        │                                                    │
        │  • UniversalAgent with multiple configurations     │
        │  • K rollouts per problem (e.g., K=10)            │
        │  • Different tools: CoT, Self-Reflection          │
        │  • Different compute: best_of_n, beam_search      │
        │  • High temperature for diversity                  │
        └───────────────────────────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────┐
                    │   Trajectory Files         │
                    │                            │
                    │  • rollouts_train.jsonl    │
                    │  • rollouts_dev.jsonl      │
                    │  • rollouts_test.jsonl     │
                    └───────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
        ┌──────────────────────┐      ┌──────────────────────┐
        │   TRAINING PIPELINE   │      │   EVALUATION ONLY    │
        │   (training_main.py)  │      │  (evaluate_only.py)  │
        └──────────────────────┘      └──────────────────────┘
```

## 🔄 Complete Training Pipeline Flow

```
╔═══════════════════════════════════════════════════════════════════╗
║                     STAGE 1: PREFERENCE PAIRS                      ║
║                  (preference_pair_generator.py)                    ║
╚═══════════════════════════════════════════════════════════════════╝
                                    │
                                    ▼
        ┌───────────────────────────────────────────────────┐
        │  For each problem:                                 │
        │    1. Load all K trajectories                      │
        │    2. Separate correct vs incorrect                │
        │    3. Compute quality scores                       │
        │       Score = correctness + efficiency - cost      │
        │    4. Create preference pairs:                     │
        │       • Correctness: best_correct vs each_incorrect│
        │       • Efficiency: best_correct vs suboptimal     │
        │       • Margin: varied difficulty levels           │
        └───────────────────────────────────────────────────┘
                                    │
                                    ▼
                ┌───────────────────────────────────┐
                │  Output: Preference Pairs         │
                │                                   │
                │  • preferences_train.jsonl        │
                │  • preferences_dev.jsonl          │
                │                                   │
                │  Each pair contains:              │
                │    - preferred_trajectory         │
                │    - rejected_trajectory          │
                │    - score_diff                   │
                │    - pair_type                    │
                └───────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
╔═══════════════════════════════════════╗ ╔════════════════════════════╗
║     STAGE 2: SFT TRAINING             ║ ║  STAGE 3: GRPO TRAINING    ║
║        (train_sft.py)                 ║ ║     (train_grpo.py)        ║
╚═══════════════════════════════════════╝ ╚════════════════════════════╝
```

## 📋 Detailed Stage Breakdown

### STAGE 1: Preference Pair Generation

```
Input: rollouts_train.jsonl (K trajectories per problem)
├─ Problem 001
│  ├─ Trajectory 1: CoT, n=1     → Correct ✓  (score: 95.0)
│  ├─ Trajectory 2: CoT, n=4     → Correct ✓  (score: 85.0)
│  ├─ Trajectory 3: Self-Ref, n=1 → Incorrect ✗ (score: -5.0)
│  ├─ Trajectory 4: Self-Ref, n=4 → Incorrect ✗ (score: -15.0)
│  └─ ... (K total trajectories)
│
└─ Processing:
   ├─ 1. Group by QID
   ├─ 2. Quality Scoring
   │     Score = 100 (if correct) + 10 (efficiency bonus)
   │           - 2×(compute_param_sum) - 1×(tool_calls) - 0.01×(steps)
   │
   ├─ 3. Create Pairs
   │     ┌────────────────────────────────────────────┐
   │     │ Correctness Pairs (4 pairs)                │
   │     │  (Traj1, Traj3), (Traj1, Traj4)           │
   │     │  (Traj2, Traj3), (Traj2, Traj4)           │
   │     ├────────────────────────────────────────────┤
   │     │ Efficiency Pairs (1 pair)                  │
   │     │  (Traj1, Traj2)  # Both correct           │
   │     ├────────────────────────────────────────────┤
   │     │ Margin Pairs (by difficulty)               │
   │     │  Easy: large score_diff (>50)              │
   │     │  Medium: moderate score_diff (20-50)       │
   │     │  Hard: small score_diff (<20)              │
   │     └────────────────────────────────────────────┘
   │
   └─ Output: preferences_train.jsonl (7 pairs for this problem)

Output Structure:
{
  "qid": "problem_001",
  "problem": "What is 2+2?",
  "preferred_trajectory": {
    "selected_tools": ["cot"],
    "compute_configs": [{"strategy": "best_of_n", "param": 1}],
    "final_answer": "4",
    "correct": true,
    "quality_score": 95.0
  },
  "rejected_trajectory": {
    "selected_tools": ["self_reflection"],
    "compute_configs": [{"strategy": "best_of_n", "param": 4}],
    "final_answer": "5",
    "correct": false,
    "quality_score": -15.0
  },
  "score_diff": 110.0,
  "pair_type": "correctness"
}
```

### STAGE 2: Supervised Fine-Tuning (SFT)

```
╔════════════════════════════════════════════════════════════════════╗
║                        SFT TRAINING FLOW                            ║
╚════════════════════════════════════════════════════════════════════╝

Input: rollouts_train.jsonl (only CORRECT trajectories)

┌─────────────────────────────────────────────────────────────────┐
│ 1. Dataset Preparation (ControllerSFTDataset)                   │
│                                                                  │
│    For each correct trajectory:                                 │
│      Extract:                                                    │
│        • problem: "What is 2+2?"                                │
│        • plan: "Add the two numbers"                            │
│        • selected_tool: "cot"                                   │
│        • compute_strategy: "best_of_n"                          │
│        • compute_param: 1                                       │
│        • reasoning: "2 + 2 = 4"                                 │
│        • final_answer: "4"                                      │
│                                                                  │
│      Format as conversation:                                     │
│      ┌──────────────────────────────────────────────────────┐  │
│      │ <|user|>                                              │  │
│      │ Problem: What is 2+2?                                 │  │
│      │ Plan: Add the two numbers                             │  │
│      │                                                        │  │
│      │ Select the best reasoning tool and compute strategy.  │  │
│      │ <|assistant|>                                         │  │
│      │ I will use the cot reasoning tool with               │  │
│      │ best_of_n(n=1) strategy.                             │  │
│      │                                                        │  │
│      │ Reasoning:                                            │  │
│      │ 2 + 2 = 4                                             │  │
│      │                                                        │  │
│      │ Final Answer: 4                                       │  │
│      └──────────────────────────────────────────────────────┘  │
│                                                                  │
│      Tokenize → [input_ids, attention_mask, labels]            │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. Model Initialization                                         │
│                                                                  │
│    Base Model: meta-llama/Llama-3.2-3B-Instruct                │
│         ↓                                                        │
│    Load with 4-bit quantization                                 │
│         ↓                                                        │
│    Apply LoRA (Low-Rank Adaptation)                            │
│      • Rank (r): 16                                             │
│      • Alpha: 32                                                │
│      • Target modules: q_proj, k_proj, v_proj, o_proj, etc.    │
│      • Trainable params: ~24M (0.75% of 3.2B)                   │
│         ↓                                                        │
│    Enable gradient checkpointing (memory efficiency)            │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. Training Loop (Hugging Face Trainer)                         │
│                                                                  │
│    Hyperparameters:                                             │
│      • Epochs: 3                                                │
│      • Batch size: 4                                            │
│      • Gradient accumulation: 4 (effective batch = 16)          │
│      • Learning rate: 2e-4                                      │
│      • Warmup steps: 100                                        │
│      • Optimizer: AdamW                                         │
│      • FP16: True                                               │
│                                                                  │
│    For each batch:                                              │
│      1. Forward pass → compute loss (causal LM)                 │
│      2. Backward pass → compute gradients                       │
│      3. Gradient accumulation (4 steps)                         │
│      4. Optimizer step → update LoRA weights                    │
│      5. Log metrics (loss, learning rate)                       │
│                                                                  │
│    Every 100 steps:                                             │
│      • Evaluate on validation set                               │
│      • Compute eval_loss                                        │
│                                                                  │
│    Every 500 steps:                                             │
│      • Save checkpoint                                          │
│      • Keep only best 3 checkpoints                             │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. Output                                                        │
│                                                                  │
│    models/experiment_name/sft/final/                            │
│      ├── adapter_config.json      # LoRA configuration          │
│      ├── adapter_model.safetensors # LoRA weights               │
│      ├── tokenizer_config.json                                  │
│      ├── tokenizer.json                                         │
│      ├── special_tokens_map.json                                │
│      └── training_metrics.json    # Loss curves, etc.           │
│                                                                  │
│    Model learns:                                                 │
│      Problem → (Tool, Compute Strategy)                         │
│      Using supervised learning on CORRECT examples              │
└─────────────────────────────────────────────────────────────────┘
```

### STAGE 3: Group Relative Policy Optimization (GRPO)

```
╔════════════════════════════════════════════════════════════════════╗
║                    GRPO TRAINING FLOW (via DPO)                     ║
╚════════════════════════════════════════════════════════════════════╝

Input: 
  • preferences_train.jsonl (preference pairs)
  • SFT model from Stage 2 (initialization)

┌─────────────────────────────────────────────────────────────────┐
│ 1. Dataset Preparation (ControllerGRPODataset)                  │
│                                                                  │
│    For each preference pair:                                    │
│                                                                  │
│      Problem: "What is 2+2?"                                    │
│                                                                  │
│      Preferred (correct):                                        │
│        Tool: cot, Strategy: best_of_n(n=1)                      │
│        Answer: "4" ✓                                            │
│                                                                  │
│      Rejected (incorrect):                                       │
│        Tool: self_reflection, Strategy: best_of_n(n=4)          │
│        Answer: "5" ✗                                            │
│                                                                  │
│    Format for DPO:                                              │
│      {                                                           │
│        "prompt": "Problem: ... Select tool and strategy...",    │
│        "chosen": "I will use cot with best_of_n(n=1)...",      │
│        "rejected": "I will use self_reflection with...",        │
│        "score_diff": 110.0                                      │
│      }                                                           │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. Model Setup                                                   │
│                                                                  │
│    Policy Model (π_θ):                                          │
│      • Initialize from SFT model                                │
│      • Will be updated during training                          │
│      • LoRA adapters applied                                    │
│                                                                  │
│    Reference Model (π_ref):                                     │
│      • Copy of SFT model                                        │
│      • Frozen (no gradient updates)                             │
│      • Provides baseline probabilities                          │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. DPO Training Loop                                            │
│                                                                  │
│    DPO Loss Formula:                                            │
│    ℒ_DPO = -log σ(β × [log(π_θ(y_w|x)/π_ref(y_w|x))           │
│                        - log(π_θ(y_l|x)/π_ref(y_l|x))])        │
│                                                                  │
│    Where:                                                        │
│      • y_w = chosen/preferred response                          │
│      • y_l = rejected response                                  │
│      • x = prompt (problem)                                     │
│      • β = temperature parameter (0.1)                          │
│      • σ = sigmoid function                                     │
│                                                                  │
│    Intuition:                                                    │
│      Increase probability of preferred responses relative to    │
│      rejected ones, while staying close to reference model      │
│                                                                  │
│    For each batch:                                              │
│      1. Forward pass on preferred:                              │
│         log_prob_chosen_policy = π_θ(chosen|prompt)            │
│         log_prob_chosen_ref = π_ref(chosen|prompt)             │
│                                                                  │
│      2. Forward pass on rejected:                               │
│         log_prob_rejected_policy = π_θ(rejected|prompt)        │
│         log_prob_rejected_ref = π_ref(rejected|prompt)         │
│                                                                  │
│      3. Compute DPO loss                                        │
│                                                                  │
│      4. Backprop and update policy model (π_θ) only            │
│                                                                  │
│      5. Log metrics:                                            │
│         • train/loss                                            │
│         • train/rewards/chosen                                  │
│         • train/rewards/rejected                                │
│         • train/rewards/margin (chosen - rejected)              │
│                                                                  │
│    Hyperparameters:                                             │
│      • Epochs: 3                                                │
│      • Batch size: 4                                            │
│      • Learning rate: 5e-5 (lower than SFT)                     │
│      • Beta: 0.1 (KL penalty strength)                          │
│      • Gradient accumulation: 4                                 │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. Output                                                        │
│                                                                  │
│    models/experiment_name/grpo/final/                           │
│      ├── adapter_config.json                                    │
│      ├── adapter_model.safetensors  # Refined LoRA weights     │
│      ├── tokenizer files                                        │
│      └── training_metrics.json                                  │
│                                                                  │
│    Model learns:                                                 │
│      Prefer: Correct + Efficient trajectories                   │
│      Avoid: Incorrect + Inefficient trajectories                │
│      Via relative preference optimization                        │
└─────────────────────────────────────────────────────────────────┘
```

### STAGE 4: Evaluation

```
╔════════════════════════════════════════════════════════════════════╗
║                      EVALUATION FLOW                                ║
╚════════════════════════════════════════════════════════════════════╝

Input: Trained GRPO model

┌─────────────────────────────────────────────────────────────────┐
│ 1. Controller Inference (controller_inference.py)               │
│                                                                  │
│    Load trained model:                                          │
│      • Load LoRA adapters                                       │
│      • Load tokenizer                                           │
│      • Set to eval mode                                         │
│      • Apply 4-bit quantization (optional)                      │
│                                                                  │
│    For test problem:                                            │
│      Problem: "Solve x^2 - 5x + 6 = 0"                         │
│                                                                  │
│      1. Format prompt:                                          │
│         "<|user|>                                               │
│          Problem: Solve x^2 - 5x + 6 = 0                       │
│          Select the best reasoning tool and compute strategy.   │
│          <|assistant|>"                                         │
│                                                                  │
│      2. Generate response (temperature=0.7):                    │
│         "I will use the cot reasoning tool with                 │
│          best_of_n(n=1) strategy."                             │
│                                                                  │
│      3. Parse response:                                         │
│         Extract: tool="cot", strategy="best_of_n", param=1     │
│                                                                  │
│      4. Compute confidence:                                     │
│         From generation probabilities (avg of top tokens)       │
│         confidence = 0.85                                       │
│                                                                  │
│      5. Return ControllerDecision:                             │
│         {                                                        │
│           tool: "cot",                                          │
│           compute_strategy: "best_of_n",                        │
│           compute_param: 1,                                     │
│           confidence: 0.85                                      │
│         }                                                        │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. Evaluation Metrics (evaluate_controller.py)                  │
│                                                                  │
│    A. Tool Selection Accuracy (vs ground truth):                │
│       • Load ground truth from correct trajectories             │
│       • Compare predicted tool vs actual best tool              │
│       • Metrics:                                                │
│         - Tool accuracy: 82%                                    │
│         - Strategy accuracy: 78%                                │
│         - Exact match (tool + strategy + param): 65%            │
│                                                                  │
│    B. Solution Quality (end-to-end):                            │
│       For each test problem:                                    │
│         1. Get controller decision                              │
│         2. Create UniversalAgent with decision                  │
│         3. Solve problem                                        │
│         4. Check correctness                                    │
│       • Metrics:                                                │
│         - Accuracy: 85%                                         │
│         - Avg compute cost: 2.3 (vs baseline 4.0)               │
│         - Compute reduction: 42.5%                              │
│                                                                  │
│    C. Confidence Calibration:                                   │
│       Bin predictions by confidence level:                      │
│       ┌─────────────────────────────────────┐                  │
│       │ Confidence  │  Accuracy  │  Count   │                  │
│       ├─────────────────────────────────────┤                  │
│       │ 0.0 - 0.1   │    12%     │    5     │                  │
│       │ 0.1 - 0.2   │    25%     │    8     │                  │
│       │ 0.2 - 0.3   │    35%     │   12     │                  │
│       │ 0.3 - 0.4   │    42%     │   15     │                  │
│       │ 0.4 - 0.5   │    55%     │   18     │                  │
│       │ 0.5 - 0.6   │    62%     │   22     │                  │
│       │ 0.6 - 0.7   │    70%     │   28     │                  │
│       │ 0.7 - 0.8   │    78%     │   32     │                  │
│       │ 0.8 - 0.9   │    85%     │   25     │                  │
│       │ 0.9 - 1.0   │    92%     │   15     │                  │
│       └─────────────────────────────────────┘                  │
│       • Expected Calibration Error (ECE): 0.08                  │
│       • Well-calibrated: confidence ≈ accuracy                  │
│                                                                  │
│    D. Analysis by Problem Type:                                 │
│       • Algebra: 88% accuracy, avg_cost=1.8                     │
│       • Geometry: 82% accuracy, avg_cost=2.5                    │
│       • Calculus: 79% accuracy, avg_cost=3.1                    │
│                                                                  │
│    Output: evaluation_results.json                              │
└─────────────────────────────────────────────────────────────────┘
```

## 🔄 Integration with Production System

```
╔════════════════════════════════════════════════════════════════════╗
║              USING TRAINED CONTROLLER IN PRODUCTION                 ║
╚════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────┐
│ 1. Load Trained Controller                                      │
│                                                                  │
│    from controller_inference import TrainedControllerPolicy     │
│                                                                  │
│    controller = TrainedControllerPolicy(                        │
│        model_path="models/experiment/grpo/final",               │
│        temperature=0.7,                                         │
│        max_new_tokens=256                                       │
│    )                                                             │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. Get Decision for New Problem                                 │
│                                                                  │
│    problem = "Find the integral of x^2 dx"                      │
│                                                                  │
│    decision = controller.predict(                               │
│        problem=problem,                                         │
│        plan=None,  # Optional planning context                  │
│        return_reasoning=False                                   │
│    )                                                             │
│                                                                  │
│    # Returns:                                                    │
│    # ControllerDecision(                                        │
│    #     tool="cot",                                            │
│    #     compute_strategy="best_of_n",                          │
│    #     compute_param=2,                                       │
│    #     confidence=0.87                                        │
│    # )                                                           │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. Use Decision with UniversalAgent                             │
│                                                                  │
│    # Option A: Direct integration                               │
│    tool, compute_config = controller.get_tool_and_compute_config(│
│        problem=problem                                          │
│    )                                                             │
│                                                                  │
│    agent = UniversalAgent(                                      │
│        client_factory=your_client,                              │
│        use_planner=False,                                       │
│        use_tool_selector=False,  # Replaced by controller       │
│        use_compute_selector=False,  # Replaced by controller    │
│        fixed_tool=tool,                                         │
│        fixed_compute=compute_config                             │
│    )                                                             │
│                                                                  │
│    # Option B: With fallback for low confidence                 │
│    from controller_inference import ControllerIntegrator        │
│                                                                  │
│    integrator = ControllerIntegrator(                           │
│        controller=controller,                                   │
│        fallback_tool="cot",                                     │
│        fallback_compute={"strategy": "best_of_n", "param": 1},  │
│        min_confidence=0.3                                       │
│    )                                                             │
│                                                                  │
│    tool, compute = integrator.select_tool_and_compute(problem)  │
│                                                                  │
│    agent = UniversalAgent(                                      │
│        client_factory=your_client,                              │
│        fixed_tool=tool,                                         │
│        fixed_compute=compute                                    │
│    )                                                             │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. Solve Problem                                                 │
│                                                                  │
│    from src.help_functions.prompts_templates import *           │
│                                                                  │
│    prompts = {                                                  │
│        "system_prompt": MATH_SYSTEM_PROMPT,                     │
│        "final_answer_system_prompt": FINAL_ANSWER_SYSTEM_PROMPT,│
│        # ... other prompts                                      │
│    }                                                             │
│                                                                  │
│    response = agent.solve(problem, prompts)                     │
│                                                                  │
│    # Get final answer                                           │
│    answer = response.answer_unstructured                        │
│    # "∫x² dx = (x³/3) + C"                                      │
│                                                                  │
│    # Get reasoning trace                                        │
│    reasoning = response.reasoning                               │
│                                                                  │
│    # Get metadata                                               │
│    tools_used = response.metadata.get("tools_used", [])         │
│    compute_cost = sum(c.get("param", 0) for c in               │
│                       response.metadata.get("compute_configs_used", []))│
└─────────────────────────────────────────────────────────────────┘
```

## 📊 Data Flow Diagram

```
┌────────────────────────────────────────────────────────────────────┐
│                        COMPLETE DATA FLOW                          │
└────────────────────────────────────────────────────────────────────┘

MATH Dataset
    │
    ├─► Problem 1: "What is 2+2?"
    │      │
    │      └─► Generate K=10 trajectories
    │             │
    │             ├─ Config 1: cot, n=1     → "4" ✓
    │             ├─ Config 2: cot, n=4     → "4" ✓
    │             ├─ Config 3: self_ref, n=1 → "5" ✗
    │             ├─ Config 4: self_ref, n=4 → "3" ✗
    │             ├─ Config 5: tool_sel, n=1 → "4" ✓
    │             ├─ Config 6: tool_sel, n=4 → "4" ✓
    │             ├─ Config 7: full, n=1    → "4" ✓
    │             ├─ Config 8: full, n=4    → "4" ✓
    │             ├─ Config 9: restricted   → "5" ✗
    │             └─ Config 10: low_budget  → "3" ✗
    │                    │
    │                    ├─► Quality Scoring
    │                    │      Config 1: 95.0 (correct, efficient)
    │                    │      Config 2: 85.0 (correct, less efficient)
    │                    │      Config 3: -5.0 (incorrect)
    │                    │      Config 4: -15.0 (incorrect, expensive)
    │                    │      ...
    │                    │
    │                    ├─► Preference Pairs (7 pairs)
    │                    │      (Config1, Config3): Correctness, diff=100.0
    │                    │      (Config1, Config4): Correctness, diff=110.0
    │                    │      (Config1, Config9): Correctness, diff=100.0
    │                    │      (Config1, Config10): Correctness, diff=98.0
    │                    │      (Config1, Config2): Efficiency, diff=10.0
    │                    │      (Config2, Config3): Margin-easy, diff=90.0
    │                    │      (Config5, Config4): Margin-hard, diff=15.0
    │                    │
    │                    └─► SFT Examples (4 correct)
    │                           Config 1, Config 2, Config 5, Config 6, 
    │                           Config 7, Config 8
    │
    ├─► Problem 2: "Solve x² - 5x + 6 = 0"
    │      └─► ... (same process)
    │
    ├─► ... (1000 training problems)
    │
    └─► Problem 1000
           └─► ... (same process)

Final Dataset Sizes:
  • SFT Examples: 1000 problems × ~5 correct trajs = ~5,000 examples
  • Preference Pairs: 1000 problems × ~7 pairs = ~7,000 pairs
```

## 🎯 Key Metrics Tracking

```
┌────────────────────────────────────────────────────────────────────┐
│                      TRAINING METRICS                               │
└────────────────────────────────────────────────────────────────────┘

SFT Training:
  ├─ train_loss: 2.5 → 0.8 → 0.3
  ├─ eval_loss: 2.7 → 1.0 → 0.5
  ├─ learning_rate: 2e-4 → ... → 1e-5 (with warmup)
  └─ gradient_norm: tracked for stability

GRPO Training:
  ├─ train_loss: 0.6 → 0.4 → 0.2
  ├─ eval_loss: 0.7 → 0.5 → 0.3
  ├─ rewards_chosen: 0.5 → 0.8 → 1.2
  ├─ rewards_rejected: -0.3 → -0.5 → -0.8
  ├─ rewards_margin: 0.8 → 1.3 → 2.0  (↑ is better)
  └─ learning_rate: 5e-5 → ... → 5e-6

Evaluation:
  ├─ tool_accuracy: 82%
  ├─ strategy_accuracy: 78%
  ├─ exact_match: 65%
  ├─ solution_accuracy: 85%
  ├─ compute_efficiency: 42.5% reduction
  ├─ avg_compute_cost: 2.3 (vs 4.0 baseline)
  └─ ECE (calibration): 0.08
```

## 🔄 Full Pipeline Command Reference

```bash
# ============================================
# COMPLETE PIPELINE (All Stages)
# ============================================
python training_main.py --config training_config.yaml

# ============================================
# INDIVIDUAL STAGES
# ============================================

# Stage 1: Preferences only
python training_main.py --config training_config.yaml --stage preferences

# Stage 2: SFT only
python training_main.py --config training_config.yaml --stage sft

# Stage 3: GRPO only
python training_main.py --config training_config.yaml --stage grpo

# Stage 4: Evaluation only
python training_main.py --config training_config.yaml --stage evaluate

# ============================================
# SKIP STAGES
# ============================================

# Skip preferences (already generated)
python training_main.py --config training_config.yaml --skip-preferences

# Skip SFT (use existing SFT model)
python training_main.py --config training_config.yaml --skip-sft

# Run only GRPO (assumes SFT exists)
python training_main.py --config training_config.yaml \
    --skip-preferences --skip-sft

# ============================================
# CUSTOM CONFIGURATION
# ============================================

# Custom experiment name
python training_main.py --config training_config.yaml \
    --experiment my_experiment_v1

# Custom output directory
python training_main.py --config training_config.yaml \
    --output-dir ./custom_models

# Debug mode (verbose logging)
python training_main.py --config training_config.yaml --debug
```

This comprehensive flow diagram shows every detail of the training pipeline from raw trajectories to production-ready controller!