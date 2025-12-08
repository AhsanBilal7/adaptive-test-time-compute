## System Model
                         ┌─────────────────────────────┐
                         │     Environment (MATH-500)  │
                         │  Problem → Agent → Solution │
                         └──────────┬──────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────────┐
                    │   UNIVERSAL AGENT (Main)          │
                    │  • Set problem context            │
                    │  • Build prompt with history      │
                    │  • Determine planning mode        │
                    │  • Coordinate tool execution      │
                    └───────────────┬───────────────────┘
                                    │
                    ┌───────────────▼───────────────────┐
                    │     PLANNING DECISION             │
                    │  Planner: Create strategic plan   │
                    │  No Planner: Direct to tools      │
                    └───────────────┬───────────────────┘
                                    │
                        ┌───────────┴───────────┐
                        │                       │
                        ▼                       ▼
            ┌──────────────────┐    ┌──────────────────┐
            │  No Planner      │    │  With Planner    │
            │  → Direct Solve  │    │  → Plan + Tools  │
            └────────┬─────────┘    └────────┬─────────┘
                     │                       │
                     │                       ▼
                     │         ┌─────────────────────────┐
                     │         │    TOOL SELECTOR        │
                     │         │  Master LLM decides:    │
                     │         │  • self_reflection      │
                     │         │  • cot                  │
                     │         │  • numeric_verifier     │
                     │         │  • verifier             │
                     │         │  • summarizer           │
                     │         │  • reframe              │
                     │         │  • web_search           │
                     │         │  (max 3 tools, ordered) │
                     │         └───────────┬─────────────┘
                     │                     │
                     │                     ▼
                     │         ┌─────────────────────────┐
                     │         │   COMPUTE SELECTOR      │
                     │         │  Master LLM decides:    │
                     │         │  • best_of_n:N          │
                     │         │  • beam_search:K        │
                     │         │  • lookahead:depth      │
                     │         │  (or fixed_compute)     │
                     │         └───────────┬─────────────┘
                     │                     │
                     │         ┌───────────┴───────────┐
                     │         │                       │
                     │         ▼                       ▼
                     │    ┌─────────┐            ┌─────────┐
                     │    │ Param=1 │            │ Param>1 │
                     │    │ Direct  │            │ Strategy│
                     │    └────┬────┘            └────┬────┘
                     │         │                      │
                     │         ▼                      ▼
                     │    ┌──────────────────────────────────┐
                     │    │       REASONER EXECUTION         │
                     │    │  ┌────────────────────────────┐  │
                     │    │  │ SelfReflectionReasoner     │  │
                     │    │  │  1. Initial attempt        │  │
                     │    │  │  2. Critique reasoning     │  │
                     │    │  │  3. Refined solution       │  │
                     │    │  │  → Best for complex/proofs │  │
                     │    │  └────────────────────────────┘  │
                     │    │  ┌────────────────────────────┐  │
                     │    │  │ CoTReasoner                │  │
                     │    │  │  • Step-by-step thinking   │  │
                     │    │  │  • Linear progression      │  │
                     │    │  │  → Best for standard tasks │  │
                     │    │  └────────────────────────────┘  │
                     │    └──────────────┬───────────────────┘
                     │                   │
                     │                   ▼
                     │         ┌───────────────────────┐
                     │         │  COMPUTE STRATEGY     │
                     │         │  (if Param > 1)       │
                     │         │  ┌─────────────────┐  │
                     │         │  │ BestOfN         │  │
                     │         │  │ • Generate N    │  │
                     │         │  │ • PRM scores    │  │
                     │         │  │ • Select best   │  │
                     │         │  └─────────────────┘  │
                     │         │  ┌─────────────────┐  │
                     │         │  │ BeamSearch      │  │
                     │         │  │ • Generate 2*K  │  │
                     │         │  │ • PRM scores    │  │
                     │         │  │ • Keep top-K    │  │
                     │         │  └─────────────────┘  │
                     │         │  ┌─────────────────┐  │
                     │         │  │ Lookahead       │  │
                     │         │  │ • Simulate D    │  │
                     │         │  │ • PRM scores    │  │
                     │         │  │ • Best traj     │  │
                     │         │  └─────────────────┘  │
                     │         └───────────┬───────────┘
                     │                     │
                     │                     ▼
                     │            ┌─────────────────┐
                     │            │   PRM MODEL     │
                     │            │ • Score steps   │
                     │            │ • Rank candid.  │
                     │            │ • Select best   │
                     │            └────────┬────────┘
                     │                     │
                     └─────────────────────┴─────────┐
                                                     │
                                                     ▼
                                        ┌─────────────────────┐
                                        │  AUXILIARY TOOLS    │
                                        │  (if selected)      │
                                        │  ┌───────────────┐  │
                                        │  │NumericVerifier│  │
                                        │  │ PRM-based     │  │
                                        │  └───────────────┘  │
                                        │  ┌───────────────┐  │
                                        │  │Verifier       │  │
                                        │  │ Correctness   │  │
                                        │  └───────────────┘  │
                                        │  ┌───────────────┐  │
                                        │  │Summarizer     │  │
                                        │  │ Compress      │  │
                                        │  └───────────────┘  │
                                        │  ┌───────────────┐  │
                                        │  │Reframe        │  │
                                        │  │ Reformulate   │  │
                                        │  └───────────────┘  │
                                        │  ┌───────────────┐  │
                                        │  │WebSearch      │  │
                                        │  │ External info │  │
                                        │  └───────────────┘  │
                                        └─────────┬───────────┘
                                                  │
                                                  ▼
                                        ┌─────────────────────┐
                                        │  FINAL ANSWER       │
                                        │  • Structured JSON  │
                                        │  • Unstructured txt │
                                        │  • Metadata tracked │
                                        │  • Return response  │
                                        └─────────┬───────────┘
                                                  │
                                                  ▼
                                        ┌───────────────────┐
                                        │   Evaluation      │
                                        │  • Check correct  │
                                        │  • Track metrics  │
                                        │  • Log results    │
                                        └───────────────────┘

### 

To grade solutions, we use the minerva_math functions from LMEval [22] to extract the model’s final answer. We then check correctness if the extracted answer is an exact string match to the ground truth, or if the is_equiv function from minerva_math in LMEval evaluates to true.


#### Run the experiments
```
# Create logs directory first
mkdir -p logs

# Run direct mode in detached mode
nohup python main.py --config configs/config_direct.yaml > logs/direct.log 2>&1 &

# Run fixed tool mode in detached mode
nohup python main.py --config configs/config_fixed_tool.yaml > logs/fixed_tool.log 2>&1 &

# Run fixed compute mode in detached mode
nohup python main.py --config configs/config_fixed_compute.yaml > logs/fixed_compute.log 2>&1 &

# Run dynamic mode in detached mode
nohup python main.py --config configs/config_dynamic.yaml > logs/dynamic.log 2>&1 &
```
## License

MIT














