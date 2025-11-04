## System Model
                         ┌─────────────────────────────┐
                         │     Environment (Crafter)   │
                         │  Observation → Agent → Act  │
                         └──────────┬──────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────────┐
                    │       CUSTOM AGENT (Main)         │
                    │  • Update history (obs + action)  │
                    │  • Get prompt from builder        │
                    │  • Determine planning mode        │
                    └───────────────┬───────────────────┘
                                    │
                    ┌───────────────▼───────────────────┐
                    │     PLANNING DECISION              │
                    │  Dynamic: Agent decides (dt∈{0,1}) │
                    │  Fixed: Every K steps (dt=1/K)     │
                    └───────────────┬───────────────────┘
                                    │
                        ┌───────────┴───────────┐
                        │                       │
                        ▼                       ▼
            ┌──────────────────┐    ┌──────────────────┐
            │  dt = 0          │    │  dt = 1          │
            │  No Planning     │    │  Planning!       │
            │  → Direct Action │    │  → New Plan      │
            └────────┬─────────┘    └────────┬─────────┘
                     │                       │
                     │                       ▼
                     │         ┌─────────────────────────┐
                     │         │    TOOL SELECTOR        │
                     │         │  Master LLM decides:    │
                     │         │  • reactive_actor       │
                     │         │  • cot                  │
                     │         │  • heuristic_script     │
                     │         └───────────┬─────────────┘
                     │                     │
                     │                     ▼
                     │         ┌─────────────────────────┐
                     │         │   COMPUTE SELECTOR      │
                     │         │  Master LLM decides:    │
                     │         │  • best_of_n:N          │
                     │         │  • beam_search:K        │
                     │         │  • lookahead:depth      │
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
                     │    │  │ ReactiveActorReasoner      │  │
                     │    │  │  • Fast action selection   │  │
                     │    │  └────────────────────────────┘  │
                     │    │  ┌────────────────────────────┐  │
                     │    │  │ CoTReasoner                │  │
                     │    │  │  • Step-by-step thinking   │  │
                     │    │  └────────────────────────────┘  │
                     │    │  ┌────────────────────────────┐  │
                     │    │  │ HeuristicScriptReasoner    │  │
                     │    │  │  • Rule-based policy       │  │
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
                     │            │ • Score resp.   │
                     │            │ • Select best   │
                     │            └────────┬────────┘
                     │                     │
                     └─────────────────────┴─────────┐
                                                     │
                                                     ▼
                                        ┌─────────────────────┐
                                        │   FINAL ACTION      │
                                        │  • Clean & format   │
                                        │  • Update metrics   │
                                        │  • Return response  │
                                        └─────────┬───────────┘
                                                  │
                                                  ▼
                                        ┌───────────────────┐
                                        │   Environment     │
                                        │  Execute action   │
                                        └───────────────────┘


## License

MIT














