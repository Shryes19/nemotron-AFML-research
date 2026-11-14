# Related work draft

Wang et al. (2024) situate retrieval-augmented generation as a way to augment frozen language models with non-parametric evidence. Our general arm follows that principle, but retrieves transition demonstrations for next-observation prediction rather than documents for factual generation. This makes the retrieval budget an experimental resource: it can improve local state-action coverage without changing the Qwen model weights.

Lambert et al. (2020) emphasize that language agents must ground textual decisions in environment state and action consequences. We measure that grounding at the decision interface: every arm predicts a next observation for the same branched state and candidate actions, then is evaluated by the common decision-utility procedure. The retrieval baseline therefore tests whether in-context transitions can supply useful local dynamics without specialist training.

Ashok et al. (2026) motivate comparing prompted general world models with data-budgeted specialists. We extend this comparison with an explicitly hierarchical RAG arm: observation similarity supplies plausible state matches, action similarity chooses the most relevant transitions, and a frozen model conditions on five demonstrations. Recall@5 is reported independently with a model-free same-template-and-object criterion, separating retrieval coverage from generator quality.
