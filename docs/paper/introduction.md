# 1 Introduction (draft, Siddanth)

Status (14 Nov 2026): numbers from `docs/paper/results_discussion.md` (Saanvi, three arms),
`results/crossover/crossover.md` and `results/compute/cost_table.csv`. One sentence depends on the
RQ3 predictor rerun with three-arm labels and is marked **TBD**.

---

Language agents act better when they can predict the consequences of an action before taking it. A
*world model* gives them that ability: given the goal, the current observation and a candidate action,
it predicts the next observation, so the agent can imagine several actions, compare the imagined
outcomes and only then act. In text environments, where every observation and action is a string, a
large language model can play this role directly.

A team that wants such a world model for a new environment has two options. It can prompt a large,
general LLM, optionally giving it similar transitions retrieved from whatever data it has collected.
Or it can collect transitions from the target environment and train a smaller *specialist*, either by
fine-tuning a small LLM to write the next observation or by learning a predictor in embedding space,
in the style of JEPA. Recent work shows that neither option wins everywhere: fine-tuning beat
retrieval in most of the settings studied by Ashok et al. (2026), Li et al. (2026) found that the
better choice depends on whether the environment is structured or open-ended, and Wang et al. (2024)
showed that LLM simulators fail most on state changes the environment makes on its own and on
numeric quantities. What these studies cannot tell a team is which option will win *on their
environment* before they have paid for data collection and training. In practice that question is
answered after the fact, by building the specialist and comparing.

We study this decision in four text environments (ALFWorld, ScienceWorld on new and on held-out seen
variations, and ByteSized32) with three ways of building the world model from the same n transitions,
n = 32 to 1024: **G**, a frozen Qwen2.5-7B-Instruct prompted with the five most similar training
transitions; **F**, Qwen2.5-1.5B fine-tuned with LoRA; and **D**, a JEPA-style predictor on a frozen
NV-Embed-v2 encoder that decodes to the nearest observation it has seen. We judge each world model by
the *decisions* it leads to, not only by the text it writes. At branched states with five candidate
actions, each arm predicts every candidate's outcome, a fixed judge rates the predicted outcomes, and
we measure the regret of the chosen action against the best true outcome. Model-based RL has shown
that more accurate predictions need not give better control (Lambert et al., 2020), and we find the
same here in places. Every result is priced in GPU-hours on the same GPU, and every transition is
labelled with the kind of change it requires (object state, containment or location, numeric or
physical, partially observable, action failure), so the gap between arms can be read per capability.

We ask three questions. **RQ1:** how much decision quality does each way of building the world model
give up, by budget and by capability? **RQ2:** is the specialist's compute worth it? **RQ3:** can cheap
signals, measured on only the first 32 transitions before any specialist is trained, predict which
option wins on an environment the predictor has not seen?

Our findings are mostly negative for the specialist, and we report them as such. The general model
and the fine-tuned model are statistically tied in 19 of 24 (environment, budget) cells; in the other
5, F is better, and G never is. Both beat the JEPA specialist at every budget in every environment, and
D never crosses over to either of them, even though D reaches 87–97% of its decoding ceiling. Our
ablations (an InfoNCE loss, and a learned text decoder that is given even the *true* next-state
vector) locate the limit in the frozen general-purpose embedding: it drops exact details such as
object numbers and quantities. On cost, F is the cheaper of the two generalists in our setting: at
n = 256 it needs 0.08, 0.21 and 0.10 GPU-hours on ALFWorld, ScienceWorld and ByteSized32, against
0.33, 1.08 and 0.14 for G. G needs no training, but it pays for NV-Embed retrieval indexing and
7B generation, and the generation part grows with every prediction (`results/crossover/crossover.md`).
So in our setting the small fine-tuned model is as good as prompting a much larger model and cheaper
to run, and the difference grows with the number of predictions. For RQ3, **TBD (three-arm predictor rerun):** with
F and D alone the decision is trivial (F wins 31 of 32 labelled rows), and only the JEPA probe ranked
the size of D's gap out of sample (Spearman 0.47, p = 0.01, with a sign that changes between D's
decoding modes).

**Contributions.**
1. A compute-priced comparison of three ways to obtain a text world model (prompting with retrieval,
   fine-tuning, and a JEPA-style latent predictor), judged by decision regret at branched states
   across four regimes and six data budgets, with paired significance tests.
2. A capability-level reading of the gap, using a rule-based capability tagger that we validate
   against ByteSized32's ground-truth state changes: it agrees on the main capability for 86% of
   2,954 transitions. Doing so also shows that 73% of ByteSized32's numeric state changes never appear
   in the text the agent observes, a limit on any text-only world model.
3. An honest test of whether the choice can be made ex ante. Four cheap features measured on 32
   transitions (a short JEPA probe, the capability mix, retrieval Recall@5 and simple environment
   statistics) are evaluated with leave-one-group-out and leave-one-environment-out cross-validation,
   majority and single-feature baselines, and permutation tests.
4. Code, data splits, branched states, judged outcomes, a per-step GPU-hour log and the feature
   matrix, so every number in the paper can be rebuilt.

We do not claim to be the first to compare general and specialist world models: Ashok et al. (2026)
and Li et al. (2026) do so for prompting, retrieval and fine-tuning. What we add is the decision-level,
compute-priced and per-capability comparison that includes a latent JEPA specialist, and the
question of whether the comparison can be predicted before it is paid for. The study is limited to
text environments, one seed per configuration and a single judge model (Section 7).
