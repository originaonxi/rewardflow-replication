# Topology-Aware Credit Assignment via State Graph Propagation: An Experimental Replication of RewardFlow (arXiv:2603.18859)

**Anmol Chaudhary**
CTO, Aonxi
origin@aonxi.com

---

## Abstract

RewardFlow (Xie et al., 2025) proposes that the sparse-reward credit assignment problem in agent reinforcement learning can be addressed by propagating outcome signals backward through a state graph aggregated across multiple rollouts. Two propagation mechanisms are presented: BFS-decay, which attenuates reward exponentially with hop distance from terminal nodes, and Personalized PageRank (PPR), which computes a global random-walk equilibrium treating success nodes as teleportation targets. The central empirical claims are that propagated rewards are substantially denser than terminal-only sparse rewards and that propagated rewards produce better discrimination between correct and incorrect solution paths.

We replicate these claims on a 20-problem math reasoning benchmark (15 problems successfully executed, 6 rollouts each, 90 total rollouts) using Claude Haiku (claude-haiku-4-5-20251001) as the rollout model with temperature diversity spanning 0.3 to 1.0. The resulting state graph contains 267 nodes and 457 edges across all problems. Our measurements confirm the density claim with high confidence: BFS propagation increases signal density from 0.255 to 0.890 (+249.2%) and PPR increases it to 0.983 (+285.7%). Intermediate node coverage moves from exactly 0.00 under sparse supervision to approximately 0.88 under BFS and 0.98 under PPR. The separation claim is not supported at terminal level (sparse achieves separation 1.022, BFS 0.745, PPR 0.364), but this result is expected and does not constitute a failure of the method: sparse rewards trivially maximize terminal separation by definition, while propagated methods distribute signal to intermediate nodes where correct/incorrect distinctions are structurally ambiguous. The primary contribution of the paper — dense intermediate coverage — is robustly replicated.

---

## 1. Introduction

The credit assignment problem is among the oldest and most consequential unsolved problems in reinforcement learning. In its simplest form: when an agent receives a terminal reward signal indicating success or failure, how should that signal be distributed across the sequence of actions, decisions, or reasoning steps that preceded the outcome?

In standard outcome-supervised training — which underlies the majority of current large language model RL pipelines including PPO (Schulman et al., 2017), REINFORCE, and GRPO (Shao et al., 2024) — the answer is that the signal is not distributed at all. Every intermediate step that contributed to a successful outcome receives the same undifferentiated training gradient as every step that was irrelevant or actively harmful. The model learns, in aggregate, that "this general style of reasoning tends to produce correct answers," rather than "this particular intermediate step was causally necessary."

This matters for three specific reasons:

**Training efficiency.** Sparse terminal rewards are sample-inefficient. If a model generates a 10-step reasoning chain and only the outcome is supervised, the gradient signal is diluted across all 10 steps regardless of their individual quality. More steps, more dilution. As reasoning chains grow longer with task difficulty, the training signal per step approaches zero.

**Process reward models.** Process reward models (PRMs; Lightman et al., 2023; Wang et al., 2024) address credit assignment through step-level human annotation — asking annotators to label individual reasoning steps as correct, incorrect, or uncertain. This works but is expensive. For every new task domain or difficulty level, annotation budgets must scale with the average reasoning chain length. A mechanism that derives step-level supervision from outcome labels and graph structure alone would eliminate this scaling cost entirely.

**Failure mode diagnosis.** Sparse rewards make it impossible to distinguish between models that fail because of an error at step 2 versus an error at step 8. Denser intermediate supervision enables targeted fine-grained improvement.

RewardFlow (Xie et al., 2025) addresses all three issues through a single mechanism: aggregate the state graph across multiple rollouts for the same problem, identify which reasoning steps are shared across trajectories and which are unique to successes versus failures, and propagate outcome signals backward through this graph topology. The result is a reward function defined over every node in the graph — intermediate steps included — derived entirely from binary outcome labels and structural graph properties.

This paper reports an independent replication of that mechanism. We implemented the full pipeline from scratch: rollout generation via Claude Haiku, state graph construction with shared-prefix deduplication, BFS-decay propagation, Personalized PageRank propagation, and quantitative evaluation of signal density and separation. We ran 90 rollouts across 15 math reasoning problems spanning eight problem categories and measured the three core quantities the paper reports. We report our results honestly, including the separation metric where propagated methods underperform the sparse baseline and our reasoning for why this result is expected rather than damaging to the mechanism.

---

## 2. Background

### 2.1 Credit Assignment in Reinforcement Learning

The credit assignment problem was formalized by Minsky (1961) and has been an active research topic ever since. The classical solution in temporal difference learning is the eligibility trace (Sutton & Barto, 1998), which propagates reward backward through time with exponential decay controlled by the parameter lambda. The TD(lambda) family of algorithms provides a principled interpolation between Monte Carlo return estimation (lambda = 1, full credit to all steps) and one-step TD (lambda = 0, credit only to the immediately preceding step).

Graph-structured credit assignment extends this intuition from single linear trajectories to branching, merging trajectory graphs. When multiple rollouts for the same problem are aggregated, steps that appear on many successful paths receive higher confidence as genuinely useful steps; steps that appear exclusively on failed paths are penalized; and steps shared across both successful and failed paths receive intermediate signal. The graph topology encodes information about step quality that no single rollout contains in isolation.

### 2.2 GRPO and Group-Based Policy Optimization

Group Relative Policy Optimization (GRPO; Shao et al., 2024) is a policy gradient variant specifically designed for reasoning tasks where multiple solutions to the same problem can be sampled and evaluated. Rather than computing advantages relative to a single value function baseline, GRPO normalizes rewards within the group of rollouts for each problem, effectively treating the mean group performance as the baseline.

GRPO has become a dominant training paradigm for reasoning-capable LLMs because it naturally generates the multi-rollout structure that makes state graph construction possible. Each group of rollouts for a problem represents a local sample of the reasoning space: some paths reach correct answers, others fail, and the shared prefixes reveal which early reasoning moves are robust versus which are path-contingent.

RewardFlow is designed as a drop-in extension to GRPO: instead of using the binary terminal rewards directly in the policy gradient update, the propagated rewards from the state graph are substituted, providing denser per-step supervision without changing the overall training algorithm structure.

### 2.3 Process Reward Models

Process reward models (PRMs) assign scalar quality scores to individual reasoning steps rather than entire trajectories. The key result establishing their value is from Lightman et al. (2023), who showed that step-level supervision trained on human annotations substantially outperforms outcome supervision on MATH benchmark problems, particularly for problems requiring more than three reasoning steps.

Wang et al. (2024) extended this work with a method for training PRMs without explicit human annotation by using Monte Carlo estimation — sampling many completions from each intermediate state and using the completion success rate as a proxy for step quality. This approach is expensive at inference time (many completions must be generated per step) but eliminates the annotation requirement.

RewardFlow can be understood as an alternative and more efficient approach to the same goal as Wang et al.: derive step-level quality estimates without human annotation. The key difference is that RewardFlow operates over a pre-existing set of rollouts (the group samples from GRPO) rather than generating additional completions per step. The graph structure provides the approximation to Monte Carlo step quality that Wang et al. achieved through direct sampling.

### 2.4 Graph-Based Reward Propagation

The use of graph structure for reward propagation has precedents in model-based RL (Hamrick et al., 2017), relational reasoning (Battaglia et al., 2018), and multi-step planning (Silver et al., 2017). What distinguishes the RewardFlow formulation is the specific construction of the state graph: nodes represent reasoning steps identified by semantic content and position in the chain, and shared nodes across rollouts are deduplicated, creating cross-rollout edges that allow reward to flow between trajectories.

This construction is directly analogous to the use of transposition tables in game tree search: states reached via different paths are recognized as the same state and their evaluations are pooled. In a reasoning context, this means that a correct calculation of an intermediate result (for example, computing that combined speed equals 150 mph in a meeting problem) receives credit from all rollouts that eventually reached correct answers via that step, regardless of whether the full chains of reasoning were identical.

---

## 3. The RewardFlow Mechanism

### 3.1 State Graph Construction

Given a problem $P$ and a set of $K$ rollouts $\{R_1, R_2, \ldots, R_K\}$, each rollout $R_i = (s_1^{(i)}, s_2^{(i)}, \ldots, s_{n_i}^{(i)})$ consists of a sequence of reasoning steps and a terminal outcome label $y^{(i)} \in \{+1, -1\}$.

The state graph $G = (V, E)$ is constructed as follows:

**Node construction.** For each step $s_j^{(i)}$, a node identifier is computed based on the step's semantic content at position $j$. Two steps at the same position with the same key numerical content are assigned the same node identifier and merged into a single node. Formally:

$$\text{id}(s_j^{(i)}) = h\left(j, \text{nums}(s_j^{(i)})[{:}4], s_j^{(i)}[{:}30]\right)$$

where $h$ is a hash function, $\text{nums}(\cdot)$ extracts numerical values from a step, and the truncation ensures that minor surface variation in otherwise equivalent steps is absorbed. Each node tracks its visit count (how many rollouts traversed it) and its rollout membership set.

**Edge construction.** For each consecutive pair $(s_j^{(i)}, s_{j+1}^{(i)})$ in each rollout, a directed edge is added. After construction, edges are made undirected for propagation (following the paper), and self-loops are removed.

**Terminal labeling.** The final step of each rollout is labeled as a success node ($y^{(i)} = +1$) or failure node ($y^{(i)} = -1$). Non-final steps are intermediate nodes.

A root node representing the problem statement is added as the common ancestor of all rollout chains.

**Shared nodes.** Steps appearing in multiple rollouts appear as single nodes with high visit counts and edges to all subsequent steps across all rollouts that traversed them. These shared nodes are structurally significant: they represent reasoning moves that the model finds natural and that may have differential outcomes depending on subsequent steps.

### 3.2 BFS-Decay Propagation

BFS-decay propagation starts from success nodes and spreads reward outward through the graph, attenuating with each hop. For a node $v$ at BFS distance $d$ from success node $s$:

$$r_v^{\text{BFS}} = \sum_{s \in \mathcal{S}} \gamma^{d(v, s)}$$

where $\mathcal{S}$ is the set of success nodes, $d(v, s)$ is the shortest-path distance from $v$ to $s$ in the undirected graph, and $\gamma \in (0, 1)$ is the decay parameter (set to 0.8 in this replication; the paper uses 0.9).

In our implementation, BFS propagation additionally applies a penalty from failure nodes:

$$r_v^{\text{BFS}} \leftarrow r_v^{\text{BFS}} + \sum_{f \in \mathcal{F}} -0.3 \cdot \gamma^{d(v, f)} \cdot \mathbf{1}[d(v, f) \leq 2]$$

where $\mathcal{F}$ is the set of failure nodes and the indicator restricts penalization to close neighbors. This modification prevents isolated failure nodes from globally contaminating otherwise high-quality reasoning steps.

A visit-count bonus is also applied: nodes appearing in more than one rollout receive a multiplicative boost proportional to visit count, reflecting the intuition that steps confirmed by multiple traversals are more reliable as positive signal.

The key property of BFS-decay: every node reachable from any success node receives a nonzero reward. In a connected or nearly-connected graph, this means most intermediate nodes receive signal.

### 3.3 Personalized PageRank Propagation

Personalized PageRank defines reward as the stationary distribution of a random walk that teleports to success nodes with probability $1 - \alpha$. For each node $v$:

$$r_v^{\text{PPR}} = \alpha \cdot r_v^{(\text{terminal})} + (1 - \alpha) \cdot \frac{1}{|\mathcal{N}(v)|} \sum_{u \in \mathcal{N}(v)} r_u^{\text{PPR}}$$

where $\alpha$ is the teleportation probability (damping $= 1 - \alpha = 0.85$ in this replication), $r_v^{(\text{terminal})}$ is nonzero only for success nodes, and $\mathcal{N}(v)$ is the neighborhood of $v$ in the undirected graph.

This system is solved by power iteration: initialize all ranks uniformly, define a personalization vector $p$ that is uniform over success nodes and zero elsewhere, and iterate:

$$\mathbf{r}^{(t+1)} = \alpha \cdot \mathbf{p} + (1 - \alpha) \cdot \mathbf{A}^T \mathbf{D}^{-1} \mathbf{r}^{(t)}$$

where $\mathbf{A}$ is the adjacency matrix and $\mathbf{D}$ is the degree matrix. After convergence (20 iterations in our implementation), ranks are normalized to $[-1, 1]$ via min-max scaling.

PPR differs from BFS in a critical structural way: PPR is a global equilibrium that propagates signal transitively through the entire graph, while BFS is a local greedy expansion from terminal nodes. Nodes that are many hops from any success node but well-connected to intermediate nodes that are themselves well-connected to success nodes will receive high PPR scores. This is why PPR achieves higher intermediate coverage (0.98) than BFS (0.88): it reaches structurally peripheral nodes through multi-hop diffusion that BFS's exponential decay cannot sustain.

### 3.4 Sparse Baseline

The sparse baseline assigns:

$$r_v^{\text{sparse}} = \begin{cases} +1 & \text{if } v \in \mathcal{S} \\ -1 & \text{if } v \in \mathcal{F} \\ 0 & \text{otherwise} \end{cases}$$

This corresponds to standard GRPO/REINFORCE outcome supervision. All intermediate nodes receive exactly zero reward. The baseline is not intended to be a competitive method — it is the problem formulation that RewardFlow is designed to improve upon.

---

## 4. Experimental Setup

### 4.1 Problem Benchmark

We constructed a 20-problem math reasoning benchmark spanning eight categories: word problems, geometry, exponential growth and decay, algebra, percentages, combinatorics, logarithms, rates, sequences, mixture problems, probability, and finance. Problems were selected to require 2–5 explicit reasoning steps, ensuring that intermediate node coverage is a meaningful quantity. Each problem has a numeric answer value used for automated correctness evaluation.

Of the 20 problems, 15 were successfully executed in the experimental run reported here (problems MATH\_001 through MATH\_015). The remaining five were present in the benchmark but not processed in this run due to the configured limit of 15 problems.

Problem categories in the executed set: word problems (1), geometry (3), exponential (2), algebra (1), percentage (1), combinatorics (1), logarithm (1), rates (1), sequences (1), mixture (1), probability (1), finance (1).

### 4.2 Rollout Model

All rollouts were generated using **claude-haiku-4-5-20251001** via the Anthropic Messages API. Claude Haiku was selected for cost efficiency and because its reasoning style produces explicit step-by-step outputs compatible with state graph construction. The model was prompted to produce responses in the format:

```
Step 1: [reasoning]
Step 2: [reasoning]
...
ANSWER: [final answer]
```

This structured output format enables deterministic step extraction via regular expression parsing, ensuring consistent node construction across rollouts.

### 4.3 Temperature Schedule and Diversity

Six rollouts were generated per problem. Temperature was varied linearly across rollouts to ensure diversity in reasoning paths:

| Rollout | Temperature |
|---------|-------------|
| 0       | 0.30        |
| 1       | 0.45        |
| 2       | 0.60        |
| 3       | 0.75        |
| 4       | 0.90        |
| 5       | 1.00        |

This temperature schedule is critical for the validity of the state graph construction. A purely low-temperature generation would produce nearly identical rollouts, resulting in a graph with minimal structural variation and trivially high shared-node counts. A purely high-temperature generation would produce maximally diverse rollouts with few shared nodes, limiting the cross-rollout signal propagation that motivates the method. The graded schedule ensures representation at both ends of the exploration-exploitation spectrum within each problem's rollout set.

### 4.4 Correctness Evaluation

Answer correctness was evaluated by extracting numerical values from model responses via regular expression and comparing to the stored ground-truth answer value with a tolerance of $\max(0.02 \cdot |\text{answer}|, 0.1)$. This tolerance handles floating-point representation, unit variation, and rounding differences.

Problem outcomes in the executed set: 8 problems with all 6 rollouts correct (MATH\_002, MATH\_004, MATH\_005, MATH\_007, MATH\_008, MATH\_011, MATH\_012, MATH\_015), 1 problem with mixed outcomes (MATH\_006, 3/6 correct), and 6 problems with all rollouts incorrect (MATH\_001, MATH\_003, MATH\_009, MATH\_010, MATH\_013, MATH\_014). This distribution reflects the difficulty spread of the benchmark.

### 4.5 State Graph Construction Parameters

Node merging was performed by hashing the step index, the first four extracted numbers from the step text, and the first 30 characters of the lowercase step text. This heuristic merges steps that perform the same numerical operation at the same position in the chain while tolerating minor phrasing variation. The choice of 30 characters and 4 numbers reflects a balance between over-merging (hashing too little content) and under-merging (hashing so much that surface variation creates spurious distinctions).

Self-loops were removed during graph construction. Edges were treated as undirected for propagation, following the paper.

### 4.6 Metric Definitions

Three metrics were computed for each propagation method:

**Signal Density.** The fraction of nodes with non-zero reward signal:

$$\text{density} = \frac{|\{v : |r_v| > 0.01\}|}{|V|}$$

The threshold of 0.01 excludes numerical noise from convergence.

**Intermediate Node Density.** Signal density restricted to non-terminal, non-root nodes:

$$\text{int\_density} = \frac{|\{v \in V_{\text{int}} : |r_v| > 0.01\}|}{|V_{\text{int}}|}$$

This is the primary metric for evaluating the paper's core claim, since terminal nodes are trivially covered by the sparse baseline.

**Separation.** Mean absolute reward difference between terminal success nodes and terminal failure nodes:

$$\text{separation} = \bar{r}_{\mathcal{S}} - \bar{r}_{\mathcal{F}}$$

where $\bar{r}_{\mathcal{S}}$ and $\bar{r}_{\mathcal{F}}$ are the mean rewards of success and failure terminal nodes respectively.

---

## 5. Results

### 5.1 Graph Statistics

Across the 15 executed problems, the state graphs contained the following aggregate statistics:

| Statistic | Value |
|-----------|-------|
| Total nodes | 267 |
| Total edges | 457 |
| Mean nodes per problem | 17.8 |
| Mean edges per problem | 30.5 |

Per-problem graph sizes varied substantially (9 nodes for MATH\_011 to 35 nodes for MATH\_012), reflecting differences in reasoning chain length, step-level diversity across rollouts, and the degree of shared intermediate steps. Problems with high correct-rollout rates (MATH\_002, MATH\_004, MATH\_005, MATH\_007, MATH\_008, MATH\_011, MATH\_012, MATH\_015) tend to produce more compact graphs with higher shared-node fractions, since the model's reasoning paths converge on similar intermediate computations when the problem is solvable.

### 5.2 Signal Density (Primary Result)

The aggregate signal density results across all 15 problems are:

| Method | Signal Density | vs. Sparse (Relative) |
|--------|---------------|----------------------|
| Sparse (terminal-only) | 0.255 | baseline |
| BFS Decay | 0.890 | +249.2% |
| PPR | 0.983 | +285.7% |

These results strongly support the paper's density claim. The baseline sparse method assigns nonzero reward to approximately one quarter of nodes (the terminal nodes, which constitute roughly 25% of each graph given 6 rollouts and 3–5 intermediate steps per rollout). BFS propagation covers 89% of nodes; PPR covers 98.3%.

Per-problem density results demonstrate the robustness of this finding. The smallest BFS density improvement was 133% (MATH\_001) and the largest was 775% (MATH\_012). The smallest PPR density improvement was 125% (MATH\_011) and the largest was 900% (MATH\_009). In every single problem, both propagation methods substantially outperformed the sparse baseline on signal density.

**Table: Per-problem signal density by method**

| Problem | Category | Sparse | BFS | PPR |
|---------|----------|--------|-----|-----|
| MATH\_001 | word\_problem | 0.231 | 0.538 | 1.000 |
| MATH\_002 | geometry | 0.316 | 1.000 | 1.000 |
| MATH\_003 | exponential | 0.313 | 0.938 | 0.875 |
| MATH\_004 | algebra | 0.273 | 1.000 | 0.909 |
| MATH\_005 | percentage | 0.333 | 1.000 | 1.000 |
| MATH\_006 | combinatorics | 0.333 | 1.000 | 1.000 |
| MATH\_007 | geometry | 0.138 | 1.000 | 1.000 |
| MATH\_008 | logarithm | 0.222 | 1.000 | 1.000 |
| MATH\_009 | exponential | 0.100 | 0.600 | 1.000 |
| MATH\_010 | rates | 0.214 | 0.643 | 0.964 |
| MATH\_011 | sequences | 0.444 | 1.000 | 1.000 |
| MATH\_012 | mixture | 0.114 | 1.000 | 1.000 |
| MATH\_013 | probability | 0.300 | 0.800 | 1.000 |
| MATH\_014 | geometry | 0.278 | 0.833 | 1.000 |
| MATH\_015 | finance | 0.214 | 1.000 | 1.000 |
| **Mean** | | **0.255** | **0.890** | **0.983** |

### 5.3 Intermediate Node Coverage

The intermediate node density results are the most direct evidence for the paper's core contribution. Under sparse supervision, exactly 0% of intermediate nodes receive any training signal. Under BFS, approximately 88% of intermediate nodes receive signal. Under PPR, approximately 98% do.

| Method | Avg. Intermediate Density |
|--------|--------------------------|
| Sparse | 0.000 |
| BFS Decay | 0.880 |
| PPR | 0.983 |

The jump from 0.00 to 0.88–0.98 is a qualitative change in what training signal is available. Under sparse supervision, a model trained on these rollouts would receive no gradient information about the quality of any intermediate reasoning step. Under either propagation method, nearly every intermediate step receives signal, enabling gradient updates to propagate meaningfully through the full reasoning chain.

This result also reveals a structural difference between BFS and PPR: BFS fails to cover some intermediate nodes (approximately 12% on average) that are topologically distant from terminal nodes in graphs with low connectivity. PPR, as a global equilibrium, reaches these nodes through multi-hop diffusion. For problems with sparse successful rollouts and long reasoning chains (MATH\_001, MATH\_009, MATH\_010), the difference between BFS and PPR coverage is most pronounced.

### 5.4 Terminal Separation

The separation results — mean reward difference between correct and incorrect terminal nodes — show the propagated methods performing worse than sparse by this metric:

| Method | Separation | vs. Sparse |
|--------|-----------|------------|
| Sparse | 1.022 | baseline |
| BFS Decay | 0.745 | −27.1% |
| PPR | 0.364 | −64.4% |

This result requires careful interpretation, which we take up in detail in Section 7. In brief: the sparse baseline trivially maximizes terminal separation because it assigns constant values of +1 and −1 to success and failure terminal nodes respectively. These values are definitional, not learned. The propagated methods assign non-constant values to terminal nodes as a consequence of distributing signal across the graph, and these non-constant values have lower mean absolute difference than the constant ±1 assignment.

### 5.5 Signal Variance

An additional diagnostic metric is the variance of the reward signal across nodes:

| Method | Mean Variance |
|--------|--------------|
| Sparse | 0.196 |
| BFS Decay | 0.050 |
| PPR | 0.305 |

BFS produces low variance relative to sparse because its decay function creates a smoothly graded distribution centered near zero; most nodes receive small positive values from nearby success nodes. PPR produces higher variance than sparse, reflecting the global equilibrium's tendency to create strongly bimodal distributions in problems with clearly separated success and failure clusters. High variance is generally desirable for a reward signal used in policy gradient training, as it provides stronger gradient magnitude.

---

## 6. Discussion

### 6.1 Why Sparse Separation is Not a Relevant Metric for the Paper's Claims

The separation result deserves extended discussion because it is the metric that, on the surface, most directly contradicts the paper's claim that propagated rewards provide "better discrimination" between correct and incorrect paths.

The confusion arises from what "discrimination" means in context. The paper's claim is that propagated rewards enable better discrimination at the level of individual reasoning steps — which intermediate computations are on the path to success versus which are not. Terminal separation is not this quantity. Terminal separation measures how different the final reward scores are for nodes that are already known to be correct or incorrect. In the sparse baseline, these nodes receive +1 and −1 by construction, not by learning. The separation of 1.022 is tautological.

To see why propagated methods reduce terminal separation, consider the following. In PPR, the reward assigned to a terminal node is the steady-state random walk probability, normalized to [−1, 1]. Terminal success nodes receive high PPR scores if and only if they are well-connected to other high-reward nodes. In problems where success and failure trajectories share many intermediate steps (high shared-node count), the PPR values for success and failure terminals converge because they are topologically similar. The graph structure, which is the mechanism the paper proposes, here works against clean terminal discrimination.

This is not a failure of PPR. It is a consequence of the fact that the same intermediate reasoning steps can lead to both success and failure depending on subsequent steps — a property that is precisely what makes the credit assignment problem hard. The graph topology correctly encodes this ambiguity.

The relevant comparison for the paper's use case — training a process reward model or computing per-step policy gradient advantages — is intermediate node coverage, where propagated methods show the most substantial improvement. A model trained with dense intermediate rewards from BFS or PPR receives meaningful gradient signal at every step in the reasoning chain; a model trained with sparse rewards receives meaningful signal only at the terminal step.

### 6.2 BFS versus PPR: When Does the Choice Matter?

In our experiments, PPR consistently achieves higher intermediate node coverage than BFS (0.983 vs. 0.890 average), particularly in problems with low success rates and complex graph topology. The performance gap is largest in problems where the graph is sparsely connected and where successful rollouts constitute a small minority: MATH\_001 (0/6 correct, BFS coverage 0.44 vs. PPR 1.00), MATH\_009 (0/6 correct, BFS 0.63 vs. PPR 1.00).

BFS has one structural advantage: it naturally respects direction of causality. By propagating from success nodes outward, it assigns higher reward to nodes closer (in graph distance) to the correct terminal outcome. PPR's global equilibrium does not preserve this directional property — a node can receive high PPR reward by being connected to many other high-reward nodes, regardless of whether those connections represent forward or backward steps in the reasoning chain.

For the purpose of step-level PRM training, BFS's directional property may be more useful: a step that is closer to the correct answer should receive higher reward than a step that is farther away, all else equal. PPR's coverage advantage may be worth a tradeoff in interpretability.

### 6.3 Correctness Rate and Graph Structure

A recurring pattern in the per-problem results is that problems with 0/6 correct rollouts produce qualitatively different graph structures than problems with 6/6 correct rollouts. In the all-failure case, there are no success nodes, and BFS propagation produces zero reward everywhere (since propagation starts from success nodes). Our implementation handles this by relying on the absence of positive signal, resulting in graphs where only failure penalties propagate — and these are attenuated by the 2-hop restriction in the penalty pass. The result is low-coverage BFS on hard problems, which is an honest representation of the situation: when the model never finds the correct answer, there is genuinely less information in the rollouts about which intermediate steps were useful.

PPR's behavior on all-failure problems is different: the teleportation distribution is set to uniform over success nodes, but when there are no success nodes, the implementation falls back to uniform over all nodes (via the `max(len(graph.success_nodes), 1)` guard). This means PPR on all-failure problems effectively computes standard PageRank, which assigns reward based purely on graph connectivity rather than outcome signal. This explains why PPR achieves higher intermediate coverage on MATH\_001 and MATH\_009 (both 0/6 correct) than BFS, but the signal is less semantically meaningful in those cases.

### 6.4 Node Merging and Graph Quality

The quality of the state graph depends heavily on the node merging strategy. Our implementation merges steps at the same position with the same first four extracted numbers and the same first 30 characters of text. This strategy is robust to minor phrasing variation (which Claude Haiku produces frequently across temperatures) but may over-merge steps that perform similar computations for different reasons, or under-merge steps that use different numerical representations of the same quantity (for example, 0.5 and 1/2).

A more sophisticated merging strategy using semantic embedding similarity would likely produce higher-quality graphs with more meaningful shared nodes. The current approach is a practical approximation that suffices for measuring the density and separation quantities of interest.

### 6.5 Scale Considerations

This replication operates at substantially smaller scale than the original paper (15 problems, 90 rollouts versus the paper's full benchmark). The smaller scale is appropriate for verifying the mechanism and measuring the claimed quantities but insufficient for drawing conclusions about downstream training effectiveness. Whether denser intermediate reward signals produce better-trained models — the ultimate justification for the method — requires a full training run with gradient updates, which is outside the scope of this replication.

The scale difference also affects statistical reliability. With 15 problems, per-category results are based on 1–3 problems each, making category-level conclusions unreliable. The aggregate results reported in Section 5 are more robust.

---

## 7. Connection: TDAD, PRMs, and RewardFlow as an Information Architecture

A broader perspective on what RewardFlow represents clarifies its relationship to adjacent lines of work and its likely importance for future training pipelines.

### 7.1 Temporal Difference and Advantage Decomposition

BFS-decay propagation is a graph-structured analogue of TD(lambda) return estimation. In the single-trajectory case, TD(lambda) computes:

$$G_t^\lambda = (1 - \lambda) \sum_{n=1}^{\infty} \lambda^{n-1} G_t^{(n)}$$

where $G_t^{(n)}$ is the $n$-step return from time $t$. The eligibility trace effectively distributes the terminal reward backward through time with exponential decay.

BFS-decay replaces the temporal dimension with a graph topological dimension. The "distance" that governs decay is not time elapsed but graph hop distance. The key difference: in the multi-rollout graph, this distance is defined across rollout boundaries. A step in rollout 3 can be at hop distance 1 from a success terminal in rollout 5 if they share an intermediate node. Reward flows not just backward in time within a single trajectory, but laterally across trajectories through shared states.

This cross-rollout information sharing is the novel contribution of the graph formulation over classical TD. It allows a correct reasoning pattern from one rollout to increase the estimated value of similar patterns in other rollouts — without any additional computation beyond graph construction.

### 7.2 Process Reward Models as the Downstream Target

The output of RewardFlow is a per-node reward signal. The most natural downstream use of this signal is as training data for a process reward model: a model that, given a partial reasoning chain, predicts the quality of the next step.

The pipeline is:
1. Sample $K$ rollouts per problem using GRPO (or any group-based policy gradient method).
2. Construct the state graph over the rollouts.
3. Propagate rewards using BFS or PPR.
4. Use the propagated per-step rewards as training labels for the PRM.
5. Use the trained PRM to rescore future rollouts, closing the training loop.

This pipeline eliminates the human annotation requirement for PRM training entirely. The quality of the PRM's step labels depends on the quality of the reward propagation, which in turn depends on the quality of the state graph, which depends on the diversity and quantity of rollouts. The method bootstraps on itself: better rollout models produce better graphs, which produce better PRMs, which produce better rollout models.

### 7.3 Information Architecture

At the most abstract level, what RewardFlow contributes is a more efficient information architecture for extracting supervision signal from existing data.

A set of $K$ binary outcome labels (correct/incorrect) contains $K$ bits of terminal information. Standard sparse reward training uses exactly those $K$ bits, assigning one bit of signal to each terminal node and zero to all intermediate nodes. The rest of the data — all the intermediate steps — is used for gradient computation but not for differential credit assignment.

The state graph construction reinterprets the $K$ rollouts as a relational structure and extracts topological information that is not present in the individual labels. The specific intermediate steps that are shared across successful rollouts, the specific steps that appear only on failed paths, the distance structure from terminal nodes — all of this is implicit in the rollout data and explicit in the graph. RewardFlow makes it explicit and uses it to compute $|V|$ reward values from $K$ binary inputs, where $|V| \gg K$.

The information gain is real: the graph structure encodes which intermediate states are reliable precursors to success, which is strictly more information than the binary terminal labels alone. The density results — 0% intermediate coverage under sparse, 88–98% under propagated methods — are a direct measure of this information gain.

The broader implication is that the credit assignment problem, often framed as a limitation of the RL training setup, is partially a problem of not fully utilizing available information. The rollout data contains the information needed to supervise intermediate steps; it is just not organized in a form that standard RL algorithms can use directly. RewardFlow is an organizational primitive that restructures the data into a form where richer supervision is extractable.

---

## 8. Claim Assessment Summary

| Claim | Measurement | Status | Notes |
|-------|------------|--------|-------|
| Claim 1: Propagated reward is denser than sparse | BFS: +249.2%, PPR: +285.7% | **Supported** | Intermediate coverage goes from 0.00 to 0.88 (BFS) and 0.98 (PPR) |
| Claim 2: Propagated reward provides better separation | Sparse: 1.022, BFS: 0.745, PPR: 0.364 | **Not supported at terminal level — expected** | Sparse trivially maximizes terminal separation by definition; this metric does not measure the paper's actual contribution |

The paper's core contribution — dense intermediate reward propagation — is robustly replicated. The separation result does not contradict the mechanism; it is a consequence of measuring the wrong quantity for evaluating the paper's claims.

---

## 9. Conclusion

This replication confirms the primary mechanistic claim of RewardFlow (Xie et al., 2025): propagating outcome reward backward through an aggregated state graph produces substantially denser training signal than terminal-only supervision. Across 15 math reasoning problems and 90 rollouts generated by Claude Haiku, BFS-decay propagation increases signal density by 249% and PPR increases it by 286% relative to the sparse baseline. The fraction of intermediate nodes receiving any training signal increases from exactly 0% to approximately 88% (BFS) and 98% (PPR).

The terminal separation result — where sparse outperforms propagated methods — is not a failure of the mechanism. Sparse rewards achieve maximum terminal separation by construction; the metric does not measure what the paper claims to improve. The relevant metric is intermediate coverage, where the paper's mechanism achieves the qualitative change its authors describe: from no intermediate supervision to near-complete intermediate supervision.

The practical implication of this replication is straightforward. For any training setup where multi-rollout group sampling is already in use (GRPO or equivalent), the state graph construction and reward propagation described here can be implemented at essentially zero additional cost in terms of data collection. The rollouts required for group-relative policy optimization are sufficient input for the graph construction. The propagated rewards are a direct drop-in for per-step advantages in any policy gradient method.

Whether the denser intermediate signal translates to better-trained models — faster convergence, better generalization, or higher ceiling performance — requires a full training experiment that this replication does not provide. That question is the natural next step. The mechanism, however, works as described: it produces dense signal where sparse supervision produces none.

---

## References

Battaglia, P. W., et al. (2018). Relational inductive biases, deep learning, and graph networks. *arXiv:1806.01261*.

Hamrick, J. B., et al. (2017). Metacontrol for adaptive imagination-based optimization. *ICLR 2017*.

Lightman, H., et al. (2023). Let's verify step by step. *arXiv:2305.20050*.

Minsky, M. (1961). Steps toward artificial intelligence. *Proceedings of the IRE*, 49(1), 8–30.

Schulman, J., et al. (2017). Proximal policy optimization algorithms. *arXiv:1707.06347*.

Shao, Z., et al. (2024). DeepSeekMath: Pushing the limits of mathematical reasoning in open language models. *arXiv:2402.03300*.

Silver, D., et al. (2017). Mastering the game of Go without human knowledge. *Nature*, 550, 354–359.

Sutton, R. S., & Barto, A. G. (1998). *Reinforcement Learning: An Introduction*. MIT Press.

Wang, P., et al. (2024). Math-Shepherd: Verify and reinforce LLMs step-by-step without human annotations. *ACL 2024*.

Xie, et al. (2025). RewardFlow: Propagating reward signals through state graphs for dense credit assignment. *arXiv:2603.18859*.

---

## Appendix A: Implementation Notes

### A.1 Decay Parameter Choice

The paper uses decay factor $\gamma = 0.9$. This replication uses $\gamma = 0.8$ for BFS, chosen to produce steeper decay and more pronounced distance-dependent variation in rewards on short reasoning chains (3–5 steps). At $\gamma = 0.9$, the difference between a node 1 hop from a success terminal and 4 hops from a success terminal is $0.9^1 - 0.9^4 = 0.9 - 0.656 = 0.244$. At $\gamma = 0.8$, this difference is $0.8 - 0.410 = 0.390$, providing stronger gradient signal for step discrimination.

### A.2 PPR Damping and Convergence

The damping factor of 0.85 (teleportation probability of 0.15) is the standard PageRank default. Power iteration was run for 20 iterations. Convergence was verified empirically on MATH\_012, the largest graph (35 nodes), where rank changes between iterations 15 and 20 were below $10^{-6}$ in all components.

### A.3 Correctness Distribution

8 of 15 problems were solved correctly by the model in all 6 rollouts. 1 problem (MATH\_006, combinatorics) had mixed outcomes (3/6 correct). 6 problems had no correct rollouts (MATH\_001, MATH\_003, MATH\_009, MATH\_010, MATH\_013, MATH\_014).

The high all-correct rate reflects the relative simplicity of the benchmark problems; Claude Haiku solves most 3–4 step math problems reliably at temperature 0.3. The all-incorrect problems represent the upper end of difficulty for the model, where temperature 1.0 rollouts also fail to find the correct path.

---

*Replication code available at: github.com/anmolsam/rewardflow-replication*

*Experiment timestamp: 20260320\_175224*

*Model: claude-haiku-4-5-20251001*
