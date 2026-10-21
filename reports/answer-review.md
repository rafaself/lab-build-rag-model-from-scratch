# Answer benchmark review notes

This is a source-based Codex inspection of `answer-quality.json`, not an independent human review. The report retains the original model verdicts and the exact references, claims, and source excerpts used in each judgment.

| Case(s) | Observed result |
| --- | --- |
| `sun-energy`, `sun-history`, `kepler-assistant`, `first-telescopes`, `neutrino-force`, `orion-stars`, `pleiades-location`, `jupiter-storm` | Answer matches the reference fact and its cited excerpt. |
| `milky-way-type`, `mercury-size` | Correct reference fact, repeated across two source citations; duplication remains a presentation limitation. |
| `gravity-relativity` | Correct spacetime-curvature answer, plus an unnecessary Mercury-orbit claim. The full second excerpt describes the unexplained perihelion shift and Einstein's successful model. The judge's rejection reason concerns relevance rather than entailment. Its `faithful: false` is a grading inconsistency, while the runtime grounding check also failed to remove this irrelevant extra claim. |
| `sun-neutrinos` | Abstained; counted as an accuracy failure. The gold answer is 2.3 seconds. Earlier retrieval measurements show that good document recall does not guarantee the required passage appears within the three supplied chunks. |
| `private-password`, `private-package` | Correct abstention. Neither expected private fact exists in this public astronomy corpus. |

The measured 11/12 answerable accuracy and 13/14 claim faithfulness are the model evaluator's outputs on this small development set. Do not interpret them as held-out accuracy, independent verification, or a proof of security. The model evaluator is exposed to untrusted answer/source text in a separate user payload and can itself be manipulated. Deterministic citation checks establish only source-ID/quote validity, not truth or relevance.
