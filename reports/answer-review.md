# Answer benchmark source review

This is a Codex source-comparison audit of the current `answer-quality.json`, not an independent human review. `answer-source-audit.json` binds the audit to the report's SHA-256 and records every reviewed claim. Model verdicts remain unchanged in the original report.

| Cases | Source-based assessment |
| --- | --- |
| `sun-energy`, `sun-neutrinos`, `sun-history`, `gravity-relativity`, `kepler-assistant`, `milky-way-type`, `neutrino-force`, `orion-stars`, `pleiades-location`, `jupiter-storm`, `mercury-size` | Essential answer facts match the reference answers and cited context. Mercury repeats its answer across two citations, a remaining presentation limitation. |
| `first-telescopes` | The Netherlands answer is supported. The extra assertion that a telescope was **patented** strengthens the source's statement that a patent was **submitted**. The model grounding and answer judges both accepted this overstatement. The source-based audit counts this answer as incorrect and that claim as unfaithful under the original strict rubric. |
| `private-password`, `private-package` | Correct abstentions; the source corpus contains neither private fact. |

The model judge scored 14/14 cases correct and 14/14 claims faithful. The stricter source audit scored 13/14 cases correct (11/12 answerable) and 13/14 claims supported. `answer-judge-calibration.json` preserves the telescope claim, original verdict, and source excerpt. An explicit extra prompt rule distinguishing applications from approvals also failed an isolated live check, so it was discarded rather than presented as a fix.

The initial answer baseline is preserved in `answer-baseline.json`; it abstained on the solar-neutrino question and its judge conflated relevance with faithfulness for an extra Mercury-orbit claim. The revised concise prompt answered the neutrino question and reduced surplus facts, but the patent example demonstrates remaining semantic-validation limits. The initial baseline's 11/12 answerable correctness and 13/14 model-judged claim faithfulness are separate from the final measurements.

Valid source IDs and verbatim quotes establish citation validity. They do not establish entailment, real-world truth, or relevance. A model can strengthen a source's qualifiers, and its own evaluator may accept the same error. Use the per-claim audit and explicit denominators; these development cases do not establish generalization or injection immunity.
