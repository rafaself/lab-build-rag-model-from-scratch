# Retrieval regression questions

This hand-authored set contains 12 answerable questions and two private-information questions that the astronomy corpus cannot answer. `expected_answer` is a short reference answer; `gold_quote` is a verbatim source excerpt. Article passages were checked against the local corpus before publication.

`rag-lab evaluate` verifies the gold quotations and indexed content hashes. Source-document Recall@K and passage-level gold-quote hit rate measure different things. The weight sweep is exploratory calibration on these same cases, not an independent test set. Do not interpret a 100% source-document score as proof of answer quality.

## Source attribution

Gold quotations are excerpts from the English Wikipedia articles below. The article links and their revision histories identify the source text and contributors. The evaluation report preserves local content hashes; re-downloading articles can change the excerpts and require fixture review.

| Local source file | Wikipedia article |
| --- | --- |
| `153797_Orion_constellation.txt` | [Orion (constellation)](https://en.wikipedia.org/wiki/Orion_(constellation)) |
| `15736_Johannes_Kepler.txt` | [Johannes Kepler](https://en.wikipedia.org/wiki/Johannes_Kepler) |
| `19694_Mercury_planet.txt` | [Mercury (planet)](https://en.wikipedia.org/wiki/Mercury_(planet)) |
| `2589714_Milky_Way.txt` | [Milky Way](https://en.wikipedia.org/wiki/Milky_Way) |
| `26751_Sun.txt` | [Sun](https://en.wikipedia.org/wiki/Sun) |
| `38579_Gravity.txt` | [Gravity](https://en.wikipedia.org/wiki/Gravity) |
| `38930_Jupiter.txt` | [Jupiter](https://en.wikipedia.org/wiki/Jupiter) |
| `56568_Pleiades.txt` | [Pleiades](https://en.wikipedia.org/wiki/Pleiades) |
| `633325_Neutrino_astronomy.txt` | [Neutrino astronomy](https://en.wikipedia.org/wiki/Neutrino_astronomy) |
| `7070301_Telescope.txt` | [Telescope](https://en.wikipedia.org/wiki/Telescope) |
