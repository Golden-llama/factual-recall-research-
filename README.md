## Project Overview

This project investigates how different embedding designs affect factual memorization capacity in small transformer language models. The experiments use a synthetic factual recall task where each entity is associated with multiple relations and attribute values. The model is trained to predict the correct attribute token given a structured factual prompt.

The central question is whether positional information helps or hurts memorization when the data follows a highly regular fixed-template format. To test this, I compare three embedding variants while keeping the transformer architecture and parameter budget approximately fixed:

| Variant | Description |
|---|---|
| Summed embeddings | Standard token and positional embeddings are added together. |
| Disentangled embeddings | Token and positional embeddings are concatenated into separate subspaces. |
| No positional embeddings | Positional embeddings are removed entirely. |

Across capacity experiments, I measure the largest number of entities each model can memorize under the same training setup. Memorization is evaluated using loss on the answer token and extraction accuracy on factual queries.

The main finding is that removing positional embeddings improves factual memorization capacity in this fixed-template setting. In the current experiments, the summed and disentangled models memorize approximately 101,000 entities, while the no-positional model memorizes approximately 103,000 entities. This suggests that when position is mostly redundant, positional embeddings may consume representational capacity or introduce interference rather than helping factual storage.

More broadly, the project explores how architectural choices in the embedding layer influence the efficiency and limits of factual recall in transformer models.
