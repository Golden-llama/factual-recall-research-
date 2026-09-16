## Project Overview

This project investigates how different embedding designs affect factual memorization capacity in small transformer language models. The experiments use a synthetic factual recall task where each entity is associated with multiple relations and attribute values. The model is trained to predict the correct attribute token given a structured factual prompt.

The central question is whether positional information helps or hurts memorization when the data follows a highly regular fixed-template format. To test this, I compare three embedding variants while keeping the transformer architecture and parameter budget approximately fixed:

| Variant | Description |
|---|---|
| Summed embeddings | Standard token and positional embeddings are added together. |
| Disentangled embeddings | Token and positional embeddings are concatenated into separate subspaces. |
| No positional embeddings | Positional embeddings are removed entirely. |

Across capacity experiments, I measure the largest number of entities each model can memorize under the same training setup. Memorization is evaluated using loss on the answer token and extraction accuracy on factual queries.

One of the findings is that removing positional embeddings improves factual memorization capacity in this fixed-template setting. In the current experiments, the summed and disentangled models memorize approximately 101,000 entities, while the no-positional model memorizes approximately 103,000 entities. This suggests that when position is mostly redundant, positional embeddings may consume representational capacity or introduce interference rather than helping factual storage.

In a related experiment, I evaluate whether models generalize to held-out entity compositions, such as deciding whether two entities share an attribute. I probe whether subject embeddings contain shared attribute directions by measuring the norm of the average embedding for entities with the same attribute value, normalized by the average subject embedding norm.

This geometry probe tests whether higher held-out composition accuracy is associated with stronger linearly organized attribute structure in the entity embedding space.

We also hypothesized that high held-out composition accuracy appears when subject embeddings encode attribute vectors more strongly. Summed embeddings would entangle positional information with token identity, leading to concatenated and no-positional embeddings achieving better generalization.

This hypothesis was proven accurate, as no-positional and concatenated embeddings achieved better generalization than summed embedding structures. In addition, after calculating the group norm ratios, we found that high held-out accuracy was associated with stronger shared directions in subject embeddings, further suggesting a linearly encoded attribute structure. 

Link to slides, graphs, and a more comprehensive view of the project: https://docs.google.com/presentation/d/1N47PYfMUzUhKTO_WuDJT-5xe5FZdEsDSneC-VwleOnA/edit?slide=id.g3f5a6a566cd_0_72#slide=id.g3f5a6a566cd_0_72 
