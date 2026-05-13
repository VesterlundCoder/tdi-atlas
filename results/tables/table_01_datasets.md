## Table 1. Dataset Sources and Preprocessing

| Source | n datasets | Domains | N range | D range | Preprocessing |
|---|---|---|---|---|---|
| scikit-learn built-ins | 4 | biology, medicine, digits | 150–1,797 | 4–64 | None |
| OpenML static | 46 | physics, finance, ecology, … | 148–10,000 | 3–617 | Stratified cap 1,000/class |
| OpenML dynamic catalog | ~300 | 25 domains | 200–10,000 | 3–10,935 | Stratified cap 1,000/class |
| Synthetic (sklearn) | 16 | synthetic | 500–2,000 | 2–50 | Label noise 0% |
| CMF hunt shards | 1 | mathematics | 6,868 | 20 | Flattened D-matrix + convergence score |
