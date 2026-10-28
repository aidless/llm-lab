# Human review packet

Open `review.html` in a browser. It is a local static page: no data is uploaded.

The packet contains **227** records. For the project acceptance gate:

1. Review all 96 task-label records.
2. Review all 131 required unsafe-attempt trajectories.
3. Enter a reviewer identifier and export `reviews-export.json`.
4. Run:

```bash
../.venv/bin/python -m agent_safety_eval ingest-review   --export review/reviews-export.json   --output review/reviews-human.jsonl
```

Only validated records with a reviewer identifier, UTC timestamp and explicit
decision are counted as human review. The assistant-authored draft labels never
count as human labels.
