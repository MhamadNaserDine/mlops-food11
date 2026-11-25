\# Lab 5 – CI/CD with GitHub Actions



\## What I did

1\. Switched torch to the \*\*CPU-only index\*\* (`pyproject.toml`): removed all CUDA packages. Image size went from \*\*9.58 GB to 2.05 GB\*\*.

2\. Made the code testable: pure functions in `data.py` (`CLASSES`, `category\_from\_filename`, `resize\_image`), and `serve.py` keeps `model` in a module variable that is only loaded if `SKIP\_MODEL` is not set (used FastAPI `lifespan`, the replacement for the deprecated `on\_event`). `/predict` returns 503 without a model.

3\. Tests: `tests/test\_data.py`, `tests/test\_api.py` (fake model) + `ruff`. Result: \*\*5 passed\*\*, ruff clean.

4\. Data contract `src/food11/validate\_data.py` (5 checks, PASS/FAIL, exit 1 on failure).

5\. CI `.github/workflows/ci.yml`: `test` → `smoke` → `push` (main only) to `ghcr.io/mhamadnaserdine/mlops-lab1:<commit SHA>`.

6\. Branch protection on `main`: PR required, `test` + `smoke` must pass, no bypass.

7\. Quality gate `src/food11/promote.py` (+ `evaluate\_version.py` to measure the imported champion's accuracy).



Real findings:

\- The tests caught a real bug (`BytesIO` import lost while editing `serve.py`).

\- The data contract found \*\*43 real duplicates between splits\*\* in Food-11 mini (data leakage).

\- The "best settings" run on mini \*\*overfitted\*\* (val\_accuracy 0.578 at epoch 3 → 0.450 at epoch 5).



\---



\## Answers



\*\*Q1. Why not load the real model in a unit test?\*\*

A unit test must check \*my code\*, fast and the same everywhere. With the real model it can fail even if the code is correct:

\- MLflow server not running / not reachable (CI cannot reach my laptop: "connection refused").

\- Registry state changed: `food11@champion` missing (e.g. after `docker compose down -v`) or alias moved to another version.

\- Artifacts not downloadable (Lab 3: `No such artifact`), slow downloads (\~45 MB + torch import).

\- A new champion changes the prediction → test fails without any bug.



\*\*Q2. What does `test\_predict\_returns\_valid\_json` prove, and not prove?\*\*

Proves: the API plumbing works. The upload is accepted, the image is decoded and preprocessed, the model output is converted to a class name, the JSON has `category` and `confidence` in \[0, 1], status 200. (It caught my real `BytesIO` bug.)

Does not prove: that the model is good or correct, that preprocessing matches training, that the real MLflow model returns this format, or that loading from MLflow works. It does not even check the confidence value: my API applies softmax to the fake scores, so it returns ≈0.20, not 0.90, and the test still passes.



\*\*Q3. Which check caught the duplicate? Why is it worse than a corrupted image?\*\*

Check \*\*4 (md5 hash across splits)\*\*. Duplicates went from 43 to \*\*44\*\*, with `training/Bread/0\_0.jpg` listed first (exit code 1). It caught the file even under a new name (`DUPLICATE\_TEST.jpg`), because it compares content.

A corrupted image makes training \*\*crash\*\*, so the problem is visible and gets fixed. A train/test duplicate fails \*\*silently\*\*: training works, but the model is tested on images it has already seen, so the test accuracy is \*\*too optimistic\*\* and we may promote a worse model. (My mini dataset really has 43 such duplicates.)



\*\*Q4. Where should `validate\_data.py` run, since CI cannot see the data?\*\*

1\. \*\*In the data pipeline\*\*, right after `data.py` creates the processed dataset and \*\*before `dvc add` / `dvc push`\*\*, so a bad dataset version is never published (as a DVC stage or a pre-commit hook).

2\. \*\*At the start of training\*\* (`train.py` / training job), on the machine that has the data, so no model is trained on data that breaks the contract.

(Also possible: a CI job that runs `dvc pull` from a remote reachable with secrets, e.g. S3 or DagsHub, then validates.)



\*\*Q5. `smoke` has `needs: test`. What if a test fails? Without `needs`?\*\*

If `test` fails, `smoke` is \*\*skipped\*\* (seen in my PR: `ci / smoke … Skipped`), and `push` too.

Without `needs`, both jobs run \*\*in parallel\*\*: smoke would build and test the image even when unit tests fail, which wastes time. And the pipeline order (test → build → publish) would be lost, so a later job could ship code that failed its tests.



\*\*Q6. Run time, first vs second run?\*\*

| Run | test | uv sync | smoke | Total |

|---|---|---|---|---|

| #1 (no cache) | 23 s | 4 s | 1 m 35 s | 2 m 4 s |

| #2 (empty commit) | 25 s | 5 s | — | 2 m 15 s |

Run #2 logged `Cache hit occurred on key setup-uv-…`, but `uv sync` was not faster (4 s → 5 s): uv is very fast, and with \*\*CPU-only torch\*\* there are no big CUDA wheels left to download. The cache matters more with large dependencies or slow networks. Most of the time is the \*\*smoke `docker build`, which has no cache\*\* (every runner starts empty), so the total did not drop. (The `push` job uses `cache-from/to: type=gha` to cache Docker layers.)



\*\*Q7. Why tag with the commit SHA, not `latest`?\*\*

A SHA tag is \*\*immutable and traceable\*\*: `ghcr.io/mhamadnaserdine/mlops-lab1:0ce100b…` always means exactly the code of commit `0ce100b`, digest `sha256:36e4e605…`. `latest` is a moving label.

Example: server A pulls `latest` on Monday. On Tuesday a new commit pushes a new `latest`, and server B (or a restarted Kubernetes pod) pulls it. Both say "latest" but run \*\*different code\*\*, and when a bug appears nobody knows which version is where. In Lab 3, my local `food11-api:latest` already pointed to two different images after one rebuild.



\*\*Q8. Why declare `permissions:` explicitly?\*\*

\*\*Least privilege\*\*: the `push` job only gets `contents: read` + `packages: write`, exactly what it needs to publish the image.

If a step were compromised (a malicious third-party action or dependency) and the token could write everything, it could push code to branches, change workflows to steal secrets, create or modify releases and tags, delete branches, or publish a malicious image under my name. Limiting permissions limits the damage.



\*\*Q9. What does the PR page show? Could you still merge?\*\*

"Some checks were not successful — 2 failing, 4 skipped": `ci / test` failing (push + pull\_request, both \*\*Required\*\*), `smoke` and `push` skipped. The \*\*Merge\*\* button was \*\*disabled\*\*, and because "Do not allow bypassing" is on, not even the admin (me) could merge. After reverting the bug, all required checks were green and I merged (PR #1).

For a team: tests are no longer "everybody runs them on their laptop, hopefully". The \*\*server enforces\*\* them on every change, in a clean environment, for everyone. Nobody can forget, skip, or say "it works on my machine", and `main` always stays green.



\*\*Q10. What did the gate decide?\*\*

Champion v2: test\_accuracy \*\*0.7418\*\* (n = 1096) → se = √(0.7418·0.2582/1096) = \*\*0.0132\*\*, so \*\*2·se = 0.0264 ≈ 2.6 accuracy points\*\*. Threshold = \*\*0.7682\*\*.

| Candidate | Settings | test\_accuracy | Decision |

|---|---|---|---|

| v3 | lr 1e-5, 5 epochs | 0.5173 | \*\*REJECTED\*\* → `challenger` |

| v4 | lr 1e-3, 5 epochs (overfitted) | 0.4699 | \*\*REJECTED\*\* → `challenger` |

Final: `champion` → v2, `challenger` → v4. Decisions and numbers are stored as tags on each version.



\*\*Q11. How to run `promote.py` automatically after training?\*\*

1\. \*\*Host MLflow where CI can reach it\*\*, e.g. \*\*DagsHub's hosted MLflow\*\* (one per repo) or a cloud server, with credentials in GitHub \*\*secrets\*\*. Then a workflow (manual `workflow\_dispatch`, or triggered at the end of training) runs `promote.py`.

2\. \*\*A self-hosted GitHub Actions runner\*\* on my laptop or lab server, which can reach the local MLflow, runs the promotion job.

3\. (Simplest) Call the gate \*\*at the end of the training pipeline\*\* itself (`train.py` → register → `promote.py`), on the machine that trains.



\*\*Q12. The pipeline\*\*

```

&#x20;CODE PATH                                                          MODEL PATH

&#x20;─────────                                                          ──────────

&#x20;\[HUMAN] edit code, git push to a branch                            \[HUMAN] run train.py (laptop, needs data)

&#x20;       │                                                                   │

&#x20;       ▼                                                                   ▼

&#x20;\[AUTO] CI on branch/PR: test (ruff + pytest)                       \[AUTO]  MLflow logs run + test\_accuracy

&#x20;       │ needs                                                             │

&#x20;       ▼                                                                   ▼

&#x20;\[AUTO] smoke: docker build + /health with SKIP\_MODEL               \[HUMAN] register the run as a new version

&#x20;       │                                                                   │

&#x20;       ▼                                                                   ▼

&#x20;\[AUTO] branch protection blocks merge until test+smoke green       \[HUMAN runs, AUTO decides] promote.py gate:

&#x20;       │                                                                   acc\_cand >= acc\_champ + 2·se ?

&#x20;       ▼                                                                    ├─ yes → alias champion

&#x20;\[HUMAN] click "Merge pull request"                                          └─ no  → alias challenger

&#x20;       │                                                           ★ MODEL VERSION decided here (alias champion)

&#x20;       ▼

&#x20;\[AUTO] CI on main: test → smoke → push

&#x20;       ghcr.io/mhamadnaserdine/mlops-lab1:<commit SHA>

&#x20;★ CODE VERSION decided here (commit SHA)

&#x20;       │

&#x20;       └──────────────────────────┬─────────────────────────────────────────┘

&#x20;                                  ▼

&#x20;\[HUMAN] docker pull image:<SHA> + docker compose up / restart inference

&#x20;                                  ▼

&#x20;\[AUTO] at startup the container loads models:/food11@champion from MLflow → serves predictions

```

