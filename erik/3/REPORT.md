# Assignment 3 Delivery

Please follow the provided structure carefully and complete all of the sections described below. The parts that you need to complete are marked with *TODO*.

## Personal information

**Name**: Erik J. Sandal

List the GitHub usernames of your team here. If you are working alone, then write your GitHub username as the team leader and leave the list of additional team members empty.

**Team leader**: slippers1

**Additional team members**: Chrystallic, vundere

If you are working as a part of a team, only the a single person, the *team leader* needs to submit the report. If that is not you, then do not edit anything below this point.

----

## Baseline retrieval

Complete the table with baseline retrieval results.

  - For this part, you need to report numbers using the Elasticsearch instance that you access via the API.
  - For the anchor text field, limit results to documents that are present in the "Category B" subset of the collection.

| **Field** | **Output file** | **NDCG@10** | **NDCG@20** |
| -- | -- | -- | -- |
| Title | data/baseline_title.txt | 0.128 | 0.114 |
| Content | data/baseline_content.txt | 0.138 | 0.128 |
| Anchors | data/baseline_anchors.txt | 0.057 | 0.042 |


List the names of Jupyter notebooks or other code files that were used for producing the results in the above table:
  - Baseline.ipynb


## Learning-to-Rank

Complete the table with learning-to-rank results.

  - These numbers should be obtained using 5-fold cross-validation on the provided queries and relevance judgments.
  - We distinguish between 3 feature groups:
      - [QD] Query-document features
      - [Q] Query features
      - [D] Document features
  - Report on four combinations of feature groups: QD, QD+Q, QD+D, QD+Q+D
  - Use the same training/test folds to make sure the numbers are comparable!

| **Features** | **Output file** | **NDCG@10** | **NDCG@20** |
| -- | -- | -- | -- |
| Only QD features | data/ltr_qd.txt | 0.141 | 0.135 |
| QD + Q features | *TODO* | *TODO* | *TODO* |
| QD + D features | *TODO* | *TODO* | *TODO* |
| ALL features (QD + Q + D) | *TODO* | *TODO* | *TODO* |


List the features used with a brief explanation:
  - Query-document features [QD] (min. 6)
    1. title & BM25
    2. title & LM
    3. content & BM25
    4. content & LM
    5. anchors & BM25
    6. anchors & LM
  - Query features [Q] (min. 2)
    1. *TODO*
    2. *TODO*
  - Document features [D] (min. 2)
    1. *TODO*
    2. *TODO*

List the names of Jupyter notebooks or other code files that were used for producing the results in the above table:
  - *TODO*
  - *TODO*
