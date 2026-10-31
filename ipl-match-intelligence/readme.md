# IPL Match Intelligence & Pattern Mining

Discovering hidden patterns in ten seasons (2008–2017) of Indian Premier League match data using **K-Means clustering**, **Apriori association rules** and a **Decision Tree classifier**, all in Python.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR-USERNAME/ipl-match-intelligence/blob/main/IPL_Mining_project1.ipynb)


## Problem statement

IPL match records hold useful information, but reading hundreds of matches by hand makes patterns hard to find. This project builds a repeatable data mining workflow that finds meaningful relationships in IPL match performance and results, such as whether the toss matters, whether big totals are safe and whether venues change how teams play.

## Dataset

Kaggle IPL dataset, two files:

| File | Rows | Columns | Content |
|---|---|---|---|
| `matches.csv` | 636 | 18 | One row per match: season, teams, venue, toss, winner, win margin |
| `deliveries.csv` | 150,460 | 21 | One row per ball: runs, wickets, batsman, bowler |

Seasons covered: 2008 to 2017. The ball-by-ball file is summarised into the runs and wickets of each innings and joined to the match table.

## Methods

| Task | Algorithm | Library | Question |
|---|---|---|---|
| Clustering | K-Means (k = 3) | scikit-learn | What types of matches are there? |
| Association rules | Apriori | mlxtend | Which conditions go together? |
| Classification | Decision Tree (depth 4) | scikit-learn | Can the winner (batting first or chasing) be predicted? |

## Workflow

1. Load `matches.csv` and `deliveries.csv`
2. Clean the data (drop umpire columns, merge the duplicate team name, fill missing values)
3. Explore the data (EDA charts)
4. Summarise deliveries into innings runs and wickets and join them to the matches
5. K-Means clustering, Apriori rules, Decision Tree
6. Visualise and interpret the results

## Key results

**K-Means.** 634 matches fall into three types:

| Cluster | Matches | Meaning |
|---|---|---|
| 0 | 181 | Low-scoring matches won by the chasing team |
| 1 | 261 | Matches won by the team batting first |
| 2 | 192 | High-scoring matches won by the chasing team |

**Apriori.**
- High first-innings score → batting-first team wins (confidence 0.67, lift 1.46)
- Low first-innings score → chasing team wins (confidence 0.78, lift 1.43)
- Chepauk → toss winners bat first (0.72); Chinnaswamy → toss winners field first (0.86)

**Decision Tree.**

| Measure | Value |
|---|---|
| Baseline (always guess the common result) | 54.2% |
| Test accuracy (126 matches) | 65.1% |
| 5-fold cross-validation | 62.8% |

The first-innings score is the most important input (importance 0.69). The toss alone gives almost no advantage: the toss winner won only 51.3% of matches.

## How to run

**Google Colab (easiest)**

1. Click the **Open In Colab** badge above (or upload `IPL_Mining_project1.ipynb` to Colab).
2. Run the first cell and upload `archive_1.zip` when asked.
3. Choose **Runtime → Restart session and run all**.

**Locally**

```bash
pip install pandas numpy matplotlib seaborn scikit-learn mlxtend jupyter
jupyter notebook IPL_Mining_project1.ipynb
```

Unzip `archive_1.zip` so that `matches.csv` and `deliveries.csv` are in the same folder as the notebook, and change the upload cell to read them directly.

Expected check values: `matches.shape = (636, 18)` and `deliveries.shape = (150460, 21)`.

## Repository structure

```
ipl-match-intelligence/
├── IPL_Mining_project1.ipynb   # notebook with all code and outputs
├── archive_1.zip               # matches.csv and deliveries.csv
├── report/                     # project report
└── README.md
```

## Technology

Python · Pandas · NumPy · Matplotlib · Seaborn · scikit-learn · mlxtend · Google Colab

## Limitations and future scope

- Only 636 matches from 2008 to 2017; the test set is small (126 matches).
- No data on pitch, weather, line-ups or player form.
- Patterns describe the past and do not prove cause and effect.
- Future work: more seasons, player-level analysis from the ball-by-ball data, more advanced models and an interactive dashboard.
