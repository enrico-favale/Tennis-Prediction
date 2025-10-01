# 🎾 Tennis Match Outcome Prediction

## Project Overview

This project is focused on building a machine learning framework to **predict the outcomes of professional tennis matches**. The system leverages historical match data, betting odds, and contextual features to identify the most likely winner of a given match.

The primary goals are:

* To design a modular and extensible pipeline for data preprocessing, feature engineering, and model training.
* To achieve **robust predictive performance** using state-of-the-art gradient boosting algorithms such as XGBoost, LightGBM, CatBoost and a Neural network.
* To allow continuous updates as new matches are played, ensuring that the model adapts to current player form and emerging trends.

This project was developed by [**Enrico Favale**](https://github.com/enrico-favale) and [**Piergaetano Pio Palella**](https://github.com/PalellaPiergaetano).

## Installation

Before setting up the environment, clone the repository:

```bash
git clone https://github.com/enrico-favale/Tennis-Prediction
cd Tennis-Prediction
```

This project provides two alternative ways to set up the Python environment for development and execution, depending on whether you want to use a pre-defined Conda environment or create a minimal environment using a `requirements.txt` file.

### Option 1: Use the provided Conda environment

If you want a fully configured environment with all dependencies already specified, follow these steps:

1. **Create the environment from the `environment.yml` file:**

```bash
conda env create -f environment.yml
conda activate tennis_prediction
```

2. **Update the environment if needed:**

```bash
conda env update -f environment.yml --prune
```

This ensures your environment is up-to-date with any new dependencies added to `environment.yml`.

This pipeline is faster for new users and guarantees that all versions are compatible.

### Option 2: Create an empty environment and install via `requirements.txt`

If you prefer to start from scratch or want more control over installed packages:

1. **Create an empty Conda environment:**

```bash
conda create -n tennis_prediction python=3.11
conda activate tennis_prediction
```

2. **Install dependencies using `requirements.txt`:**

```bash
pip install -r requirements.txt
```

This pipeline provides flexibility and allows you to adjust individual package versions, but it may require resolving compatibility issues manually.

Both options result in an environment ready for data processing, feature engineering, and model training.

## Usage

## Dataset

The project uses historical tennis match data collected from official ATP and WTA sources. The dataset includes:

* Match results and scores.
* Player statistics and rankings at the time of the match.
* Betting odds from multiple bookmakers.
* Surface type, tournament level, and match format (best-of-3, best-of-5).

Data is stored in CSV format, with each row representing a single match. The dataset is preprocessed to handle missing values, normalize numerical features, and encode categorical variables. Additionally, context-based features are generated, such as:

* Player form in recent matches and recovery times.
* Career performances.
* Tournament and Surface performances.
* Head-to-head statistics.
* Close match indicators.

We would like to acknowledge and thank the data provider: [Dissfya on Kaggle](https://www.kaggle.com/datasets/dissfya/atp-tennis-2000-2023daily-pull) for making this comprehensive ATP dataset available, covering matches from 2000 to 2025.

## Features

The dataset used in this project contains a rich set of features capturing match, player, and context information. Key features include:

* **Tournament Info:** `Tournament`, `Date`, `Season`, `Series`, `Court`, `Surface`, `Round`, `Best of`
* **Player Info:** `Player_1`, `Player_2`, `Winner`
* **Ranking and Points:** `Rank_1`, `Rank_2`, `Pts_1`, `Pts_2`, `Rank_diff`, `Pts_diff`
* **Betting Odds:** `Odd_1`, `Odd_2`, `Odds_diff`
* **Head-to-Head Stats:** `H2H_wins_1`, `H2H_wins_2`, `H2H_diff`
* **Recent Performance:** `Win_pct_1_lastN`, `Win_pct_2_lastN`, `Win_pct_diff_lastN`, `Matches_played_1`, `Matches_played_2`
* **Career Stats:** `Career_wins_1`, `Career_losses_1`, `Career_matches_1`, `Career_win_pct_1`, `Career_wins_2`, `Career_losses_2`, `Career_matches_2`, `Career_win_pct_2`, `Career_win_pct_diff`, `Career_matches_diff`
* **Surface Performance:** `Surface_wins_1`, `Surface_losses_1`, `Surface_matches_1`, `Surface_win_pct_1`, `Surface_wins_2`, `Surface_losses_2`, `Surface_matches_2`, `Surface_win_pct_2`, `Surface_win_pct_diff`, `Surface_matches_diff`
* **Tournament Performance:** `Tournament_wins_1`, `Tournament_losses_1`, `Tournament_matches_1`, `Tournament_win_pct_1`, `Tournament_wins_2`, `Tournament_losses_2`, `Tournament_matches_2`, `Tournament_win_pct_2`, `Tournament_win_pct_diff`, `Tournament_matches_diff`
* **Match Timing:** `Days_since_last_match_1`, `Days_since_last_match_2`, `Days_diff`

These features allow the model to capture player form, historical performance, matchup dynamics, and betting market information, providing a comprehensive context for predicting match outcomes.

## Model Architecture

## Evaluation

## Contributing

We welcome contributions from the community! To contribute to this project, please follow these guidelines:

1. **Fork the repository** and create a new branch for your feature or bug fix.
2. **Write clear and concise code** following the existing style and conventions.
3. **Include tests** when applicable and ensure all tests pass.
4. **Document your changes** in the README or appropriate documentation files.
5. **Submit a pull request** with a detailed description of your changes for review.

By contributing, you agree that your contributions will be licensed under the same terms as this project.

## License

This project is licensed under the MIT License.

```
MIT License

Copyright (c) 2025 Enrico Favale and Piergaetano Pio Palella

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

