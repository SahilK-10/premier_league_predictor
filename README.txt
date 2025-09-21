Premier League Match Outcome Predictor Using Machine Learning

This project implements a machine learning pipeline to predict Premier League football match scores based on historical match data. Multiple seasons of match results from 2022/23 to 2025/26 (partial) are used to train gradient boosting and random forest regression models with advanced feature engineering.
Key ML Components

Features: Multi-window rolling averages of team form (short- and long-term), goal-scoring, conceding averages, goal difference, and team strength ratings derived from historical data.

    Model: Ensemble of RandomForest and XGBoost.

    Preprocessing: One-hot encoding of teams, feature scaling, handling missing values.

    Training & Evaluation: Models trained on all games prior to the target gameweek; held-out matches from the latest available season are used for testing.

    Predictions: Output continuous expected goals, rounded to nearest integer for final score predictions.



Important Notes on Accuracy

Football is an inherently unpredictable sport influenced by many external factors — player fitness, tactical changes, weather, refereeing decisions, and random events — which are not captured in typical historical match data. As such, the accuracy of machine learning predictions is inherently limited, with typical models achieving around 60-70% accuracy for match outcomes in research literature.


The focus of this project is to identify statistical trends and improve predictions incrementally as more data becomes available and models are refined.
Ongoing Work & Future Plans
    Continuous updating and retraining as new match results are added to the dataset.

    Improvements to feature engineering, including player-level data incorporation.

    Posting regular gameweek predictions and accuracy analysis.

    Using APIs instead of csv dataset files.