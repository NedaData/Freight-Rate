Original report and loom link is HERE.

How to run:
python Prediction.py \
        --train-csv train-test.csv \
        --validation-csv validation.csv \
        --output-csv validation_predictions.csv \
        --december-csv december-chart-inputs.csv \
        --december-output december-chart-inputs-completed.csv \
        [--test-size 0.2] [--random-state 42] [--metric MAE]

python score.py \
        --predictions data/validation_predictions.csv \
        --december-predictions /data/december-chart-inputs-completed.csv
