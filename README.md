# Used-car sale price regression

This project fits a linear model of used-car sale price, checks it on held-out sales, and applies the fitted model to auction lots. `used_car_sales.csv` and `auction_lots.csv` are inputs and are not modified.

## Setup

From this folder, use the existing environment:

```bash
source .venv/bin/activate
python analysis.py
```

To recreate the environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python analysis.py
```

The script uses pandas, statsmodels, and matplotlib. On macOS or Linux, activate the environment with `source .venv/bin/activate`. On Windows, use `.venv\Scripts\activate`.

## What the script does

- Checks that the sales file has 240 cars and the auction file has 40 lots, and reports missing values.
- Holds out 30% of the sales rows with random seed 42 (168 training cars, 72 test cars).
- Fits sale price on mileage, age, horsepower, accident, and number of owners with statsmodels OLS. Accident is categorical, with `no` as the reference level.
- Keeps every feature in the selected model, called v1, because each coefficient is significant and dropping any feature makes the training fit worse.
- Compares v1 on the test cars with a baseline that always predicts the training-set average sale price.
- Saves `test_predictions.png`, a predicted-versus-actual chart for the test cars.
- Buys an auction lot when a rule's predicted sale price is more than the auction price plus $1,500. `Later_Sale_Price` is used only after that decision, to score profit on purchases.
- Saves every lot's prediction, decision, and realized profit to `auction_decisions.csv`.

Running `python analysis.py` prints the model summary, coefficients, p-values, adjusted R-squared, test R-squared, test RMSE, and auction profit totals. Those printed numbers are the results to cite.
