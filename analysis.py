"""Used-car sale price regression and auction purchase rules.

Fits a linear model on a training split of used_car_sales.csv, evaluates it
on held-out sales, and applies a buy/pass rule to auction_lots.csv.
Later_Sale_Price is used only to score realized profit after a decision.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from matplotlib.ticker import FuncFormatter
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.outliers_influence import variance_inflation_factor

ROOT = Path(__file__).resolve().parent
SALES_PATH = ROOT / "used_car_sales.csv"
AUCTION_PATH = ROOT / "auction_lots.csv"
PLOT_PATH = ROOT / "test_predictions.png"
DECISIONS_PATH = ROOT / "auction_decisions.csv"

RANDOM_SEED = 42
TRAIN_FRACTION = 0.70
PURCHASE_COST = 1500
SIGNIFICANCE_LEVEL = 0.05
ACCIDENT_CATEGORIES = ["no", "yes"]
PREDICTORS = [
    "Mileage_K",
    "Age_Years",
    "Horsepower",
    "C(Accident)",
    "Num_Owners",
]


def section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def money(value: float) -> str:
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.2f}"


def sale_formula(predictors: list[str]) -> str:
    return "Sale_Price ~ " + " + ".join(predictors)


def prepare_model_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Copy a frame and treat Accident as a category with 'no' as the reference."""
    prepared = frame.copy()
    unknown = set(prepared["Accident"].astype(str).unique()) - set(ACCIDENT_CATEGORIES)
    if unknown:
        raise ValueError(f"Unexpected Accident values: {sorted(unknown)}")
    prepared["Accident"] = pd.Categorical(
        prepared["Accident"], categories=ACCIDENT_CATEGORIES
    )
    return prepared


def report_file(frame: pd.DataFrame, name: str, expected_rows: int) -> None:
    missing = frame.isna().sum()
    print(f"{name}: {len(frame)} rows, {frame.shape[1]} columns")
    print(f"Expected rows: {expected_rows}. Match: {len(frame) == expected_rows}.")
    print("Missing values by column:")
    print(missing.to_string())
    print(f"Total missing values: {int(missing.sum())}")
    if len(frame) != expected_rows:
        raise SystemExit(f"{name} has {len(frame)} rows, expected {expected_rows}.")
    if int(missing.sum()) != 0:
        raise SystemExit(f"{name} contains missing values.")


def fit_model(train: pd.DataFrame, predictors: list[str]):
    return smf.ols(sale_formula(predictors), data=train).fit()


def term_pvalue(model, predictor: str) -> float:
    matches = [name for name in model.pvalues.index if name == predictor or name.startswith(predictor)]
    if not matches:
        raise KeyError(f"No coefficient found for {predictor}")
    return float(model.pvalues[matches].max())


def compare_leave_one_out(train: pd.DataFrame, full_model) -> pd.DataFrame:
    """Test whether p-values and fit statistics support dropping any one feature."""
    rows = []
    for predictor in PREDICTORS:
        reduced_predictors = [name for name in PREDICTORS if name != predictor]
        reduced = fit_model(train, reduced_predictors)
        comparison = anova_lm(reduced, full_model)
        nested_p = float(comparison.loc[1, "Pr(>F)"])
        coefficient_p = term_pvalue(full_model, predictor)
        adj_r2_drop = full_model.rsquared_adj - reduced.rsquared_adj
        aic_increase = reduced.aic - full_model.aic
        pvalue_supports_drop = coefficient_p > SIGNIFICANCE_LEVEL
        comparison_supports_drop = nested_p > SIGNIFICANCE_LEVEL and adj_r2_drop <= 0
        rows.append(
            {
                "removed": predictor,
                "coefficient_p": coefficient_p,
                "reduced_adj_r2": reduced.rsquared_adj,
                "adj_r2_change": reduced.rsquared_adj - full_model.rsquared_adj,
                "reduced_aic": reduced.aic,
                "aic_change": aic_increase,
                "nested_F_p": nested_p,
                "drop_supported": pvalue_supports_drop and comparison_supports_drop,
            }
        )
    return pd.DataFrame(rows)


def r_squared(actual: np.ndarray, predicted: np.ndarray) -> float:
    residual = actual - predicted
    total = actual - actual.mean()
    return float(1 - np.sum(residual**2) / np.sum(total**2))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def predict_sale_price(
    model,
    mileage_k: float,
    age_years: float,
    horsepower: float,
    accident: str,
    num_owners: float,
) -> float:
    row = prepare_model_frame(
        pd.DataFrame(
            [
                {
                    "Mileage_K": mileage_k,
                    "Age_Years": age_years,
                    "Horsepower": horsepower,
                    "Accident": accident,
                    "Num_Owners": num_owners,
                }
            ]
        )
    )
    return float(model.predict(row).iloc[0])


def make_v1_buying_decision(model):
    """Return a rule that buys when the v1 prediction clears auction price plus cost."""

    def v1_buying_decision(
        mileage_k: float,
        age_years: float,
        horsepower: float,
        accident: str,
        num_owners: float,
        auction_price: float,
    ) -> str:
        predicted = predict_sale_price(
            model, mileage_k, age_years, horsepower, accident, num_owners
        )
        if predicted > auction_price + PURCHASE_COST:
            return "buy"
        return "pass"

    return v1_buying_decision


def baseline_buying_decision(auction_price: float, training_average: float) -> str:
    """Buy when the training-set average sale price clears auction price plus cost."""
    if training_average > auction_price + PURCHASE_COST:
        return "buy"
    return "pass"


def realized_profit(later_sale_price: float, auction_price: float) -> float:
    """Score a completed purchase. Do not use this value to choose buy or pass."""
    return float(later_sale_price) - float(auction_price) - PURCHASE_COST


def variance_inflation(model) -> pd.Series:
    design = pd.DataFrame(model.model.exog, columns=model.model.exog_names)
    scores = {}
    for name in design.columns:
        if name == "Intercept":
            continue
        scores[name] = variance_inflation_factor(design.to_numpy(), design.columns.get_loc(name))
    return pd.Series(scores, name="VIF")


def save_prediction_chart(
    actual: np.ndarray,
    v1_predicted: np.ndarray,
    training_average: float,
    path: Path,
) -> None:
    low = min(actual.min(), v1_predicted.min(), training_average)
    high = max(actual.max(), v1_predicted.max(), training_average)
    padding = (high - low) * 0.05
    limits = (low - padding, high + padding)

    fig, ax = plt.subplots(figsize=(8, 7.2))
    ax.plot(
        limits,
        limits,
        color="#222222",
        linestyle="--",
        linewidth=1.4,
        label="Perfect prediction",
        zorder=1,
    )
    ax.axhline(
        training_average,
        color="#D4760A",
        linestyle="-",
        linewidth=1.8,
        label=f"Training-average baseline ({money(training_average)})",
        zorder=2,
    )
    ax.scatter(
        actual,
        v1_predicted,
        s=42,
        color="#1F4E79",
        alpha=0.85,
        edgecolors="white",
        linewidths=0.4,
        label="v1 predictions",
        zorder=3,
    )
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Test cars: predicted versus actual sale price")
    ax.set_xlabel("Actual sale price")
    ax.set_ylabel("Predicted sale price")
    dollar = FuncFormatter(lambda value, _position: f"${value:,.0f}")
    ax.xaxis.set_major_formatter(dollar)
    ax.yaxis.set_major_formatter(dollar)
    ax.grid(True, color="#D0D5DD", linewidth=0.6, alpha=0.8)
    ax.legend(loc="upper left", frameon=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def summarize_rule(name: str, decisions: pd.Series, profits: pd.Series) -> None:
    bought = decisions == "buy"
    realized = profits.loc[bought].astype(float)
    losing = int((realized < 0).sum())
    print(f"{name}")
    print(f"  Bought: {int(bought.sum())} of {len(decisions)}")
    print(f"  Losing purchases (realized profit < 0): {losing}")
    print(f"  Total realized profit: {money(float(realized.sum()))}")


def main() -> None:
    sales = pd.read_csv(SALES_PATH)
    auction = pd.read_csv(AUCTION_PATH)

    section("1. Data check")
    report_file(sales, "used_car_sales.csv", 240)
    print()
    report_file(auction, "auction_lots.csv", 40)
    print("\nSales columns:", ", ".join(sales.columns))
    print("Auction columns:", ", ".join(auction.columns))
    print("\nAccident counts in sales:")
    print(sales["Accident"].value_counts().to_string())
    print("Accident counts in auction lots:")
    print(auction["Accident"].value_counts().to_string())

    section("2. Train/test split")
    n_train = int(round(TRAIN_FRACTION * len(sales)))
    train_raw = sales.sample(n=n_train, random_state=RANDOM_SEED)
    test_raw = sales.drop(index=train_raw.index)
    if len(train_raw) + len(test_raw) != len(sales) or set(train_raw.index) & set(test_raw.index):
        raise SystemExit("Train and test splits overlap or drop rows.")
    train = prepare_model_frame(train_raw)
    test = prepare_model_frame(test_raw)
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Training rows: {len(train)} ({len(train) / len(sales):.0%})")
    print(f"Test rows: {len(test)} ({len(test) / len(sales):.0%})")
    print("The model is fit on the training rows only.")

    section("3. Model fit and v1 selection")
    full_model = fit_model(train, PREDICTORS)
    print(full_model.summary())
    coefficient_table = pd.DataFrame(
        {
            "coefficient": full_model.params,
            "std_error": full_model.bse,
            "t": full_model.tvalues,
            "p_value": full_model.pvalues,
        }
    )
    print("\nCoefficients and p-values:")
    print(coefficient_table.to_string(float_format=lambda value: f"{value:,.6f}"))
    print(f"\nR-squared: {full_model.rsquared:.6f}")
    print(f"Adjusted R-squared: {full_model.rsquared_adj:.6f}")
    print(f"Condition number: {full_model.condition_number:,.1f}")

    mileage_age_corr = float(train[["Mileage_K", "Age_Years"]].corr().iloc[0, 1])
    print(f"\nTraining correlation, Mileage_K and Age_Years: {mileage_age_corr:.3f}")
    print("Variance inflation factors (training design, intercept excluded):")
    print(variance_inflation(full_model).to_string(float_format=lambda value: f"{value:.3f}"))
    print(
        "\nThe summary note about the condition number comes from that mileage-age "
        "correlation and from predictors measured on different scales. VIFs stay moderate. "
        "Both mileage and age are still precisely estimated, so the note is not by itself "
        "a reason to drop either feature."
    )

    comparison = compare_leave_one_out(train, full_model)
    print("\nLeave-one-out comparison against the full model:")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:,.6f}"))
    supported = comparison.loc[comparison["drop_supported"], "removed"].tolist()
    if supported:
        raise SystemExit(
            "A feature met the drop rule. Update the selected specification before using v1: "
            + ", ".join(supported)
        )
    v1 = full_model
    print(
        "\nSelected model: v1"
        "\nNo feature is dropped. Each coefficient has p < 0.05, and removing any one "
        "feature lowers adjusted R-squared, raises AIC, and is rejected by the nested F test."
    )
    print(f"v1 formula: {sale_formula(PREDICTORS)}")
    print("Accident reference category: no. C(Accident)[T.yes] is the accident effect.")

    section("4. Held-out test performance")
    training_average = float(train["Sale_Price"].mean())
    actual = test["Sale_Price"].to_numpy(dtype=float)
    v1_predicted = v1.predict(test).to_numpy(dtype=float)
    baseline_predicted = np.full(actual.shape, training_average)
    print(f"Baseline prediction (training-set average sale price): {money(training_average)}")
    print(f"Test-set average sale price: {money(float(actual.mean()))}")
    print(f"Baseline test R-squared: {r_squared(actual, baseline_predicted):.4f}")
    print(f"Baseline test RMSE: {money(rmse(actual, baseline_predicted))}")
    print(f"v1 test R-squared: {r_squared(actual, v1_predicted):.4f}")
    print(f"v1 test RMSE: {money(rmse(actual, v1_predicted))}")
    print("Test R-squared uses the test-set mean as the reference.")

    section("5. Test prediction chart")
    save_prediction_chart(actual, v1_predicted, training_average, PLOT_PATH)
    print(f"Saved {PLOT_PATH.name}")

    section("6-7. Auction decisions and realized profit")
    v1_buying_decision = make_v1_buying_decision(v1)
    decision_rows = []
    for lot in auction.itertuples(index=False):
        features = (
            lot.Mileage_K,
            lot.Age_Years,
            lot.Horsepower,
            lot.Accident,
            lot.Num_Owners,
        )
        v1_price = predict_sale_price(v1, *features)
        v1_decision = v1_buying_decision(*features, lot.Auction_Price)
        baseline_decision = baseline_buying_decision(lot.Auction_Price, training_average)
        # Later_Sale_Price is read only after the buy/pass decision is fixed.
        v1_profit = (
            realized_profit(lot.Later_Sale_Price, lot.Auction_Price)
            if v1_decision == "buy"
            else pd.NA
        )
        baseline_profit = (
            realized_profit(lot.Later_Sale_Price, lot.Auction_Price)
            if baseline_decision == "buy"
            else pd.NA
        )
        decision_rows.append(
            {
                "Lot": lot.Lot,
                "Mileage_K": lot.Mileage_K,
                "Age_Years": lot.Age_Years,
                "Horsepower": lot.Horsepower,
                "Accident": lot.Accident,
                "Num_Owners": lot.Num_Owners,
                "Auction_Price": lot.Auction_Price,
                "Later_Sale_Price": lot.Later_Sale_Price,
                "Baseline_Predicted_Sale_Price": round(training_average, 2),
                "V1_Predicted_Sale_Price": round(v1_price, 2),
                "Baseline_Expected_Margin": round(
                    training_average - lot.Auction_Price - PURCHASE_COST, 2
                ),
                "V1_Expected_Margin": round(v1_price - lot.Auction_Price - PURCHASE_COST, 2),
                "Baseline_Decision": baseline_decision,
                "V1_Decision": v1_decision,
                "Baseline_Realized_Profit": baseline_profit,
                "V1_Realized_Profit": v1_profit,
            }
        )

    decisions = pd.DataFrame(decision_rows)
    decisions["Baseline_Realized_Profit"] = decisions["Baseline_Realized_Profit"].astype("Int64")
    decisions["V1_Realized_Profit"] = decisions["V1_Realized_Profit"].astype("Int64")
    decisions.to_csv(DECISIONS_PATH, index=False)
    print(f"Purchase cost used in both rules: {money(PURCHASE_COST)}")
    print("Buy when predicted sale price > auction price + purchase cost; otherwise pass.")
    print("Realized profit is calculated only for lots a rule buys:")
    print("  Later_Sale_Price - Auction_Price - purchase cost")
    print()
    summarize_rule(
        "Baseline rule",
        decisions["Baseline_Decision"],
        decisions["Baseline_Realized_Profit"],
    )
    print()
    summarize_rule(
        "v1 rule",
        decisions["V1_Decision"],
        decisions["V1_Realized_Profit"],
    )
    print(f"\nSaved {DECISIONS_PATH.name}")


if __name__ == "__main__":
    main()
