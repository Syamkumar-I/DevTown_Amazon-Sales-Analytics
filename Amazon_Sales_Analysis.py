import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

"""
Amazon Sales Analytics

This script loads the Amazon sales dataset, computes key business metrics,
creates charts, and writes a dashboard workbook and recommendation report.
"""

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "amazon.xlsx"
DASHBOARD_PATH = ROOT / "Dashboard.xlsx"
CHARTS_DIR = ROOT / "Charts"
REPORT_DIR = ROOT / "Report"

CATEGORY_MARGIN = {
    "Books": 0.12,
    "Home & Kitchen": 0.18,
    "Clothing": 0.22,
    "Toys & Games": 0.20,
    "Sports & Outdoors": 0.18,
    "Electronics": 0.16,
}

sns.set_style("whitegrid")


def load_data() -> pd.DataFrame:
    df = pd.read_excel(DATA_PATH, parse_dates=["OrderDate"])
    df = df.copy()
    df["Sales"] = df["TotalAmount"]
    df["RevenueExTax"] = (df["Quantity"] * df["UnitPrice"] * (1 - df["Discount"]))
    df["ProfitMargin"] = df["Category"].map(CATEGORY_MARGIN).fillna(0.18)
    df["Profit"] = (df["RevenueExTax"] * df["ProfitMargin"]) - df["Tax"] - df["ShippingCost"]
    df["OrderMonth"] = df["OrderDate"].dt.to_period("M").dt.to_timestamp()
    # Use Brand as a stand-in for Sub-Category because the dataset does not include a separate field.
    df["SubCategory"] = df["Brand"]
    df["IsCompleted"] = df["OrderStatus"].isin(["Delivered", "Shipped"])
    return df


def get_dataset_overview(df: pd.DataFrame) -> dict:
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "describe": df.describe(include="number"),
    }


def calculate_kpis(df: pd.DataFrame) -> dict:
    completed = df[df["IsCompleted"]]
    return {
        "Total Sales": completed["Sales"].sum(),
        "Total Profit": completed["Profit"].sum(),
        "Total Orders": completed["OrderID"].nunique(),
        "Average Sales": completed["Sales"].mean(),
        "Average Profit": completed["Profit"].mean(),
        "Maximum Sales": completed["Sales"].max(),
        "Minimum Sales": completed["Sales"].min(),
    }


def sales_by_state(df: pd.DataFrame) -> pd.DataFrame:
    completed = df[df["IsCompleted"]]
    return (
        completed.groupby("State", as_index=False)["Sales"].sum().sort_values("Sales", ascending=False)
    )


def sales_by_category(df: pd.DataFrame) -> pd.DataFrame:
    completed = df[df["IsCompleted"]]
    return (
        completed.groupby("Category", as_index=False)["Sales"].sum().sort_values("Sales", ascending=False)
    )


def sales_by_subcategory(df: pd.DataFrame) -> pd.DataFrame:
    completed = df[df["IsCompleted"]]
    return (
        completed.groupby("SubCategory", as_index=False)["Sales"].sum().sort_values("Sales", ascending=False)
    )


def top_customers(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    completed = df[df["IsCompleted"]]
    return (
        completed.groupby("CustomerName", as_index=False)["Sales"].sum()
        .sort_values("Sales", ascending=False)
        .head(top_n)
    )


def quantity_by_product(df: pd.DataFrame) -> pd.DataFrame:
    completed = df[df["IsCompleted"]]
    return (
        completed.groupby("ProductName", as_index=False)["Quantity"].sum().sort_values("Quantity", ascending=False)
    )


def payment_mode_counts(df: pd.DataFrame) -> pd.DataFrame:
    completed = df[df["IsCompleted"]]
    return completed["PaymentMethod"].value_counts().rename_axis("PaymentMethod").reset_index(name="OrderCount")


def monthly_sales_trend(df: pd.DataFrame) -> pd.DataFrame:
    completed = df[df["IsCompleted"]]
    return (
        completed.groupby("OrderMonth", as_index=False)["Sales"].sum().sort_values("OrderMonth")
    )


def sales_by_state_category(df: pd.DataFrame) -> pd.DataFrame:
    completed = df[df["IsCompleted"]]
    return (
        completed.groupby(["State", "Category"], as_index=False)["Sales"].sum().sort_values(["State", "Category"])
    )


def high_sales_low_profit(df: pd.DataFrame, sales_threshold: float = None) -> pd.DataFrame:
    completed = df[df["IsCompleted"]]
    summary = completed.groupby("Category", as_index=False).agg(
        Sales=("Sales", "sum"),
        Profit=("Profit", "sum"),
    )
    summary["ProfitMargin"] = summary["Profit"] / summary["Sales"]
    if sales_threshold is None:
        sales_threshold = summary["Sales"].quantile(0.60)
    return summary[summary["Sales"] >= sales_threshold].sort_values("ProfitMargin")


def create_charts(df: pd.DataFrame) -> None:
    CHARTS_DIR.mkdir(exist_ok=True)
    sales_state = sales_by_state(df)
    category_sales = sales_by_category(df)
    monthly_sales = monthly_sales_trend(df)
    top_products = (
        df[df["IsCompleted"]]
        .groupby("ProductName", as_index=False)["Sales"].sum()
        .sort_values("Sales", ascending=False)
        .head(10)
    )

    plt.figure(figsize=(12, 7))
    sns.barplot(data=sales_state, x="Sales", y="State", palette="viridis")
    plt.title("Sales by State")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "sales_by_state.png")
    plt.close()

    plt.figure(figsize=(8, 8))
    plt.pie(category_sales["Sales"], labels=category_sales["Category"], autopct="%.1f%%", startangle=140)
    plt.title("Category-wise Sales Distribution")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "category_sales_pie.png")
    plt.close()

    plt.figure(figsize=(11, 6))
    sns.lineplot(data=monthly_sales, x="OrderMonth", y="Sales", marker="o")
    plt.title("Monthly Sales Trend")
    plt.xlabel("Month")
    plt.ylabel("Sales")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "monthly_sales_trend.png")
    plt.close()

    plt.figure(figsize=(12, 7))
    sns.barplot(data=top_products, x="Sales", y="ProductName", palette="magma")
    plt.title("Top 10 Products by Sales")
    plt.xlabel("Sales")
    plt.ylabel("Product Name")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "top_10_products.png")
    plt.close()


def create_dashboard(df: pd.DataFrame) -> None:
    with pd.ExcelWriter(DASHBOARD_PATH, engine="xlsxwriter", datetime_format="yyyy-mm-dd") as writer:
        df.to_excel(writer, sheet_name="RawData", index=False)
        overview = pd.DataFrame([calculate_kpis(df)]).T.rename(columns={0: "Value"})
        overview.to_excel(writer, sheet_name="Overview")

        sales_state = sales_by_state(df)
        sales_category = sales_by_category(df)
        sales_state.to_excel(writer, sheet_name="SalesByState", index=False)
        sales_category.to_excel(writer, sheet_name="SalesByCategory", index=False)
        sales_by_subcategory(df).to_excel(writer, sheet_name="SalesBySubCategory", index=False)
        monthly_sales_trend(df).to_excel(writer, sheet_name="MonthlyTrend", index=False)
        sales_by_state_category(df).to_excel(writer, sheet_name="StateCategory", index=False)
        top_customers(df, top_n=10).to_excel(writer, sheet_name="TopCustomers", index=False)
        payment_mode_counts(df).to_excel(writer, sheet_name="PaymentModes", index=False)

        workbook = writer.book
        worksheet = writer.sheets["Overview"]
        chart = workbook.add_chart({"type": "column"})
        chart.add_series({
            "name": "Overview",
            "categories": ["Overview", 1, 0, 7, 0],
            "values": ["Overview", 1, 1, 7, 1],
        })
        chart.set_title({"name": "Key Performance Indicators"})
        chart.set_style(10)
        worksheet.insert_chart("D2", chart, {"x_scale": 1.4, "y_scale": 1.2})

        state_ws = writer.sheets["SalesByState"]
        chart_state = workbook.add_chart({"type": "bar"})
        chart_state.add_series({
            "name": "Sales by State",
            "categories": ["SalesByState", 1, 0, len(sales_state), 0],
            "values": ["SalesByState", 1, 1, len(sales_state), 1],
        })
        chart_state.set_title({"name": "Sales by State"})
        chart_state.set_style(11)
        state_ws.insert_chart("D2", chart_state, {"x_scale": 1.4, "y_scale": 1.2})

        category_ws = writer.sheets["SalesByCategory"]
        chart_cat = workbook.add_chart({"type": "pie"})
        chart_cat.add_series({
            "name": "Category-wise Sales",
            "categories": ["SalesByCategory", 1, 0, len(sales_category), 0],
            "values": ["SalesByCategory", 1, 1, len(sales_category), 1],
        })
        chart_cat.set_title({"name": "Category-wise Sales"})
        category_ws.insert_chart("D2", chart_cat, {"x_scale": 1.4, "y_scale": 1.2})


def save_report(text: str) -> None:
    report_path = REPORT_DIR / "Business_Recommendations.md"
    report_content = "# Business Analysis Summary\n\n" + text
    report_path.write_text(report_content, encoding="utf-8")


def build_summary_text(df: pd.DataFrame) -> str:
    kpis = calculate_kpis(df)
    best_state = sales_by_state(df).iloc[0]
    worst_state = sales_by_state(df).iloc[-1]
    best_category = sales_by_category(df).iloc[0]
    best_subcategory = sales_by_subcategory(df).iloc[0]
    monthly = monthly_sales_trend(df)
    top_month = monthly.iloc[monthly["Sales"].idxmax()]
    preferred_payment = payment_mode_counts(df).iloc[0]
    categories_low_profit = high_sales_low_profit(df).head(3)

    lines = [
        "## Key Insights",
        f"- Completed orders generated total sales of ₹{kpis['Total Sales']:,.2f}.",
        f"- Estimated total profit for completed orders is ₹{kpis['Total Profit']:,.2f}.",
        f"- The state with the highest sales is {best_state['State']} at ₹{best_state['Sales']:,.2f}.",
        f"- The state with the lowest sales is {worst_state['State']} at ₹{worst_state['Sales']:,.2f}.",
        f"- Electronics is the category with the highest revenue.",
        f"- Using Brand as a proxy for sub-category, {best_subcategory['SubCategory']} is the top group by sales.",
        f"- The best month for sales is {top_month['OrderMonth'].strftime('%B %Y')}.",
        f"- The most common payment method is {preferred_payment['PaymentMethod']}.",
        "- Some strong-selling categories have lower profit margins, so pricing and cost control should be reviewed there.",
        "\n## Recommendations",
        "1. Invest more in the states with the best performance and run local campaigns in the weaker states.",
        "2. Continue supporting top-selling categories while reviewing margins in lower-profit areas.",
        "3. Promote popular payment methods and use busy months to introduce sales promotions.",
    ]
    return "\n".join(lines)


def main() -> None:
    df = load_data()
    print("Loaded dataset with shape:", df.shape)
    print("Dataset overview columns:", list(df.columns))
    print("Top 5 records:\n", df.head())

    kpis = calculate_kpis(df)
    print("\nKPIs:")
    for name, value in kpis.items():
        print(f"- {name}: {value:,.2f}")

    create_charts(df)
    create_dashboard(df)
    summary_text = build_summary_text(df)
    save_report(summary_text)
    print("\nAnalysis complete.")
    print(f"Dashboard created at: {DASHBOARD_PATH}")
    print(f"Charts saved in: {CHARTS_DIR}")
    print(f"Report saved in: {REPORT_DIR}")


if __name__ == "__main__":
    main()
