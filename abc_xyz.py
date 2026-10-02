import argparse

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["date", "article", "qty", "revenue"]

RECOMMENDATIONS = {
    "AX": "Ключевой стабильный товар: постоянный запас, автопополнение, минимальный страховой запас.",
    "AY": "Важный товар с колебаниями: повышенный страховой запас, пополнение по точке заказа.",
    "AZ": "Важный, но непредсказуемый товар: страховой запас по прогнозу, ручной контроль, разбор причин всплесков.",
    "BX": "Средний вклад, стабильный спрос: регулярные поставки по расписанию.",
    "BY": "Средний вклад, умеренные колебания: пополнение по точке заказа, периодический пересмотр.",
    "BZ": "Средний вклад, нестабильный спрос: небольшие партии, поставки под акции.",
    "CX": "Малый вклад, стабильный спрос: минимальный запас, редкие укрупнённые поставки.",
    "CY": "Малый вклад, колебания: поставки по факту, контроль оборачиваемости.",
    "CZ": "Малый вклад и непредсказуемый спрос: кандидат на вывод из ассортимента или работу под заказ.",
}


def load_sales(source):
    df = pd.read_csv(source)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    df["date"] = pd.to_datetime(df["date"])
    df["article"] = df["article"].astype(str)
    df["qty"] = pd.to_numeric(df["qty"])
    df["revenue"] = pd.to_numeric(df["revenue"])
    if "category" not in df.columns:
        df["category"] = "Без категории"
    return df


def classify_abc(df, a=0.8, b=0.95):
    rev = df.groupby("article", as_index=False)["revenue"].sum()
    rev = rev.sort_values("revenue", ascending=False).reset_index(drop=True)
    total = rev["revenue"].sum()
    rev["share"] = rev["revenue"] / total if total else 0.0
    rev["cum_share"] = rev["share"].cumsum()
    before = rev["cum_share"] - rev["share"]
    rev["abc"] = np.where(before < a, "A", np.where(before < b, "B", "C"))
    return rev


def complete_periods(df, freq):
    periods = df["date"].dt.to_period(freq)
    full = pd.period_range(periods.min(), periods.max(), freq=freq)
    keep = list(full)
    if len(keep) > 2:
        if df["date"].min() > keep[0].start_time:
            keep = keep[1:]
        if df["date"].max().normalize() < keep[-1].end_time.normalize():
            keep = keep[:-1]
    if len(keep) < 3:
        keep = list(full)
    return keep


def classify_xyz(df, freq="M", x=0.25, y=0.5):
    keep = complete_periods(df, freq)
    period = df["date"].dt.to_period(freq)
    pivot = (
        df.assign(period=period)
        .pivot_table(index="article", columns="period", values="qty", aggfunc="sum", fill_value=0)
        .reindex(columns=keep, fill_value=0)
    )
    mean = pivot.mean(axis=1)
    std = pivot.std(axis=1, ddof=0)
    cv = (std / mean).replace([np.inf, -np.inf], np.nan)
    out = pd.DataFrame({"article": pivot.index, "mean_qty": mean.values, "cv": cv.values})
    out["xyz"] = np.where(out["cv"] <= x, "X", np.where(out["cv"] <= y, "Y", "Z"))
    return out


def analyze(df, freq="M", abc_a=0.8, abc_b=0.95, xyz_x=0.25, xyz_y=0.5):
    abc = classify_abc(df, abc_a, abc_b)
    xyz = classify_xyz(df, freq, xyz_x, xyz_y)
    meta = df.groupby("article", as_index=False).agg(category=("category", "first"), qty=("qty", "sum"))
    result = abc.merge(xyz, on="article").merge(meta, on="article")
    result["segment"] = result["abc"] + result["xyz"]
    result["recommendation"] = result["segment"].map(RECOMMENDATIONS)
    return result.sort_values("revenue", ascending=False).reset_index(drop=True)


def segment_matrix(result, value="count"):
    if value == "count":
        m = result.pivot_table(index="abc", columns="xyz", values="article", aggfunc="count", fill_value=0)
    else:
        total = result["revenue"].sum()
        m = result.pivot_table(index="abc", columns="xyz", values="revenue", aggfunc="sum", fill_value=0)
        m = m / total if total else m
    return m.reindex(index=list("ABC"), columns=list("XYZ"), fill_value=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/sales.csv")
    p.add_argument("--freq", choices=["W", "M"], default="M")
    p.add_argument("--abc-a", type=float, default=0.8)
    p.add_argument("--abc-b", type=float, default=0.95)
    p.add_argument("--xyz-x", type=float, default=0.25)
    p.add_argument("--xyz-y", type=float, default=0.5)
    p.add_argument("--out", default="abc_xyz_result.csv")
    args = p.parse_args()

    df = load_sales(args.input)
    result = analyze(df, args.freq, args.abc_a, args.abc_b, args.xyz_x, args.xyz_y)
    result.to_csv(args.out, index=False, encoding="utf-8-sig")

    print("Articles per segment:")
    print(segment_matrix(result, "count").to_string())
    print()
    print("Revenue share per segment, %:")
    print((segment_matrix(result, "revenue") * 100).round(1).to_string())
    print()
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
