import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CATEGORIES = {
    "Хлопок": ("HLP", 450, 0.30),
    "Лён": ("LEN", 780, 0.15),
    "Шёлк": ("SHK", 1800, 0.06),
    "Вискоза": ("VIS", 520, 0.14),
    "Трикотаж": ("TRK", 620, 0.15),
    "Шерсть": ("SHR", 1400, 0.06),
    "Подкладочные": ("PDK", 300, 0.07),
    "Фурнитура": ("FRN", 90, 0.07),
}

PROFILES = ["stable", "seasonal", "spiky", "growing", "fading"]
PROFILE_WEIGHTS = [0.40, 0.25, 0.20, 0.09, 0.06]

WEEKDAY_FACTORS = np.array([1.0, 1.0, 1.0, 1.05, 1.1, 1.25, 1.2])


def profile_curve(profile, days, rng):
    t = np.arange(days)
    if profile == "stable":
        return np.ones(days)
    if profile == "seasonal":
        phase = rng.uniform(0, 365)
        amplitude = rng.uniform(0.4, 0.8)
        return 1 + amplitude * np.sin(2 * np.pi * (t - phase) / 365)
    if profile == "spiky":
        curve = np.full(days, 0.6)
        for _ in range(rng.integers(3, 7)):
            start = rng.integers(0, days - 10)
            length = rng.integers(4, 11)
            curve[start:start + length] = rng.uniform(2.5, 5.0)
        return curve
    if profile == "growing":
        return np.linspace(0.4, 1.7, days)
    return np.linspace(1.7, 0.4, days)


def build(n_articles, start, days, seed):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=days, freq="D")
    weekday = WEEKDAY_FACTORS[dates.dayofweek.to_numpy()]

    names = list(CATEGORIES)
    shares = np.array([CATEGORIES[n][2] for n in names])
    shares = shares / shares.sum()
    assigned = rng.choice(names, size=n_articles, p=shares)

    counters = {n: 0 for n in names}
    frames = []

    for category in assigned:
        prefix, base_price, _ = CATEGORIES[category]
        counters[category] += 1
        article = f"{prefix}-{counters[category]:03d}"

        profile = rng.choice(PROFILES, p=PROFILE_WEIGHTS)
        base_rate = rng.lognormal(mean=np.log(7), sigma=1.1)
        price = base_price * rng.lognormal(0, 0.25)

        curve = profile_curve(profile, days, rng)
        noise = rng.gamma(shape=4, scale=0.25, size=days)
        lam = base_rate * curve * weekday * noise
        qty = rng.poisson(lam)

        discount = np.where(curve > 2, 0.9, 1.0)
        revenue = qty * price * discount * (1 + rng.normal(0, 0.03, size=days))

        mask = qty > 0
        frames.append(
            pd.DataFrame(
                {
                    "date": dates[mask],
                    "article": article,
                    "category": category,
                    "qty": qty[mask],
                    "revenue": revenue[mask].round(2),
                }
            )
        )

    return pd.concat(frames, ignore_index=True).sort_values(["date", "article"]).reset_index(drop=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--articles", type=int, default=150)
    p.add_argument("--start", default="2025-10-01")
    p.add_argument("--days", type=int, default=365)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default="data/sales.csv")
    args = p.parse_args()

    df = build(args.articles, args.start, args.days, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows, {df['article'].nunique()} articles: {out}")


if __name__ == "__main__":
    main()
