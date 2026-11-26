import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

data=yf.download(
    tickers=["SPY", "FLCSX"],
    start="2016-01-01",
    end="2026-11-14",
    interval="1d",
    auto_adjust=False,
    actions=True,
    progress=False
)

prices=data["Adj Close"]
prices=prices[["SPY", "FLCSX"]]



daily_returns=prices.pct_change(fill_method=None)
daily_returns=daily_returns.dropna()
historical_returns=daily_returns[["SPY", "FLCSX"]].to_numpy()

start_investment=10000
n_simulations=100000
simulation_returns=np.zeros((n_simulations, 2))
rng=np.random.default_rng()
n_days=21
n_paths=min(250, n_simulations)
saved_paths=np.zeros((n_paths, n_days+1, 2))

for simulation in range(n_simulations):
    
    sampled_returns=rng.choice(
        historical_returns,
        size=n_days,
        replace=True,
        axis=0
    )
    values=np.array([start_investment, start_investment])
    investment_path=np.zeros((len(sampled_returns)+1,2))
    investment_path[0]=values

    for day, returns_pair in enumerate(sampled_returns, start=1):
        values=values*(1+returns_pair)
        investment_path[day]=values

    if simulation<n_paths:
        saved_paths[simulation]=investment_path

    monthly_returns=values/start_investment - 1 
    simulation_returns[simulation]=monthly_returns
    
flcsx_ahead=simulation_returns[:,1]>simulation_returns[:,0]
win=np.sum(flcsx_ahead)
win_rate=win/n_simulations
average_returns=np.average(simulation_returns, axis=0) 
excess_returns=simulation_returns[:,1]-simulation_returns[:,0]
average_gap=np.mean(excess_returns)
lower_gap = np.percentile(excess_returns, 5)
upper_gap = np.percentile(excess_returns, 95)

plot_returns=simulation_returns*100
bin_edges=np.linspace(
    plot_returns.min(),
    plot_returns.max(),
    61
)

standard_deviations = np.std(simulation_returns, axis=0)
plot_returns = simulation_returns * 100

spy_p5, spy_p95 = np.percentile(plot_returns[:, 0], [5, 95])
flcsx_p5, flcsx_p95 = np.percentile(plot_returns[:, 1], [5, 95])

print(f"FLCSX wins {win} times")
print(f"FLCSX performance: {win_rate:.1%}")
print(simulation_returns)
print(f"Average returns SPY: {average_returns[0]:.2%}")
print(f"Average returns FLCSX: {average_returns[1]:.2%}")
print(f"Average gap: {average_gap * 100:.3f} percentage points")
print(f"gap 5th percentile: {lower_gap * 100:.3f} percentage points")
print(f"gap 95th percentile: {upper_gap * 100:.3f} percentage points")
print(f"SPY standard deviation: {standard_deviations[0] * 100:.2f} percentage points")
print(f"FLCSX standard deviation: {standard_deviations[1] * 100:.2f} percentage points")
print(f"SPY 5th percentile: {spy_p5:.3f} percentage points")
print(f"SPY 95th percentile: {spy_p95:.3f} percentage points")
print(f"FLCSX 5th percentile: {flcsx_p5:.3f} percentage points")
print(f"FLCSX 95th percentile: {flcsx_p95:.3f} percentage points")

plt.figure(figsize=(10,6))

plt.hist(
    plot_returns[:,0],
    bins=bin_edges,
    alpha=0.55,
    color="#0072B2",
    label="SPY"
)

plt.hist(
    plot_returns[:,1],
    bins=bin_edges,
    alpha=0.55,
    color="#E69F00",
    label="FLCSX"
)

plt.axvline(
    spy_p5,
    color="#0072B2",
    linestyle="--",
    linewidth=1.5,
    label="SPY: 5th / 95th percentiles"
)
plt.axvline(spy_p95, color="#0072B2", linestyle="--", linewidth=1.5)

plt.axvline(
    flcsx_p5,
    color="#E69F00",
    linestyle=":",
    linewidth=2,
    label="FLCSX: 5th / 95th percentiles"
)
plt.axvline(flcsx_p95, color="#E69F00", linestyle=":", linewidth=2)
plt.axvline(0, color="black", linestyle="--", linewidth=1)

plt.xlabel("Simulated 21 day return (%)", fontsize=12)
plt.ylabel("Number of simulated months", fontsize=12)
plt.title("SPY vs FLCSX: simulated return distributions", fontsize=15)

plt.legend(frameon=False)
plt.grid(axis="y", alpha=0.2)
plt.tight_layout()

plt.savefig(
    "return_distributions.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

days=np.arange(n_days+1)

fig, axes = plt.subplots(
    1,2,
    figsize=(12,5),
    sharex=True,
    sharey=True
)
for n in range(n_paths):
    axes[0].plot(
        days, 
        saved_paths[n, :, 0],
        color="#0072B2",
        linewidth=1.5,
        label="SPY",
        alpha=0.25
    )

    axes[1].plot(
        days, 
        saved_paths[n, :, 1],
        color="#E66B00",
        linewidth=1.5,
        label="FLCSX",
        alpha=0.25
    )

axes[0].set_title("SPY")
axes[1].set_title("FLCSX")
axes[0].set_ylabel("Investment value (USD)", fontsize=12)

for ax in axes:
    ax.set_xlabel("Simulated trading day", fontsize=12)
    ax.axhline(
        start_investment,
        color="black",
        linestyle="--",
        linewidth=1
    )

fig.suptitle(
    f"{n_paths} simulated scenarios - 21 trading days", 
    fontsize=15
)
fig.tight_layout()
plt.show()