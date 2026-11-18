import os, sys, json
from pathlib import Path
import numpy as np
import pandas as pd

# Setup paths
repo_root = Path(".").resolve()
package_dir = repo_root / "quant_review"
sys.path.insert(0, str(package_dir))

from exit_engine import ExitPolicy, run_policy

RES = package_dir / "results"

BASKET = [
    "NAS100.p", "BTCUSD.pi", "UK100.p", "AU200.p", "EURUSD.pi",
    "SOLUSD.p", "USDCHF.pi", "USDJPY.pi", "XAGUSD.pi", "JP225.p"
]

COST_R = {
    "EURUSD.pi": 0.017, "USDJPY.pi": 0.019, "USDCHF.pi": 0.025,
    "BTCUSD.pi": 0.044, "SOLUSD.p": 0.072, "NAS100.p": 0.031,
    "UK100.p": 0.034, "AU200.p": 0.037, "JP225.p": 0.078, "XAGUSD.pi": 0.017,
}

R_DOLLAR = 10.0
INIT_EQ = 2500.0

def sym_key(s):
    return s.replace(".", "_")

def calc_stats(t_df, initial_equity=INIT_EQ):
    if len(t_df) == 0:
        return {
            "n": 0, "total_r": 0.0, "net_usd": 0.0, "pf": 0.0,
            "max_dd_usd": 0.0, "max_dd_pct": 0.0, "expectancy_r": 0.0,
            "hit2r_pct": 0.0, "be_pct": 0.0, "loss_pct": 0.0,
            "avg_win_r": 0.0, "avg_runner_r": 0.0
        }
    t = t_df.sort_values("entry_time").reset_index(drop=True)
    pnl = t["total_r"] * R_DOLLAR
    eq = initial_equity + pnl.cumsum()
    peak = eq.cummax()
    dd = eq - peak
    gw = pnl[pnl > 0].sum()
    gl = abs(pnl[pnl < 0].sum())
    
    return {
        "n": int(len(t)),
        "total_r": round(float(t["total_r"].sum()), 2),
        "net_usd": round(float(pnl.sum()), 2),
        "pf": round(float(gw / gl), 2) if gl > 0 else (99.0 if gw > 0 else 0.0),
        "max_dd_usd": round(float(dd.min()), 2),
        "max_dd_pct": round(float(dd.min() / peak.max() * 100), 2),
        "expectancy_r": round(float(t["total_r"].mean()), 3),
        "hit2r_pct": round(float(t["hit_2r"].mean() * 100), 1),
        "be_pct": round(float(((t["total_r"] >= 0) & (t["total_r"] <= 0.5)).mean() * 100), 1),
        "loss_pct": round(float((t["total_r"] < 0).mean() * 100), 1),
        "avg_win_r": round(float(t.loc[t["total_r"] > 0.5, "total_r"].mean()), 2) if (t["total_r"] > 0.5).any() else 0.0,
        "avg_runner_r": round(float(t.loc[t["hit_2r"], "t2_r"].mean()), 2) if t["hit_2r"].any() else 0.0,
    }

def main():
    # Load frames
    frames = {}
    for sym in BASKET:
        k = sym_key(sym)
        mp = RES / f"merged_{k}.parquet"
        sp = RES / f"setups_{k}.parquet"
        if mp.exists() and sp.exists():
            frames[sym] = (pd.read_parquet(mp), pd.read_parquet(sp))
            
    print(f"Loaded {len(frames)} assets.")
    
    pol_b4 = ExitPolicy("chandelier", k=1.5, intrabar="pessimistic",
                        entry_bar_adverse=True, trail_same_bar=False, gap_fill=True)
    pol_psar = ExitPolicy("psar", psar_af0=0.02, psar_af_max=0.20,
                          intrabar="pessimistic",
                          entry_bar_adverse=True, trail_same_bar=False, gap_fill=True)
    pol_a0 = ExitPolicy("chandelier", k=1.5, intrabar="optimistic",
                        entry_bar_adverse=False, trail_same_bar=True, gap_fill=False)

    policies = {
        "Base_Honest_B4": (pol_b4, lambda s: COST_R.get(s, 0.03)),
        "Best_PSAR_Runner": (pol_psar, lambda s: COST_R.get(s, 0.03)),
        "Legacy_Repo_A0": (pol_a0, None)
    }

    # Generate all trade logs
    all_trades_by_pol = {}
    for pol_name, (pol, cost_fn) in policies.items():
        trade_list = []
        for sym, (merged, setups) in frames.items():
            t = run_policy(merged, setups, pol, cost_r_fn=cost_fn)
            if len(t):
                trade_list.append(t)
        all_trades_by_pol[pol_name] = pd.concat(trade_list, ignore_index=True)

    # Let's inspect the date range for trades
    sample_trades = all_trades_by_pol["Best_PSAR_Runner"]
    max_date = pd.to_datetime(sample_trades["entry_time"]).max()
    six_months_ago = max_date - pd.DateOffset(months=6)
    
    print(f"\nMax trade entry date in dataset: {max_date}")
    print(f"Last 6-month cutoff date: {six_months_ago}\n")

    # Asset-wise breakdown for Last 6 Months (May 16, 2026 - Nov 16, 2026)
    out_data = {
        "meta": {
            "max_date": str(max_date),
            "cutoff_6m": str(six_months_ago),
            "account_equity": INIT_EQ,
            "risk_per_trade_usd": R_DOLLAR
        },
        "last_6m": {},
        "full_history": {}
    }

    for pol_name, df_trades in all_trades_by_pol.items():
        df_trades["entry_dt"] = pd.to_datetime(df_trades["entry_time"])
        df_6m = df_trades[df_trades["entry_dt"] >= six_months_ago].copy()
        
        # Portfolio level
        port_full = calc_stats(df_trades)
        port_6m = calc_stats(df_6m)
        
        # Asset level in 6m
        assets_6m = {}
        for sym in BASKET:
            df_sym_6m = df_6m[df_6m["symbol"] == sym]
            assets_6m[sym] = calc_stats(df_sym_6m)
            
        out_data["last_6m"][pol_name] = {
            "portfolio": port_6m,
            "assets": assets_6m
        }
        out_data["full_history"][pol_name] = {
            "portfolio": port_full
        }

    # Print summary tables
    print("=" * 95)
    print("LAST 6 MONTHS PERFORMANCE (Asset-by-Asset) — BEST PSAR RUNNER (Costs Included)")
    print("=" * 95)
    print(f"{'Asset':<12} | {'Trades':<6} | {'2R Hit %':<8} | {'BE %':<6} | {'Loss %':<6} | {'Total R':<8} | {'Net USD':<9} | {'PF':<6} | {'Max DD ($)':<10} | {'Exp (R)':<7}")
    print("-" * 95)
    
    psar_6m_assets = out_data["last_6m"]["Best_PSAR_Runner"]["assets"]
    for sym in BASKET:
        st = psar_6m_assets[sym]
        print(f"{sym:<12} | {st['n']:<6} | {st['hit2r_pct']:<7.1f}% | {st['be_pct']:<5.1f}% | {st['loss_pct']:<5.1f}% | {st['total_r']:<+8.2f} | ${st['net_usd']:<+8.2f} | {st['pf']:<6.2f} | ${st['max_dd_usd']:<9.2f} | {st['expectancy_r']:<+7.3f}")

    print("-" * 95)
    p_st = out_data["last_6m"]["Best_PSAR_Runner"]["portfolio"]
    print(f"{'PORTFOLIO':<12} | {p_st['n']:<6} | {p_st['hit2r_pct']:<7.1f}% | {p_st['be_pct']:<5.1f}% | {p_st['loss_pct']:<5.1f}% | {p_st['total_r']:<+8.2f} | ${p_st['net_usd']:<+8.2f} | {p_st['pf']:<6.2f} | ${p_st['max_dd_usd']:<9.2f} | {p_st['expectancy_r']:<+7.3f}")
    print("=" * 95)

    print("\n" + "=" * 95)
    print("LAST 6 MONTHS PERFORMANCE (Asset-by-Asset) — BASE HONEST B4 (Costs Included)")
    print("=" * 95)
    print(f"{'Asset':<12} | {'Trades':<6} | {'2R Hit %':<8} | {'BE %':<6} | {'Loss %':<6} | {'Total R':<8} | {'Net USD':<9} | {'PF':<6} | {'Max DD ($)':<10} | {'Exp (R)':<7}")
    print("-" * 95)
    
    b4_6m_assets = out_data["last_6m"]["Base_Honest_B4"]["assets"]
    for sym in BASKET:
        st = b4_6m_assets[sym]
        print(f"{sym:<12} | {st['n']:<6} | {st['hit2r_pct']:<7.1f}% | {st['be_pct']:<5.1f}% | {st['loss_pct']:<5.1f}% | {st['total_r']:<+8.2f} | ${st['net_usd']:<+8.2f} | {st['pf']:<6.2f} | ${st['max_dd_usd']:<9.2f} | {st['expectancy_r']:<+7.3f}")

    print("-" * 95)
    p_b4 = out_data["last_6m"]["Base_Honest_B4"]["portfolio"]
    print(f"{'PORTFOLIO':<12} | {p_b4['n']:<6} | {p_b4['hit2r_pct']:<7.1f}% | {p_b4['be_pct']:<5.1f}% | {p_b4['loss_pct']:<5.1f}% | {p_b4['total_r']:<+8.2f} | ${p_b4['net_usd']:<+8.2f} | {p_b4['pf']:<6.2f} | ${p_b4['max_dd_usd']:<9.2f} | {p_b4['expectancy_r']:<+7.3f}")
    print("=" * 95)

    with open("quant_review/results/last_6m_results.json", "w") as f:
        json.dump(out_data, f, indent=2)
    print("\nSaved full last 6-month breakdown to quant_review/results/last_6m_results.json")

if __name__ == "__main__":
    main()
