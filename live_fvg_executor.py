#!/usr/bin/env python3
"""
live_fvg_executor.py — Institutional Live MT5 Automated Execution Engine
========================================================================
Strategy: Multi-Asset Mechanical FVG Momentum with 2R Take-Profit + Trailing Runner
Account: Blueberry Markets MT5 (Account #5126976)
Isolation: Comment 'fvg_strategy', Magic Number 7771399
Risk: Exactly $10.00 USD per trade on $2,500 capital (0.40% risk)
Heat Budget: Maximum 3 simultaneous active positions
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import numpy as np

try:
    import MetaTrader5 as mt5
except ImportError:
    print("ERROR: MetaTrader5 package is not installed. Please install it with 'pip install MetaTrader5'.")
    sys.exit(1)

# ── Strategy Invariants ────────────────────────────────────────────────────────
MAGIC_NUMBER = 7771399
ORDER_COMMENT = "fvg_strategy"
COMMENT_TRANCHE_A = "fvg_2R"
COMMENT_TRANCHE_B = "fvg_runner"
ACCOUNT_RISK_USD = 10.0
MAX_OPEN_POSITIONS = 3
BASKET = [
    "NAS100.p", "BTCUSD.pi", "AU200.p", "UK100.p", "EURUSD.pi",
    "XAGUSD.pi", "SOLUSD.p", "USDJPY.pi", "USDCHF.pi"
]

JOURNAL_FILE = Path("live_trade_journal.jsonl")

# ── Helper Functions ───────────────────────────────────────────────────────────
def log_event(event_type, data):
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event_type,
        **data
    }
    with open(JOURNAL_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    print(f"[{record['timestamp'][:19]}] [{event_type}] {data}")

def calculate_atr(df, period=14):
    high, low, close = df['high'].values, df['low'].values, df['close'].values
    n = len(df)
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, hc, lc)
    return pd.Series(tr).ewm(span=period, adjust=False).mean().values

def get_fvg_positions():
    """Returns only positions belonging to fvg_strategy (isolated from other algos)."""
    positions = mt5.positions_get()
    if not positions:
        return []
    return [p for p in positions if p.magic == MAGIC_NUMBER or "fvg" in str(p.comment).lower()]

def calculate_lots(symbol, order_type, entry_price, sl_price):
    """Calculates exact lot size for $10 risk using MT5's broker profit calculator."""
    info = mt5.symbol_info(symbol)
    if not info:
        return 0.0
        
    loss_1_lot = mt5.order_calc_profit(order_type, symbol, 1.0, entry_price, sl_price)
    if loss_1_lot is None or abs(loss_1_lot) < 1e-4:
        # Fallback to manual point value calculation
        risk_points = abs(entry_price - sl_price) / info.point
        loss_1_lot = risk_points * (info.trade_tick_value / (info.trade_tick_size / info.point))
        
    unit_risk = abs(loss_1_lot)
    if unit_risk <= 0:
        return 0.0
        
    raw_lots = ACCOUNT_RISK_USD / unit_risk
    step = info.volume_step if info.volume_step > 0 else 0.01
    lots = round(round(raw_lots / step) * step, 2)
    lots = max(info.volume_min, min(lots, info.volume_max))
    return lots

def punch_order(symbol, direction, entry_price, sl_price, tp2r_price, risk_dist):
    """Places split-tranche market orders for the FVG setup."""
    info = mt5.symbol_info(symbol)
    if not info or not info.visible:
        if not mt5.symbol_select(symbol, True):
            print(f"Failed to select symbol {symbol}")
            return False
            
    is_buy = (direction == "Long")
    order_type = mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL
    
    total_lots = calculate_lots(symbol, order_type, entry_price, sl_price)
    if total_lots < info.volume_min:
        print(f"[{symbol}] Calculated lot {total_lots} below broker min {info.volume_min}")
        return False
        
    tick = mt5.symbol_info_tick(symbol)
    fill_price = tick.ask if is_buy else tick.bid
    
    # Split volume into Tranche A (2R TP) and Tranche B (Runner)
    if total_lots >= 0.02:
        vol_a = round(total_lots / 2.0, 2)
        vol_b = round(total_lots - vol_a, 2)
    else:
        vol_a = total_lots
        vol_b = 0.0
        
    filling = mt5.ORDER_FILLING_IOC if (info.filling_mode & 2) else mt5.ORDER_FILLING_FOK
    
    # Order A: 2.0R First Target
    req_a = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": vol_a,
        "type": order_type,
        "price": fill_price,
        "sl": round(sl_price, info.digits),
        "tp": round(tp2r_price, info.digits),
        "deviation": 20,
        "magic": MAGIC_NUMBER,
        "comment": COMMENT_TRANCHE_A,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": filling,
    }
    
    res_a = mt5.order_send(req_a)
    if res_a.retcode != mt5.TRADE_RETCODE_DONE:
        log_event("ORDER_REJECTED", {"symbol": symbol, "tranche": "A", "retcode": res_a.retcode, "comment": res_a.comment})
        return False
        
    log_event("ORDER_PUNCHED", {
        "symbol": symbol,
        "tranche": "A (2R TP)",
        "ticket": res_a.order,
        "direction": direction,
        "volume": vol_a,
        "price": res_a.price,
        "sl": sl_price,
        "tp": tp2r_price,
        "risk_dist": risk_dist,
        "target_risk_usd": ACCOUNT_RISK_USD * (vol_a / total_lots)
    })
    
    # Order B: Runner (Open TP, trailing stop)
    if vol_b > 0:
        req_b = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": vol_b,
            "type": order_type,
            "price": fill_price,
            "sl": round(sl_price, info.digits),
            "tp": 0.0,  # Open runner
            "deviation": 20,
            "magic": MAGIC_NUMBER,
            "comment": COMMENT_TRANCHE_B,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling,
        }
        res_b = mt5.order_send(req_b)
        if res_b.retcode == mt5.TRADE_RETCODE_DONE:
            log_event("ORDER_PUNCHED", {
                "symbol": symbol,
                "tranche": "B (Runner)",
                "ticket": res_b.order,
                "direction": direction,
                "volume": vol_b,
                "price": res_b.price,
                "sl": sl_price,
                "tp": 0.0,
                "risk_dist": risk_dist,
                "target_risk_usd": ACCOUNT_RISK_USD * (vol_b / total_lots)
            })
        else:
            log_event("ORDER_REJECTED", {"symbol": symbol, "tranche": "B", "retcode": res_b.retcode, "comment": res_b.comment})
            
    return True

def manage_open_positions():
    """Manages active FVG trades: moves to Breakeven at +1.0R and trails runners."""
    positions = get_fvg_positions()
    if not positions:
        return
        
    for p in positions:
        info = mt5.symbol_info(p.symbol)
        if not info:
            continue
            
        tick = mt5.symbol_info_tick(p.symbol)
        if not tick:
            continue
            
        is_long = (p.type == mt5.POSITION_TYPE_BUY)
        cur_price = tick.bid if is_long else tick.ask
        open_price = p.price_open
        cur_sl = p.sl
        cur_tp = p.tp
        
        # Estimate initial risk distance from SL
        if cur_sl == 0:
            continue
            
        risk_dist = abs(open_price - cur_sl)
        if risk_dist <= 0:
            continue
            
        r_gain = (cur_price - open_price) / risk_dist if is_long else (open_price - cur_price) / risk_dist
        
        # 1. Breakeven at +1.0R (with +0.1R buffer)
        be_sl = open_price + (0.1 * risk_dist) if is_long else open_price - (0.1 * risk_dist)
        be_sl = round(be_sl, info.digits)
        
        needs_be = False
        if r_gain >= 1.0:
            if is_long and cur_sl < (open_price + 0.05 * risk_dist):
                needs_be = True
            elif not is_long and cur_sl > (open_price - 0.05 * risk_dist):
                needs_be = True
                
        if needs_be:
            req = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": p.ticket,
                "symbol": p.symbol,
                "sl": be_sl,
                "tp": cur_tp
            }
            res = mt5.order_send(req)
            if res.retcode == mt5.TRADE_RETCODE_DONE:
                log_event("BREAKEVEN_TRIGGERED", {
                    "ticket": p.ticket, "symbol": p.symbol, "r_gain": round(r_gain, 2), "new_sl": be_sl
                })
                continue

        # 2. Trailing Stop on Runner (Position has tp == 0)
        if cur_tp == 0.0 and r_gain >= 1.5:
            rates = mt5.copy_rates_from_pos(p.symbol, mt5.TIMEFRAME_H1, 0, 20)
            if rates is not None and len(rates) >= 14:
                df = pd.DataFrame(rates)
                atr = calculate_atr(df, 14)[-1]
                # Chandelier / Swing trail level (k = 1.5)
                if is_long:
                    highest_close = df['close'].max()
                    trail_level = round(highest_close - 1.5 * atr, info.digits)
                    if trail_level > cur_sl and trail_level < cur_price:
                        req = {"action": mt5.TRADE_ACTION_SLTP, "position": p.ticket, "symbol": p.symbol, "sl": trail_level, "tp": 0.0}
                        res = mt5.order_send(req)
                        if res.retcode == mt5.TRADE_RETCODE_DONE:
                            log_event("TRAIL_UPDATED", {"ticket": p.ticket, "symbol": p.symbol, "new_sl": trail_level})
                else:
                    lowest_close = df['close'].min()
                    trail_level = round(lowest_close + 1.5 * atr, info.digits)
                    if trail_level < cur_sl and trail_level > cur_price:
                        req = {"action": mt5.TRADE_ACTION_SLTP, "position": p.ticket, "symbol": p.symbol, "sl": trail_level, "tp": 0.0}
                        res = mt5.order_send(req)
                        if res.retcode == mt5.TRADE_RETCODE_DONE:
                            log_event("TRAIL_UPDATED", {"ticket": p.ticket, "symbol": p.symbol, "new_sl": trail_level})

def scan_and_execute_setups():
    """Scans all 9 basket assets for fresh mechanical setups on completed 1H bars."""
    open_positions = get_fvg_positions()
    active_symbols = {p.symbol for p in open_positions}
    
    # Heat budget: cap by number of distinct active assets (not raw ticket count)
    if len(active_symbols) >= MAX_OPEN_POSITIONS:
        return
        
    for symbol in BASKET:
        if symbol in active_symbols:
            continue
            
        if len(active_symbols) >= MAX_OPEN_POSITIONS:
            break
            
        rates_1h = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 60)
        rates_4h = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H4, 0, 40)
        if rates_1h is None or rates_4h is None or len(rates_1h) < 40:
            continue
            
        df = pd.DataFrame(rates_1h)
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        
        df_4h = pd.DataFrame(rates_4h)
        df_4h['time'] = pd.to_datetime(df_4h['time'], unit='s', utc=True)
        df_4h['ema20'] = df_4h['close'].ewm(span=20, adjust=False).mean()
        df_4h['ema50'] = df_4h['close'].ewm(span=50, adjust=False).mean()
        
        htf_clean = df_4h[['time', 'ema20', 'ema50']].dropna().copy()
        htf_clean['time'] = (htf_clean['time'] + pd.Timedelta(hours=4)).astype(df['time'].dtype)
        
        merged = pd.merge_asof(df, htf_clean, on='time', direction='backward')
        merged['atr14'] = calculate_atr(merged, 14)
        
        vol = merged['tick_volume']
        vol_sma = vol.rolling(20).mean().replace(0, 1.0)
        merged['rvol'] = vol / vol_sma
        
        body = (merged['close'] - merged['open']).abs()
        rng = (merged['high'] - merged['low']).replace(0, 1e-6)
        merged['body_ratio'] = body / rng
        merged['is_bull'] = merged['close'] > merged['open']
        merged['is_bear'] = merged['close'] < merged['open']
        
        merged['swing_high_10'] = merged['high'].shift(2).rolling(10).max()
        merged['swing_low_10'] = merged['low'].shift(2).rolling(10).min()
        
        n = len(merged)
        # Check if an FVG was formed in recent bars and tapped by the latest completed bar
        latest_idx = n - 2  # n-1 is forming bar, n-2 is latest completed 1H bar
        latest_bar = merged.iloc[latest_idx]
        
        # Session filter: London & NY liquid hours (08:00 to 17:00 UTC)
        bar_hour = latest_bar['time'].hour
        if not (8 <= bar_hour <= 17):
            continue
            
        # Scan backward up to 10 bars for an active FVG formation
        for k in range(latest_idx - 1, max(latest_idx - 10, 5), -1):
            fvg_long = (merged['high'].iloc[k-2] < merged['low'].iloc[k])
            fvg_short = (merged['low'].iloc[k-2] > merged['high'].iloc[k])
            
            if not (fvg_long or fvg_short):
                continue
                
            direction = "Long" if fvg_long else "Short"
            midpoint = (merged['high'].iloc[k-2] + merged['low'].iloc[k]) / 2.0 if fvg_long else (merged['low'].iloc[k-2] + merged['high'].iloc[k]) / 2.0
            structural_inval = merged['high'].iloc[k-2] if fvg_long else merged['low'].iloc[k-2]
            
            # Check 4H trend filter on the completed bar
            h_ema20 = latest_bar['ema20']
            h_ema50 = latest_bar['ema50']
            if pd.notna(h_ema20) and pd.notna(h_ema50):
                if direction == "Long" and (latest_bar['close'] < h_ema50 or h_ema20 < h_ema50 * 0.998):
                    continue
                elif direction == "Short" and (latest_bar['close'] > h_ema50 or h_ema20 > h_ema50 * 1.002):
                    continue
                    
            # Check if latest bar tapped the midpoint
            if latest_bar['low'] <= midpoint <= latest_bar['high']:
                # Prior displacement check at bar k-1, k-2
                rvol_k = merged['rvol'].iloc[k]
                if rvol_k < 1.15:
                    continue
                    
                atr_val = latest_bar['atr14']
                raw_risk = abs(midpoint - structural_inval)
                risk_dist = max(raw_risk, 1.2 * atr_val)
                sl_price = midpoint - risk_dist if direction == "Long" else midpoint + risk_dist
                tp2r_price = midpoint + 2.0 * risk_dist if direction == "Long" else midpoint - 2.0 * risk_dist
                
                print(f"\n⚡ [MECHANICAL TRIGGER DETECTED] {symbol} {direction} Tap at {latest_bar['time']}")
                punch_order(symbol, direction, midpoint, sl_price, tp2r_price, risk_dist)
                break

def run_loop(single_pass=False):
    if not mt5.initialize():
        print(f"Failed to connect to MT5: {mt5.last_error()}")
        return
        
    acc = mt5.account_info()
    print("=" * 80)
    print(f"FVG LIVE EXECUTION DAEMON ONLINE | Account: {acc.login} ({acc.server})")
    print(f"Capital: ${acc.balance:.2f} | Risk Budget: ${ACCOUNT_RISK_USD:.2f}/trade | Comment: '{ORDER_COMMENT}'")
    print(f"Basket: {', '.join(BASKET)}")
    print("=" * 80)
    
    try:
        while True:
            # 1. Manage active FVG trades
            manage_open_positions()
            
            # 2. Scan for fresh mechanical entries
            scan_and_execute_setups()
            
            active = get_fvg_positions()
            active_symbols = {p.symbol for p in active}
            now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            print(f"[{now_str}] Heartbeat: Active FVG assets = {len(active_symbols)} / {MAX_OPEN_POSITIONS} (open tickets = {len(active)})", end="\r")
            
            if single_pass:
                break
            time.sleep(30)
            
    except KeyboardInterrupt:
        print("\nShutdown requested by user.")
    finally:
        mt5.shutdown()
        print("\nMT5 connection closed gracefully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FVG Strategy MT5 Live Automated Execution Daemon")
    parser.add_argument("--once", action="store_true", help="Run a single scan & manage cycle then exit")
    args = parser.parse_args()
    run_loop(single_pass=args.once)
