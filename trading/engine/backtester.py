import pandas as pd, numpy as np, matplotlib.pyplot as plt
from joblib import Parallel, delayed
from datetime import datetime
import sys
from pathlib import Path

import sys
# sys.path.append("../src")
sys.path.append(str(Path.cwd().parent))

# from indicators.indicators import load_yahoo, build_indicators

import vars

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_colwidth', 200)

color_back = "#0d1117"

path_repo = vars.path_repo
path_data = vars.path_data



# Parameters
risk_per_trade          = 0.01
fee_rate                = 0.0007
slippage_pct            = 0.0005
atr_period              = 14
# min_holding_period    = 0
leverage                = 100     # 10 = 1:10
# max_loss_streak_limit = 5
# cooldown_after_losses = 10
# ST y TP
# atr_sl                = 2          # Value for SL. Scalping: 0.8-1.5 | Intradia: 1.5-2.5 | Swing: 2-3.5 | Largo: 3-5
# multiplier_tp         = 1.5        # Value for TP -> atr_tp = atr_sl x multiplier_tp. 
# use_r_tp              = False      # Si se activa, no se usara ATR si no R para fijar sl, tp. R: distancia a sl, para calcular tp. 
# factor_r_tp           = 2          # Factor para fijar TP = entry price + factor_r_tp * R. R=price - sl. Conservador: 1-1.5 | Balanceado: 2 | Tendencial | 3-5
# # Trailing
# use_trailing_stop     = False
# use_breakeven         = use_trailing_stop  # Siempre usar
# trailing_atr_mult     = 1                  # default=2. Value por trailing stop
# trailing_activation_r = 2.5                # default=2. Solo activar trailing después de 2R de ganancia, el break-even se activa antes, al llegar a 1R. Evitar que sea igual que breakeven_trigger_r
# breakeven_trigger_r   = 3                  # default=1. break-even al llegar a 1R si valor = 0 -> stop loss se posicione en precio de entrada. Cero ganancias


# ── Gestión de riesgo de cartera ─────────────────────────────
# max_dd_daily_pct       = 0.05        # parar si el día pierde >5%
# cooldown_bars          = 3           # barras de espera tras un SL
# max_trades_per_day     = 6



# V1
# def engine(
#     df, 
#     name_signal,
#     symbol=None,
#     initial_capital=1000,
#     risk_per_trade=0.01,
#     fee_rate=0.0007,
#     slippage_pct=0.0005,
#     leverage=100,
#     atr_sl=2,
#     multiplier_tp=1.5,
#     use_r_tp=False,
#     factor_r_tp=2,
#     use_trailing_stop=False,
#     use_breakeven=False,
#     trailing_atr_mult=1,
#     trailing_activation_r=2.5,
#     breakeven_trigger_r=3,
#     same_bar_policy="SL_FIRST",
#     close_on_reverse=True,
#     annualization_factor=None
#     ):


#     resume = []
#     state = {}
    
#     df = df.copy()

#     patrimony = initial_capital 
#     position_opened = False
#     candle_trade = []
#     state = []
#     idx = 0
#     bars_held = np.nan
#     cooldown = 0
#     n = len(df)

#     df["signal"] = df[name_signal]

#     for i in range(n):

#         bar        = df.iloc[i]
#         prev_bar   = df.iloc[i - 1]
#         signal     = bar["signal"]
#         high       = bar["High"]
#         low        = bar["Low"]
#         atr        = bar["ATR"]
#         price_var  = "Close"
#         price      = bar[price_var]

#         # COOLDOWN CONTROL
#         if cooldown > 0:
#             cooldown -= 1
#             signal = 0  # bloquear entradas


#         #-------------- 1. ENTRY
#         if not position_opened and signal != 0:

#             if signal == 1: 
#                 entry_price = price * (1 + slippage_pct)
#             else:
#                 entry_price = price * (1 - slippage_pct)

#             sl_distance = atr_sl * atr

#             if signal == 1:
#                 stop_loss = entry_price - sl_distance
#             else:
#                 stop_loss = entry_price + sl_distance

#             tp_distance = abs(entry_price - stop_loss) * multiplier_tp

#             if signal == 1:
#                 take_profit = entry_price + tp_distance
#             else:
#                 take_profit = entry_price - tp_distance

#             amount = patrimony * risk_per_trade
#             patrimony = patrimony - amount

#             risk_unit = abs(entry_price - stop_loss)
#             size = amount / risk_unit
#             position_value = size * entry_price
#             max_position_value = amount * leverage      
#             if position_value > max_position_value:
#                 size = max_position_value / entry_price

#             # Niveles auxiliares (se calculan UNA vez en ENTRY)
#             r_unit        = abs(entry_price - stop_loss)
#             be_price      = entry_price + breakeven_trigger_r * r_unit * signal

#             bars_held = 0

#             temp = {
#                 "id": [idx],
#                 "time": [bar.name],
#                 "signal": [signal],
#                 "signal_trade": [signal],
#                 "close": [entry_price],
#                 "high": [high],
#                 "low": [low],
#                 "entry_price": [entry_price],
#                 "sl": [stop_loss],
#                 "sl_inicial":     [stop_loss],   # guardamos el SL original para calcular R
#                 "tp": [take_profit],
#                 "size": size,
#                 "bars_held": [bars_held],
#                 "entry_amount":[amount],
#                 "be_price":       [be_price],
#                 "breakeven_hit":  [False],
#                 "partial_hit":    [False],
#                 "trailing_active":[False],
#                 "price_prev":[prev_bar[price_var]],
#                 "diff":[bar[price_var] - prev_bar[price_var]],
#                 "delta":[(bar[price_var] - prev_bar[price_var])/prev_bar[price_var]],
#                 "diff_open": [np.nan],
#                 "delta_open":[np.nan],  
#                 "patrimony": [patrimony],
#                 "state": ["ENTRY"],
#                 "RSI":[bar["RSI"]],
#                 "ADX":[bar["ADX"]],
#                 "ADX+DI":[bar["ADX+DI"]],
#                 "ADX-DI":[bar["ADX-DI"]],
#                 "ATR":[bar["ATR"]]
#             }
#             temp = pd.DataFrame(temp)
#             state.append(temp)
#             position_opened = True
#             candle_trade = temp.copy()
#             pass


#         #-------------- 2. MANAGE TRADE
#         if position_opened:
#             _exit        = False
#             exitintrail  = False
#             reason       = np.nan

#             _close = bar["Close"]
#             _high = bar["High"]
#             _low = bar["Low"]
#             _sl = candle_trade["sl"][0]
#             _tp = candle_trade["tp"][0]
#             _entry_amount = candle_trade["entry_amount"][0]
#             _entry_price = candle_trade["entry_price"][0]
#             _size = candle_trade["size"][0]        
#             signal_trade    = candle_trade["signal_trade"][0]
#             _sl_inicial     = candle_trade["sl_inicial"][0]
#             _be_price       = candle_trade["be_price"][0]
#             _breakeven_hit  = candle_trade["breakeven_hit"][0]
#             _trailing_active= candle_trade["trailing_active"][0]
#             _close          = bar["Close"]


#             # REVERSION OF SIGNAL CONDITION
#             if signal != 0 and signal != signal_trade:
#                 _exit = True
#                 reason = "REV"


#             temp = candle_trade.copy()

#             temp["diff_open"] = (bar[price_var] - candle_trade[price_var.lower()])[0]
#             temp["delta_open"] = ((bar[price_var] - candle_trade[price_var.lower()]) /
#                                 candle_trade[price_var.lower()])[0]
            
#             # P&L actual de la posición (sobre valor real apalancado)
#             actual_pos_value = _size * _entry_price          
#             delta_open       = (_close - _entry_price) / _entry_price
#             pnl              = actual_pos_value * delta_open * signal_trade

#             # BREAK-EVEN
#             if use_breakeven and not _breakeven_hit:
#                 be_hit = (signal_trade ==  1 and _close >= _be_price) or \
#                          (signal_trade == -1 and _close <= _be_price)
#                 if be_hit:
#                     _sl = _entry_price
#                     candle_trade["sl"]           = _sl
#                     candle_trade["breakeven_hit"]= True
#                     _breakeven_hit               = True

#             # TRAILING STOP (solo se activa después de trailing_activation_r)
#             if use_trailing_stop:
#                 r_unit       = abs(_entry_price - _sl_inicial)
#                 ganancia_en_r = abs(_close - _entry_price) / r_unit if r_unit != 0 else 0

#                 if ganancia_en_r >= trailing_activation_r:
#                     candle_trade["trailing_active"] = True
#                     _trailing_active = True

#                 if _trailing_active:
#                     trail_dist = trailing_atr_mult * atr

#                     if signal_trade == 1:                     
#                         new_trail = high - trail_dist
#                         if new_trail > _sl:
#                             _sl = new_trail
#                             candle_trade["sl"] = _sl
#                             exitintrail = True
#                     else:
#                         new_trail = low + trail_dist
#                         if new_trail < _sl:
#                             _sl = new_trail
#                             candle_trade["sl"] = _sl
#                             exitintrail = True


#             if signal_trade == 1:
#                 sl_hit = low <= _sl
#                 tp_hit = high >= _tp
#                 if sl_hit:
#                     _exit = True
#                     reason = "SL"
#                     reason = "TS" if reason == "SL" and exitintrail else reason
#                 elif tp_hit:
#                     _exit = True
#                     reason = "TP"


#             if signal_trade == -1:
#                 sl_hit = high >= _sl
#                 tp_hit = low <= _tp
#                 if sl_hit:
#                     _exit = True
#                     reason = "SL"
#                     reason = "TS" if reason == "SL" and exitintrail else reason
#                 elif tp_hit:
#                     _exit = True
#                     reason = "TP"


#             # EXIT TRADE
#             if _exit and bars_held >= 1:
#                 exit_price = _tp if reason == "TP" else _sl

#                 if reason == "REV":
#                     exit_price = price  # cerrar al precio actual

#                 if signal_trade == 1:
#                     _return = (exit_price - _entry_price) / _entry_price
#                 else:
#                     _return = (_entry_price - exit_price) / _entry_price
                
#                 actual_position_value = _size * _entry_price          
#                 gross_profit = _return * actual_position_value          
#                 fees         = actual_position_value * fee_rate * 2       
#                 profit_f     = gross_profit - fees
#                 patrimony    += _entry_amount + profit_f


#                 temp = {
#                     "id": [idx],
#                     "time": [bar.name],
#                     "close": [price],
#                     "high": [high],
#                     "low": [low],
#                     "signal_trade":[0],
#                     "price_prev":prev_bar[price_var],
#                     "diff":bar[price_var] - prev_bar[price_var],
#                     "delta":[(bar[price_var] - prev_bar[price_var])/prev_bar[price_var]],
#                     "patrimony": [patrimony],
#                     "state":"EXIT",
#                     "reason":[reason],
#                     "pnl":[_entry_amount + profit_f],
#                     "entry_amount":[_entry_amount],
#                     "profit_ex":[profit_f],
#                     "return_ex":[_return],
#                     "bars_held_ex":[bars_held],
#                     "RSI":[bar["RSI"]],
#                     "ADX":[bar["ADX"]],
#                     "ADX+DI":[bar["ADX+DI"]],
#                     "ADX-DI":[bar["ADX-DI"]],
#                     "ATR":[bar["ATR"]],
#                     "breakeven_hit":[_breakeven_hit],
#                     "be_price":[_be_price],
#                     "sl":[_sl],
#                     "tp":[_tp],
                    
#                 }
#                 temp = pd.DataFrame(temp)
#                 state.append(temp)

#                 _exit = False
#                 position_opened = False
#                 candle_trade = []

#                 pass

#             else:
#                 temp = candle_trade.copy()
#                 temp["id"] = idx
#                 temp["signal"] = np.nan
#                 temp["close"] = price
#                 temp["high"] = high
#                 temp["low"] = low
#                 temp["time"] = bar.name
#                 temp["bars_held"] = [bars_held]
#                 temp["price_prev"] = prev_bar[price_var]
#                 temp["diff"] = bar[price_var] - prev_bar[price_var]
#                 temp["delta"] = (bar[price_var] - prev_bar[price_var]) / prev_bar[price_var]
#                 temp["diff_open"] = (bar[price_var] - candle_trade[price_var.lower()])[0]
#                 temp["delta_open"] = ((bar[price_var] - candle_trade[price_var.lower()]) / candle_trade[price_var.lower()])[0]
#                 temp["patrimony"] = patrimony
#                 temp["state"] = "HOLD"
#                 temp["pnl"] = pnl
#                 temp["RSI"] = bar["RSI"]
#                 temp["ADX"] = bar["ADX"]
#                 temp["ADX+DI"] = bar["ADX+DI"]
#                 temp["ADX-DI"] = bar["ADX-DI"]
#                 temp["ATR"] = bar["ATR"]
#                 temp["breakeven_hit"] = _breakeven_hit
#                 temp["be_price"] = _be_price

#                 state.append(temp)

#                 pass


#         if not position_opened and signal == 0:
#             temp = {
#                 "id": [idx],
#                 "time": [bar.name],
#                 "signal_trade": [signal],
#                 "close": [price],
#                 "high": [high],
#                 "low": [low],
#                 "price_prev":prev_bar[price_var],
#                 "diff":bar[price_var] - prev_bar[price_var],
#                 "delta":[(bar[price_var] - prev_bar[price_var])/prev_bar[price_var]],
#                 "patrimony": [patrimony],
#                 "state":"NO SIGNAL",
#                 "RSI":[bar["RSI"]],
#                 "ADX":[bar["ADX"]],
#                 "ADX+DI":[bar["ADX+DI"]],
#                 "ADX-DI":[bar["ADX-DI"]],
#                 "ATR":[bar["ATR"]]
#             }
#             temp = pd.DataFrame(temp)
#             state.append(temp)

#             pass

#         bars_held += 1
#         idx += 1

#     state = pd.concat(state)
#     state.drop_duplicates("time", inplace = True) # Ajustar

#     # Devolviendo capital invertido a la ultima vela si finalizando se tiene una operacion abierta y esta no cerró
#     last = state.iloc[-1]
#     indeter_signal = False
#     if last["state"] == "HOLD":
#         indeter_signal = True
#         state["patrimony"] = np.where(state["id"] == last["id"], state["patrimony"] + state["entry_amount"], state["patrimony"])

#     if "profit_ex" not in state:
#         state["profit_ex"] = np.nan
#     if "return_ex" not in state:
#         state["return_ex"] = np.nan

#     state["profit"] = state["pnl"] - state["entry_amount"]
#     state["return%"] = state["profit"] / state["entry_amount"]
#     state["profit"] = np.where(state["state"] == "EXIT", state["profit_ex"], state["profit"])
#     state["return%"] = np.where(state["state"] == "EXIT", state["return_ex"], state["return%"])
#     state["peak"] = state["patrimony"].cummax()
#     state["drawdown"] = (state["patrimony"] - state["peak"]) / state["peak"]
#     exits = state[state["state"] == "EXIT"].copy()

#     returns = state.loc[state["state"] == "EXIT", "return%"].dropna()
#     returns_pct = exits["return%"]

#     if len(returns) > 1 and returns.std() != 0:
#         sharpe = (returns.mean() / returns.std()) * np.sqrt(len(returns))
#     else:
#         sharpe = np.nan

#     max_dd = state["drawdown"].min()

#     capital_f = state["patrimony"].iloc[-1]
#     pnl_f = capital_f - capital
#     return_f = ((capital_f / capital) - 1)*100
#     exits["return%"] = exits["return%"]*100
#     n_trades = exits.shape[0] if not indeter_signal else exits.shape[0] - 1
#     win_rate = (exits[(exits["profit"] > 0)].shape[0] / n_trades)*100 if n_trades != 0 else 0

#     days = (state["time"].max() - state["time"].min()) / pd.Timedelta(days=1)
#     downside = exits["return%"]
#     downside = downside[downside < 0]
#     sortino = (exits["return%"].mean() / downside.std()) * np.sqrt(24*365) 
#     gains = exits[exits["profit"] > 0]
#     loss = exits[exits["profit"] < 0]
#     exits["time_t1"] = exits["time"].shift(1)
#     exits["diff_time"] = (exits["time"] - exits["time_t1"]) / pd.Timedelta(days=1)
#     cagr   = ((capital_f / capital) ** (365 / max(days, 1)) - 1) * 100
#     calmar = cagr / abs(max_dd * 100) if max_dd != 0 else np.nan
#     win_rate_dec = len(gains) / n_trades if n_trades > 0 else 0
#     max_loss_streak, cur_streak = 0, 0
#     for s in exits["profit"]:
#         cur_streak      = cur_streak + 1 if s < 0 else 0
#         max_loss_streak = max(max_loss_streak, cur_streak)
#     max_gain_streak, cur_streak = 0, 0
#     for s in exits["profit"]:
#         cur_streak      = cur_streak + 1 if s > 0 else 0
#         max_gain_streak = max(max_gain_streak, cur_streak)
#     hold_bars  = state[state["state"] == "HOLD"].shape[0]
#     exposure   = hold_bars / len(state) * 100
#     var_95  = np.percentile(returns_pct, 5)  if len(returns_pct) > 0 else np.nan
#     cvar_95 = returns_pct[returns_pct <= var_95].mean() if len(returns_pct) > 0 else np.nan
#     recovery_factor = pnl_f / abs(max_dd * capital) if max_dd != 0 else np.nan
#     n_sl = exits[exits["reason"] == "SL"].shape[0]
#     n_tp = exits[exits["reason"] == "TP"].shape[0]
#     n_ts = exits[exits["reason"] == "TS"].shape[0]

#     print("─────────────── Symbol ──────────────")
#     print(f"               {symb}")

#     print("\n───────── Time and Candles ──────────")
#     print("Candles:", f"{len(df)} | Days: {int(days)}") # {window}
#     print("Trades:           ", n_trades, f"| Wins: {len(gains)} | Loss: {len(loss)}")
#     print("Trades/day:       ", f"{(n_trades/days):.1f}")
#     print("Time/trade:       ", f"Mean: {exits["bars_held_ex"].mean():.0f} | Min: {exits["bars_held_ex"].min():.0f} | Max: {exits["bars_held_ex"].max():.0f}")
#     print("Time inter trades:", f"Mean: {exits["diff_time"].mean():.0f} | Min: {exits["diff_time"].min():.0f} | Max: {exits["diff_time"].max():.0f}")

#     print("\n─────────── Final Results ───────────")
#     print("Win rate:         ", f"{win_rate:.0f}%")
#     print("Capital initial:  ", f"${capital}")
#     print("Capital final:    ", f"${capital_f:.1f}")
#     print("PnL:              ", f"${pnl_f:.1f}")
#     print("Return:           ", f"{(return_f):.1f}%")
#     print("Max drawdown:     ", f"{max_dd:.1%}\n")

#     print("Sharpe:           ", f"{sharpe:.1f}")
#     print("Sortino:          ", f"{sortino:.0f}")
#     print("Calmar:           ", f"{calmar:.0f}")
#     print("CAGR:             ", f"{cagr:.0f}\n")

#     print("Total gains:      ", f"${gains["profit"].sum():.0f}")
#     print("Total loss:       ", f"${loss["profit"].sum():.0f}")
#     print(" ")
#     print("Return/trade:     ", f"Mean: {exits["return%"].mean():.0f}% | Min: {exits["return%"].min():.0f}% | Max: {exits["return%"].max():.0f}%")
#     print("Profit/trade:     ", f"Mean: ${exits["profit"].mean():.1f} | Min: ${exits["profit"].min():.1f} | Max: ${exits["profit"].max():.1f}")

#     print("\n─────────── Metrics Risk ───────────") # Risk:Reward Ratio
#     exits["return_on_margin"] = exits["profit"] / exits["entry_amount"] * 100 # Retorno promedio real sobre margen (no sobre precio)
#     avg_win = exits[exits['profit']>0]['return%'].mean()
#     avg_loss = exits[exits['profit']<0]['return%'].mean()
#     avg_win_margin = exits[exits['profit']>0]['return_on_margin'].mean()
#     avg_loss_margin = exits[exits['profit']<0]['return_on_margin'].mean()
#     loss_rate = 1 - win_rate   
#     expectancy = (win_rate_dec * avg_win) - (loss_rate * avg_loss)
#     prom_fee = exits['entry_amount'].mean() * fee_rate * 2 * leverage
#     print(f"Prom Position Value:${exits['entry_amount'].mean() * leverage:.0f} (with leverage)")
#     print(f"Prom Position Value:${exits['entry_amount'].mean():.0f} (without leverage)")

#     print(f"Expectancy:         {expectancy:.1f}")
#     print(f"Avg win (price):    {avg_win:.1f}%")
#     print(f"Avg loss (price):   {avg_loss:.1f}%\n")

#     print(f"Ratio W/L:          {abs(avg_win / avg_loss):.1f}x | Gain prom = {abs(avg_win / avg_loss):.1f} x Loss") # en promedio cada ganancia es 2.6 veces más grande que cada pérdida.
#     print(f"Recovery factor:    {recovery_factor:.2f}")
#     print(f"VaR 95%:            {var_95:.1f}%")
#     print(f"CVaR 95%:           {cvar_95:.1f}%\n")

#     print(f"Max racha loss:     {max_loss_streak:.0f}")
#     print(f"Max racha gain:     {max_gain_streak:.0f}\n")

#     clos_tp = n_tp/n_trades*100 if n_trades != 0 else 0
#     clos_sl = n_sl/n_trades*100 if n_trades != 0 else 0
#     clos_ts = n_ts/n_trades*100 if n_trades != 0 else 0

#     print(f"Closed x TP:        {n_tp} ({clos_tp:.0f}%)")
#     print(f"Closed x SL:        {n_sl} ({clos_sl:.0f}%)")
#     print(f"Closed x TS:        {n_ts} ({clos_ts:.0f}%)\n")

#     print(f"Avg win (margin):   {avg_win_margin:.0f}%")
#     print(f"Avg loss (margin):  {avg_loss_margin:.0f}%\n")

#     print(f"Fees estim/trade:   ${prom_fee:.1f}")
#     print(f"Total fees estim:   ${prom_fee * n_trades:.0f}")
#     print(f"Fees as %avg_win:   {prom_fee / exits[exits['profit']>0]['profit'].mean() * 100:.0f}%\n")

#     resume = pd.DataFrame({"symbol":[symb], "signal":[name_signal], "capital":[capital_f], "win_rate":[win_rate], "pnl":[pnl_f], "return":[return_f], 
#                         "trades":[n_trades], "tradesxday":[n_trades/days], "drawdown":[max_dd], "sharpe":[max_dd], "sharpe":[sharpe], "ratio_wl":[abs(avg_win / avg_loss)],
#                         "total_gains":[gains["profit"].sum()], "total_loss":[loss["profit"].sum()], "mean_timextrade":[exits["bars_held_ex"].mean()],
#                         "mean_profit":[exits["profit"].mean()], "min_profit":[exits["profit"].min()],"max_profit":[exits["profit"].mean()],
#                         "mean_position":[exits['entry_amount'].mean() * leverage], "expectancy":[expectancy], "recov_factor":[recovery_factor], "max_rach_loss":[max_loss_streak],
#                         "max_rach_gain":[max_gain_streak], "close_tp":[clos_tp], "close_sl":[clos_tp],"close_ts":[clos_tp],
#                         "mean_fees":[prom_fee], "total_fees":[prom_fee * n_trades], "param.atr_sl":[atr_sl], "param.multiplier_tp":[multiplier_tp], 
#                         "param.trailing_atr_mult":[trailing_atr_mult], "param.use_trailing_stop":[use_trailing_stop], "param.use_breakeven":[use_breakeven],
#                         "param.trailing_activation_r":[trailing_activation_r], "param.breakeven_trigger_r":[breakeven_trigger_r],
#                         "param.use_r_tp":[use_r_tp], "param.factor_r_tp":[factor_r_tp],
#                         })

#     state.drop(columns = ["diff_open", "price_prev", "diff", "delta", "profit_ex", "return_ex", "peak", "bars_held_ex", "delta_open",# "be_price", "breakeven_hit",
#                           "trailing_active", "sl_inicial"], inplace =True)
#     return resume, state
#     print("="*120)
#     print("="*120, "\n")



# V2
def engine(
    df,
    name_signal,
    symbol=None,
    initial_capital=1000,
    risk_per_trade=0.01,
    fee_rate=0.0007,
    slippage_pct=0.0005,
    leverage=100,
    atr_sl=2,
    multiplier_tp=1.5, # Es X veces atr_sl
    use_r_tp=False,
    factor_r_tp=2,
    use_trailing_stop=False,
    use_breakeven=False, # Se activa solo si está activo use_trailing_stop
    trailing_atr_mult=1,
    trailing_activation_r=2.5,
    breakeven_trigger_r=3,
    same_bar_policy="SL_FIRST",
    close_on_reverse=True,
    annualization_factor=None
):

    # ============================================================
    # VALIDACIONES
    # ============================================================

    required_cols = [
        "Open",
        "High",
        "Low",
        "Close",
        "ATR",
        name_signal
    ]

    missing = [x for x in required_cols if x not in df.columns]

    if missing:
        raise ValueError(
            f"Faltan columnas necesarias para el backtest: {missing}"
        )

    if same_bar_policy not in ["SL_FIRST", "TP_FIRST"]:
        raise ValueError(
            "same_bar_policy debe ser 'SL_FIRST' o 'TP_FIRST'"
        )

    df = df.copy()
    df = df.sort_index()
    df = df.reset_index()

    if "Datetime" not in df.columns:

        if "Date" in df.columns:
            df.rename(columns={"Date": "Datetime"}, inplace=True)

        else:
            # Conservamos el índice original como tiempo.
            df.rename(columns={df.columns[0]: "Datetime"}, inplace=True)

    df["signal"] = (
        pd.to_numeric(df[name_signal], errors="coerce")
        .fillna(0)
        .clip(-1, 1)
    )

    df["ATR"] = pd.to_numeric(
        df["ATR"],
        errors="coerce"
    )

    # ============================================================
    # VARIABLES DEL MOTOR
    # ============================================================

    cash = float(initial_capital)
    position = None
    trades = []
    equity_records = []
    pending_signal = 0
    trade_id = 0
    cooldown = 0

    # ============================================================
    # LOOP PRINCIPAL
    # ============================================================

    for i in range(len(df)):

        bar = df.iloc[i]

        dt = bar["Datetime"]

        open_price = float(bar["Open"])
        high = float(bar["High"])
        low = float(bar["Low"])
        close = float(bar["Close"])

        atr = bar["ATR"]

        if pd.isna(atr) or atr <= 0:

            equity = cash

            if position is not None:

                if position["side"] == 1:
                    unrealized = (
                        close - position["entry_price"]
                    ) * position["size"]

                else:
                    unrealized = (
                        position["entry_price"] - close
                    ) * position["size"]

                equity += (
                    position["margin"]
                    + unrealized
                )

            equity_records.append({
                "time": dt,
                "equity": equity,
                "cash": cash,
                "position": 1 if position else 0
            })

            continue

        # ========================================================
        # 1. EJECUTAR SEÑAL DE LA VELA ANTERIOR
        # ========================================================

        if position is None and pending_signal != 0:

            signal = pending_signal

            raw_entry = open_price

            # Slippage desfavorable.
            if signal == 1:
                entry_price = raw_entry * (1 + slippage_pct)

            else:
                entry_price = raw_entry * (1 - slippage_pct)

            # ----------------------------------------------------
            # STOP LOSS
            # ----------------------------------------------------

            sl_distance = atr_sl * atr

            if signal == 1:
                stop_loss = entry_price - sl_distance

            else:
                stop_loss = entry_price + sl_distance

            # ----------------------------------------------------
            # TAKE PROFIT
            # ----------------------------------------------------

            risk_unit = abs(
                entry_price - stop_loss
            )

            if use_r_tp:

                # Si use_r_tp=True, TP debe depender de R
                tp_distance = risk_unit * factor_r_tp

            else:

                tp_distance = (risk_unit * multiplier_tp)

            if signal == 1:
                take_profit = (entry_price + tp_distance)

            else:
                take_profit = (entry_price - tp_distance)

            # ----------------------------------------------------
            # POSITION SIZING
            # ----------------------------------------------------

            risk_amount = (cash * risk_per_trade)

            size_by_risk = (risk_amount / risk_unit)

            max_position_value = (cash * leverage)

            size_by_leverage = (max_position_value / entry_price)

            size = min(size_by_risk, size_by_leverage)

            position_value = (size * entry_price)

            margin = (position_value / leverage)

            # ----------------------------------------------------
            # COSTO DE ENTRADA
            # ----------------------------------------------------

            entry_fee = (position_value * fee_rate)

            cash -= (margin + entry_fee)

            # ----------------------------------------------------
            # BREAK EVEN
            # ----------------------------------------------------

            be_price = None

            if use_breakeven:

                be_price = (entry_price + signal * breakeven_trigger_r * risk_unit)

            position = {
                "trade_id": trade_id,
                "side": signal,
                "entry_time": dt,
                "entry_price": entry_price,
                "size": size,
                "margin": margin,
                "entry_fee": entry_fee,
                "risk_unit": risk_unit,
                "sl_initial": stop_loss,
                "sl": stop_loss,
                "tp": take_profit,
                "be_price": be_price,
                "breakeven_hit": False,
                "trailing_active": False,
                "bars_held": 0,
                "entry_signal": signal
            }

            trade_id += 1

            # IMPORTANTE:
            # La señal ya fue consumida.
            pending_signal = 0

        # ========================================================
        # 2. GESTIONAR POSICIÓN
        # ========================================================

        if position is not None:

            position["bars_held"] += 1

            side = position["side"]

            entry_price = position["entry_price"]

            sl = position["sl"]

            tp = position["tp"]

            risk_unit = position["risk_unit"]

            exit_reason = None

            exit_price = None

            # ====================================================
            # REVERSIÓN
            # ====================================================

            current_signal = int(np.sign(bar["signal"]))

            if (close_on_reverse and current_signal != 0 and current_signal != side):

                # La reversión se ejecuta al precio disponible,
                exit_price = close

                if side == 1:

                    exit_price *= (1 - slippage_pct)

                else:

                    exit_price *= (1 + slippage_pct)

                exit_reason = "REV"

            # ====================================================
            # BREAK EVEN
            # ====================================================

            if (exit_reason is None and use_breakeven and not position["breakeven_hit"]):

                be_price = position["be_price"]

                be_triggered = (side == 1 and high >= be_price) or (side == -1 and low <= be_price)

                if be_triggered:

                    # El breakeven se activa para la siguiente evaluación, no se utiliza para volver a evaluar retrospectivamente la misma vela.
                    position["breakeven_hit"] = True

            # ====================================================
            # TRAILING
            # ====================================================

            if (
                exit_reason is None
                and use_trailing_stop
            ):

                favorable_move = (

                    high - entry_price
                    if side == 1
                    else
                    entry_price - low
                )

                current_r = (
                    favorable_move / risk_unit
                )

                if current_r >= trailing_activation_r:

                    position[
                        "trailing_active"
                    ] = True

            # ====================================================
            # SL / TP
            # ====================================================

            if exit_reason is None:

                current_sl = position["sl"]


                sl_hit = (

                    low <= current_sl
                    if side == 1
                    else
                    high >= current_sl
                )

                tp_hit = (

                    high >= tp
                    if side == 1
                    else
                    low <= tp
                )

                if sl_hit and tp_hit:

                    if same_bar_policy == "SL_FIRST":

                        exit_price = current_sl
                        exit_reason = "SL"

                    else:

                        exit_price = tp
                        exit_reason = "TP"

                elif sl_hit:

                    exit_price = current_sl
                    exit_reason = "SL"

                elif tp_hit:

                    exit_price = tp
                    exit_reason = "TP"

            # ====================================================
            # EJECUCIÓN DEL EXIT
            # ====================================================

            if exit_reason is not None:

                # ------------------------------------------------
                # GAP
                # ------------------------------------------------

                if exit_reason == "SL":

                    if side == 1:

                        exit_price = min(
                            exit_price,
                            open_price
                        )

                    else:

                        exit_price = max(
                            exit_price,
                            open_price
                        )

                # ------------------------------------------------
                # SLIPPAGE
                # ------------------------------------------------

                if exit_reason in ["SL", "TP"]:

                    if side == 1:

                        if exit_reason == "SL":

                            exit_price *= (
                                1 - slippage_pct
                            )

                        else:

                            exit_price *= (
                                1 - slippage_pct
                            )

                    else:

                        if exit_reason == "SL":

                            exit_price *= (
                                1 + slippage_pct
                            )

                        else:

                            exit_price *= (
                                1 + slippage_pct
                            )

                # ------------------------------------------------
                # PNL
                # ------------------------------------------------

                if side == 1:

                    gross_pnl = (
                        exit_price -
                        entry_price
                    ) * position["size"]

                else:

                    gross_pnl = (
                        entry_price -
                        exit_price
                    ) * position["size"]

                exit_position_value = (
                    abs(exit_price)
                    * position["size"]
                )

                exit_fee = (
                    exit_position_value *
                    fee_rate
                )

                net_pnl = (
                    gross_pnl
                    - exit_fee
                )

                total_fees = (
                    position["entry_fee"]
                    + exit_fee
                )

                # ------------------------------------------------
                # DEVOLVER MARGIN + PNL
                # ------------------------------------------------

                cash += (
                    position["margin"]
                    + net_pnl
                )

                trade_return = (
                    net_pnl /
                    position["margin"]
                )

                trades.append({

                    "trade_id":
                        position["trade_id"],

                    "entry_time":
                        position["entry_time"],

                    "exit_time":
                        dt,

                    "side":
                        side,

                    "entry_price":
                        entry_price,

                    "exit_price":
                        exit_price,

                    "size":
                        position["size"],

                    "margin":
                        position["margin"],

                    "sl_initial":
                        position["sl_initial"],

                    "sl_final":
                        position["sl"],

                    "tp":
                        position["tp"],

                    "bars_held":
                        position["bars_held"],

                    "reason":
                        exit_reason,

                    "gross_pnl":
                        gross_pnl,

                    "fees":
                        total_fees,

                    "net_pnl":
                        net_pnl,

                    "return":
                        trade_return,

                    "breakeven_hit":
                        position["breakeven_hit"],

                    "trailing_active":
                        position["trailing_active"]

                })

                position = None

        # ========================================================
        # 3. ACTUALIZAR TRAILING PARA LA SIGUIENTE VELA
        # ========================================================

        if position is not None:

            if position["trailing_active"]:

                trail_distance = (
                    trailing_atr_mult * atr
                )

                old_sl = position["sl"]

                if position["side"] == 1:

                    new_sl = (
                        high -
                        trail_distance
                    )

                    if new_sl > old_sl:

                        position["sl"] = new_sl

                else:

                    new_sl = (
                        low +
                        trail_distance
                    )

                    if new_sl < old_sl:

                        position["sl"] = new_sl

            # --------------------------------------------
            # BREAK EVEN PARA LA SIGUIENTE VELA
            # --------------------------------------------

            if (
                position["breakeven_hit"]
            ):

                if position["side"] == 1:

                    position["sl"] = max(
                        position["sl"],
                        position["entry_price"]
                    )

                else:

                    position["sl"] = min(
                        position["sl"],
                        position["entry_price"]
                    )

        # ========================================================
        # 4. PREPARAR SEÑAL PARA LA SIGUIENTE VELA
        # ========================================================

        if position is None:

            if cooldown > 0:

                cooldown -= 1

                pending_signal = 0

            else:

                pending_signal = int(
                    np.sign(bar["signal"])
                )

        else:

            pending_signal = 0

        # ========================================================
        # 5. EQUITY
        # ========================================================

        equity = cash

        if position is not None:

            if position["side"] == 1:

                unrealized = (
                    close -
                    position["entry_price"]
                ) * position["size"]

            else:

                unrealized = (
                    position["entry_price"] -
                    close
                ) * position["size"]

            equity += (
                position["margin"]
                + unrealized
            )

        equity_records.append({

            "time": dt,

            "equity": equity,

            "cash": cash,

            "position": (
                position["side"]
                if position is not None
                else 0
            ),

            "close": close

        })

    # ============================================================
    # DATAFRAMES
    # ============================================================

    trades = pd.DataFrame(trades)

    equity_df = pd.DataFrame(
        equity_records
    )

    # ============================================================
    # OPERACIÓN ABIERTA AL FINAL
    # ============================================================

    open_trade = position is not None

    # ERROR CORREGIDO:
    # Una operación abierta al final NO debe contabilizarse
    # como ganadora/perdedora realizada.
    # Se mantiene separada para evitar contaminar las métricas.
    if open_trade:

        final_close = float(
            df.iloc[-1]["Close"]
        )

        if position["side"] == 1:

            unrealized_pnl = (
                final_close -
                position["entry_price"]
            ) * position["size"]

        else:

            unrealized_pnl = (
                position["entry_price"] -
                final_close
            ) * position["size"]

    else:

        unrealized_pnl = 0

    # ============================================================
    # MÉTRICAS
    # ============================================================

    equity_df["peak"] = (
        equity_df["equity"].cummax()
    )

    equity_df["drawdown"] = (
        equity_df["equity"]
        /
        equity_df["peak"]
        - 1
    )

    max_dd = (
        equity_df["drawdown"].min()
    )

    final_equity = (
        equity_df["equity"].iloc[-1]
    )

    total_pnl = (
        final_equity -
        initial_capital
    )

    total_return = (
        final_equity /
        initial_capital
        - 1
    )

    # ============================================================
    # TRADE METRICS
    # ============================================================

    if len(trades) > 0:

        wins = trades[
            trades["net_pnl"] > 0
        ]

        losses = trades[
            trades["net_pnl"] < 0
        ]

        win_rate = (
            len(wins) /
            len(trades)
        )

        avg_win = (
            wins["net_pnl"].mean()
            if len(wins) > 0
            else 0
        )

        avg_loss = (
            losses["net_pnl"].mean()
            if len(losses) > 0
            else 0
        )

        loss_rate = (
            1 - win_rate
        )

        # ERROR CORREGIDO:
        # avg_loss ya es negativo.
        expectancy = (
            win_rate * avg_win
            +
            loss_rate * avg_loss
        )

        gross_profit = (
            wins["net_pnl"].sum()
        )

        gross_loss = abs(
            losses["net_pnl"].sum()
        )

        profit_factor = (
            gross_profit /
            gross_loss
            if gross_loss > 0
            else np.inf
        )

        avg_trade = (
            trades["net_pnl"].mean()
        )

        avg_return = (
            trades["return"].mean()
        )

        median_return = (
            trades["return"].median()
        )

        avg_bars = (
            trades["bars_held"].mean()
        )

        max_loss_streak = 0
        current_streak = 0

        for pnl in trades["net_pnl"]:

            if pnl < 0:

                current_streak += 1

                max_loss_streak = max(
                    max_loss_streak,
                    current_streak
                )

            else:

                current_streak = 0

        max_win_streak = 0
        current_streak = 0

        for pnl in trades["net_pnl"]:

            if pnl > 0:

                current_streak += 1

                max_win_streak = max(
                    max_win_streak,
                    current_streak
                )

            else:

                current_streak = 0

    else:

        win_rate = 0
        avg_win = np.nan
        avg_loss = np.nan
        expectancy = np.nan
        profit_factor = np.nan
        avg_trade = np.nan
        avg_return = np.nan
        median_return = np.nan
        avg_bars = np.nan
        max_loss_streak = 0
        max_win_streak = 0

    # ============================================================
    # SHARPE / SORTINO
    # ============================================================

    equity_returns = (
        equity_df["equity"]
        .pct_change()
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .dropna()
    )

    if annualization_factor is None:

        # Por defecto asumimos datos horarios.
        # IMPORTANTE: cambiar según timeframe.
        annualization_factor = 24 * 365

    if (
        len(equity_returns) > 1
        and equity_returns.std() > 0
    ):

        sharpe = (
            equity_returns.mean()
            /
            equity_returns.std()
            *
            np.sqrt(
                annualization_factor
            )
        )

    else:

        sharpe = np.nan

    downside_returns = (
        equity_returns[
            equity_returns < 0
        ]
    )

    if (
        len(downside_returns) > 0
        and downside_returns.std() > 0
    ):

        sortino = (
            equity_returns.mean()
            /
            downside_returns.std()
            *
            np.sqrt(
                annualization_factor
            )
        )

    else:

        sortino = np.nan

    # ============================================================
    # CALMAR / CAGR
    # ============================================================

    days = (

        equity_df["time"].iloc[-1]
        -
        equity_df["time"].iloc[0]

    ).total_seconds() / 86400

    if days > 0:

        cagr = (
            (
                final_equity /
                initial_capital
            )
            **
            (365 / days)
            - 1
        )

    else:

        cagr = np.nan

    calmar = (
        cagr / abs(max_dd)
        if max_dd < 0
        else np.nan
    )

    # ============================================================
    # VAR / CVAR
    # ============================================================

    if len(equity_returns) > 0:

        var_95 = np.percentile(
            equity_returns,
            5
        )

        cvar_95 = (
            equity_returns[
                equity_returns <= var_95
            ].mean()
        )

    else:

        var_95 = np.nan
        cvar_95 = np.nan

    # ============================================================
    # RECOVERY FACTOR
    # ============================================================

    recovery_factor = (

        total_pnl /
        abs(
            max_dd *
            initial_capital
        )

        if max_dd < 0
        else np.nan
    )

    # ============================================================
    # EXIT REASONS
    # ============================================================

    if len(trades) > 0:

        n_tp = (
            trades["reason"]
            .eq("TP")
            .sum()
        )

        n_sl = (
            trades["reason"]
            .eq("SL")
            .sum()
        )

        n_ts = (
            trades["reason"]
            .eq("TS")
            .sum()
        )

        n_rev = (
            trades["reason"]
            .eq("REV")
            .sum()
        )

        total_fees = (
            trades["fees"].sum()
        )

    else:

        n_tp = 0
        n_sl = 0
        n_ts = 0
        n_rev = 0
        total_fees = 0

    n_trades = len(trades)

    # ============================================================
    # EXPOSURE
    # ============================================================

    exposure = (
        equity_df["position"]
        .ne(0)
        .mean()
    )

    # ============================================================
    # RESULTADO
    # ============================================================

    resume = pd.DataFrame({

        "symbol": [symbol],

        "signal": [name_signal],

        "capital": [
            initial_capital
        ],

        "capital_final": [
            final_equity
        ],

        "pnl": [
            total_pnl
        ],

        "return": [
            total_return
        ],

        "trades": [
            n_trades
        ],

        "win_rate": [
            win_rate
        ],

        "avg_win": [
            avg_win
        ],

        "avg_loss": [
            avg_loss
        ],

        "expectancy": [
            expectancy
        ],

        "profit_factor": [
            profit_factor
        ],

        "avg_trade": [
            avg_trade
        ],

        "avg_return": [
            avg_return
        ],

        "median_return": [
            median_return
        ],

        "max_drawdown": [
            max_dd
        ],

        "sharpe": [
            sharpe
        ],

        "sortino": [
            sortino
        ],

        "cagr": [
            cagr
        ],

        "calmar": [
            calmar
        ],

        "recovery_factor": [
            recovery_factor
        ],

        "var_95": [
            var_95
        ],

        "cvar_95": [
            cvar_95
        ],

        "exposure": [
            exposure
        ],

        "avg_bars": [
            avg_bars
        ],

        "max_loss_streak": [
            max_loss_streak
        ],

        "max_win_streak": [
            max_win_streak
        ],

        "n_tp": [
            n_tp
        ],

        "n_sl": [
            n_sl
        ],

        "n_ts": [
            n_ts
        ],

        "n_rev": [
            n_rev
        ],

        "fees": [
            total_fees
        ],

        "open_trade": [
            open_trade
        ]

    })

    return resume, trades, equity_df