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


## Signals & Strategies


EMA_LONG        = 200

# Bollinger
BB_PERIOD       = 20
BB_STD          = 2.0

# ATR
ATR_PERIOD      = 14

# VPVMA
BANDWITH = 0.1 # 0.1 filtro para reducir señales falsas (default 0.1)

def build_signals(df):

    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    ema20 = df["EMA20"]
    ema50 = df["EMA50"]
    ema200 = df["EMA200"]

    rsi = df["RSI"]

    macd = df["MACD"]
    macd_signal = df["MACD_signal"]

    atr = df["ATR"]

    # =========================================================
    # 1. MACD + RSI
    # =========================================================

    # ---------------------------------------------------------
    # 1.1 MACD + RSI ROBUST
    # ---------------------------------------------------------

    rsi_oversold = rsi < 30
    rsi_overbought = rsi > 70

    count_oversold = (
        rsi_oversold
        .rolling(6, min_periods=6)
        .sum()
    )

    count_overbought = (
        rsi_overbought
        .rolling(6, min_periods=6)
        .sum()
    )

    macd_bull = macd > macd_signal
    macd_bear = macd < macd_signal

    df["sign_macd_rsi_robust"] = np.select(
        [
            (count_oversold >= 4) & macd_bull,
            (count_overbought >= 4) & macd_bear
        ],
        [
            1,
            -1
        ],
        default=0
    )

    # ---------------------------------------------------------
    # 1.2 MACD + RSI SIMPLE
    # ---------------------------------------------------------

    df["sign_macd_rsi_simple"] = np.select(
        [
            (macd > macd_signal) & (rsi < 30),
            (macd < macd_signal) & (rsi > 70)
        ],
        [
            1,
            -1
        ],
        default=0
    )

    # ---------------------------------------------------------
    # 1.3 MACD + RSI + BREAKOUT
    # ---------------------------------------------------------

    previous_high = high.shift(1)
    previous_low = low.shift(1)

    df["sign_macd_rsi_simple_break"] = np.select(
        [
            (df["sign_macd_rsi_simple"] == 1) &
            (close > previous_high),

            (df["sign_macd_rsi_simple"] == -1) &
            (close < previous_low)
        ],
        [
            1,
            -1
        ],
        default=0
    )

    # ---------------------------------------------------------
    # 1.4 MACD + RSI + BB
    # ---------------------------------------------------------

    bb_pos = df["BB_position"]

    df["sign_macd_rsi_simple_bb_pos"] = np.select(
        [
            (df["sign_macd_rsi_simple"] == 1) &
            (bb_pos < 0.20),

            (df["sign_macd_rsi_simple"] == -1) &
            (bb_pos > 0.80)
        ],
        [
            1,
            -1
        ],
        default=0
    )

    # ---------------------------------------------------------
    # 1.5 RSI + BB
    # ---------------------------------------------------------

    df["sign_macd_rsi_bb"] = np.select(
        [
            (rsi < 30) &
            (close < df["BB_lower"]),

            (rsi > 70) &
            (close > df["BB_upper"])
        ],
        [
            1,
            -1
        ],
        default=0
    )

    # ---------------------------------------------------------
    # 1.6 RSI + BB + EMA200
    # ---------------------------------------------------------

    df["sign_macd_rsi_bb_ema"] = np.select(
        [
            (rsi < 30) &
            (close < df["BB_lower"]) &
            (close > ema200),

            (rsi > 70) &
            (close > df["BB_upper"]) &
            (close < ema200)
        ],
        [
            1,
            -1
        ],
        default=0
    )

    # =========================================================
    # 2. ADX
    # =========================================================

    adx = df["ADX"]
    plus_di = df["ADX+DI"]
    minus_di = df["ADX-DI"]

    # ---------------------------------------------------------
    # 2.1 ADX + DI
    # ---------------------------------------------------------

    df["ind_adx"] = np.select(
        [
            (adx >= 25) & (plus_di > minus_di),
            (adx >= 25) & (minus_di > plus_di)
        ],
        [
            1,
            -1
        ],
        default=0
    )

    # ---------------------------------------------------------
    # 2.2 ADX CRECIENTE
    # ---------------------------------------------------------

    adx_rising = adx > adx.shift(1)

    df["sign_adx_crec"] = np.select(
        [
            (df["ind_adx"] == 1) & adx_rising,
            (df["ind_adx"] == -1) & adx_rising
        ],
        [
            1,
            -1
        ],
        default=0
    )

    # ---------------------------------------------------------
    # 2.3 ADX + RSI
    # ---------------------------------------------------------

    df["sign_adx_crec_rsi"] = np.select(
        [
            (df["ind_adx"] == 1) &
            adx_rising &
            (rsi < 45),

            (df["ind_adx"] == -1) &
            adx_rising &
            (rsi > 55)
        ],
        [
            1,
            -1
        ],
        default=0
    )

    df["sign_adx_rsi"] = np.select(
        [
            (df["ind_adx"] == 1) & (rsi < 40),
            (df["ind_adx"] == -1) & (rsi > 60)
        ],
        [
            1,
            -1
        ],
        default=0
    )

    df["sign_adx_rsi_robust"] = np.select(
        [
            (df["ind_adx"] == 1) & (rsi < 30),
            (df["ind_adx"] == -1) & (rsi > 70)
        ],
        [
            1,
            -1
        ],
        default=0
    )

    df["sign_adx_rsi_xtrem"] = np.select(
        [
            (df["ind_adx"] == 1) & (rsi < 20),
            (df["ind_adx"] == -1) & (rsi > 80)
        ],
        [
            1,
            -1
        ],
        default=0
    )

    # =========================================================
    # 3. VPVMA
    # =========================================================

    vpvma = df["VPVMA"]
    vpvmas = df["VPVMAS"]

    vpvma_bull_cross = (
        (vpvma > (1 + BANDWITH) * vpvmas) &
        (vpvma.shift(1) <=
         (1 + BANDWITH) * vpvmas.shift(1))
    )

    vpvma_bear_cross = (
        (vpvma < (1 - BANDWITH) * vpvmas) &
        (vpvma.shift(1) >=
         (1 - BANDWITH) * vpvmas.shift(1))
    )

    df["sign_vpvma"] = np.select(
        [
            vpvma_bull_cross,
            vpvma_bear_cross
        ],
        [
            1,
            -1
        ],
        default=0
    )

    # =========================================================
    # 4. SINERGIA
    # =========================================================

    long_condition = (
        (close > ema20) &
        (rsi > 55) &
        (macd > macd_signal) &
        (close > df["VWAP"]) &
        (df["spread"] < 0.01) &
        df["volume_increasing"] &
        (df["order_flow"] == "buy")
    )

    short_condition = (
        (close < ema20) &
        (rsi < 45) &
        (macd < macd_signal) &
        (close < df["VWAP"]) &
        (df["spread"] < 0.01) &
        df["volume_increasing"] &
        (df["order_flow"] == "sell")
    )

    df["sign_sinergia"] = np.select(
        [
            long_condition,
            short_condition
        ],
        [
            1,
            -1
        ],
        default=0
    )

    # =========================================================
    # 5. RSI + BB MEAN REVERSION
    # =========================================================

    df["sign_rsi_bb_rever"] = np.select(
        [
            (rsi < 30) & (bb_pos <= 0.05),
            (rsi > 70) & (bb_pos >= 0.95)
        ],
        [
            1,
            -1
        ],
        default=0
    )

    # =========================================================
    # 6. MA + MACD TREND FOLLOWING
    # =========================================================

    long_ma_macd = (
        (ema50 > ema200) &
        (macd > macd_signal) &
        (close > ema200)
    )

    short_ma_macd = (
        (ema50 < ema200) &
        (macd < macd_signal) &
        (close < ema200)
    )


    df["sign_ma_macd"] = np.select(
        [
            long_ma_macd,
            short_ma_macd
        ],
        [
            1,
            -1
        ],
        default=0
    )

    # =========================================================
    # 7. SQUEEZE + KELTNER + BB
    # =========================================================

    squeeze_length = 20
    mult_bb = 2.0
    mult_kc = 1.5

    ma20 = close.rolling(
        squeeze_length,
        min_periods=squeeze_length
    ).mean()

    std20 = close.rolling(
        squeeze_length,
        min_periods=squeeze_length
    ).std()

    bb_upper = ma20 + mult_bb * std20
    bb_lower = ma20 - mult_bb * std20

    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ],
        axis=1
    ).max(axis=1)

    atr_kc = tr.rolling(
        squeeze_length,
        min_periods=squeeze_length
    ).mean()

    kc_upper = ma20 + mult_kc * atr_kc
    kc_lower = ma20 - mult_kc * atr_kc

    squeeze = (
        (bb_upper < kc_upper) &
        (bb_lower > kc_lower)
    )

    # CORRECCIÓN IMPORTANTE:
    # El squeeze NO es una señal de compra.
    # Es un estado de compresión.
    # La señal aparece cuando sale del squeeze.

    squeeze_release = (
        squeeze.shift(1).fillna(False) &
        ~squeeze
    )

    df["str_squ_kel_bb"] = np.select(
        [
            squeeze_release & (macd > macd_signal),
            squeeze_release & (macd < macd_signal)
        ],
        [
            1,
            -1
        ],
        default=0
    )

    # =========================================================
    # 8. GOLDEN / DEATH CROSS
    # =========================================================

    golden_cross = (
        (ema20 > ema50) &
        (ema20.shift(1) <= ema50.shift(1))
    )

    death_cross = (
        (ema20 < ema50) &
        (ema20.shift(1) >= ema50.shift(1))
    )

    df["sign_gold"] = np.select(
        [
            golden_cross,
            death_cross
        ],
        [
            1,
            -1
        ],
        default=0
    )

    # =========================================================
    # 9. 3 EMA
    # =========================================================

    df["ind_3ema"] = np.select(
        [
            (ema20 > ema50) &
            (ema50 > ema200),

            (ema20 < ema50) &
            (ema50 < ema200)
        ],
        [
            1,
            -1
        ],
        default=0
    )

    # =========================================================
    # 10. RSI EXTREMES
    # =========================================================

    df["sign_rsi30"] = np.select([rsi < 30, rsi > 70], [1,-1],default=0)

    df["sign_rsi20"] = np.select([rsi < 20, rsi > 80], [1, -1],
        default=0
    )

    df["sign_rsi15"] = np.select(
        [
            rsi < 15,
            rsi > 85
        ],
        [
            1,
            -1
        ],
        default=0
    )

    # =========================================================
    # 11. ADX TREND STRENGTH
    # =========================================================

    adx_direction = np.select(
        [
            plus_di > minus_di,
            minus_di > plus_di
        ],
        [
            1,
            -1
        ],
        default=0
    )

    df["trend_adx"] = (
        adx_direction *
        pd.cut(
            adx,
            bins=[-np.inf, 15, 25, 40, 60, 70, np.inf],
            labels=[0, 1, 2, 3, 4, 5]
        ).astype(float)
    )

    df.loc[adx.isna(), "trend_adx"] = 0

    # =========================================================
    # 12. PULLBACK + EMA
    # =========================================================

    pullback_long = (
        (close > ema50) &
        (low <= ema20) &
        (close > ema20)
    )

    pullback_short = (
        (close < ema50) &
        (high >= ema20) &
        (close < ema20)
    )

    df["sign_pullb_ema"] = np.select(
        [
            pullback_long,
            pullback_short
        ],
        [
            1,
            -1
        ],
        default=0
    )

    # =========================================================
    # 13. RSI DIVERGENCE
    # =========================================================

    def find_pivots(series, left=3, right=3):

        pivot_high = pd.Series(False, index=series.index)
        pivot_low = pd.Series(False, index=series.index)

        for i in range(left, len(series) - right):

            current = series.iloc[i]

            left_values = series.iloc[
                i - left:i
            ]

            right_values = series.iloc[
                i + 1:i + right + 1
            ]

            if current > left_values.max() and current >= right_values.max():
                pivot_high.iloc[i] = True

            if current < left_values.min() and current <= right_values.min():
                pivot_low.iloc[i] = True

        return pivot_high, pivot_low

    rsi_pivot_high, rsi_pivot_low = find_pivots(
        close,
        left=3,
        right=3
    )

    pivot_high_idx = np.where(
        rsi_pivot_high.values
    )[0]

    pivot_low_idx = np.where(
        rsi_pivot_low.values
    )[0]

    bullish_rsi_div = pd.Series(
        False,
        index=df.index
    )

    bearish_rsi_div = pd.Series(
        False,
        index=df.index
    )

    for i in range(1, len(pivot_low_idx)):

        prev_idx = pivot_low_idx[i - 1]
        curr_idx = pivot_low_idx[i]

        price_ll = (
            close.iloc[curr_idx] <
            close.iloc[prev_idx]
        )

        rsi_hl = (
            rsi.iloc[curr_idx] >
            rsi.iloc[prev_idx]
        )

        if price_ll and rsi_hl:

            confirmation_idx = curr_idx + 3

            if confirmation_idx < len(df):
                bullish_rsi_div.iloc[
                    confirmation_idx
                ] = True

    for i in range(1, len(pivot_high_idx)):

        prev_idx = pivot_high_idx[i - 1]
        curr_idx = pivot_high_idx[i]

        price_hh = (
            close.iloc[curr_idx] >
            close.iloc[prev_idx]
        )

        rsi_lh = (
            rsi.iloc[curr_idx] <
            rsi.iloc[prev_idx]
        )

        if price_hh and rsi_lh:

            confirmation_idx = curr_idx + 3

            if confirmation_idx < len(df):
                bearish_rsi_div.iloc[
                    confirmation_idx
                ] = True

    df["sign_rsi_div"] = np.select(
        [
            bullish_rsi_div,
            bearish_rsi_div
        ],
        [
            1,
            -1
        ],
        default=0
    )

    # =========================================================
    # 14. VWAP DEVIATION
    # =========================================================

    vwap = df["VWAP"]

    vwap_dev = (
        (close - vwap) /
        vwap.replace(0, np.nan)
    )

    df["ind_vwap_deviation"] = np.select(
        [
            vwap_dev < -0.02,
            vwap_dev > 0.02
        ],
        [
            1,
            -1
        ],
        default=0
    )

    # =========================================================
    # 15. SUPPORT / RESISTANCE BREAKOUT
    # =========================================================

    sr_period = 50

    resistance = high.shift(1).rolling(
        sr_period,
        min_periods=sr_period
    ).max()

    support = low.shift(1).rolling(
        sr_period,
        min_periods=sr_period
    ).min()

    breakout_up = close > resistance
    breakout_down = close < support

    # En vez de invertir inmediatamente el breakout,
    # mantenemos la dirección del rompimiento.

    df["sign_supo_resi"] = np.select(
        [
            breakout_up,
            breakout_down
        ],
        [
            1,
            -1
        ],
        default=0
    )

    # =========================================================
    # 16. MOMENTUM
    # =========================================================

    # roc_period = 10

    # roc = close.pct_change(roc_period)

    # df["ROC"] = roc

    # df["ind_momentum"] = np.select(
    #     [
    #         roc > 0.02,
    #         roc < -0.02
    #     ],
    #     [
    #         1,
    #         -1
    #     ],
    #     default=0
    # )

    # =========================================================
    # 17. VOLUME ANOMALY
    # =========================================================

    volume_mean = df["Volume"].rolling(
        20,
        min_periods=20
    ).mean()

    volume_ratio = (
        df["Volume"] /
        volume_mean.replace(0, np.nan)
    )

    # Esto NO es una señal direccional.
    #
    # 1 = volumen anormal
    # 0 = normal

    df["ind_volume_anomaly"] = (
        volume_ratio > 2.0
    ).astype(int)

    # =========================================================
    # 18. RANGE COMPRESSION
    # =========================================================

    candle_range = high - low

    range_mean = candle_range.rolling(
        20,
        min_periods=20
    ).mean()

    compression = (
        candle_range <
        0.5 * range_mean
    )

    # Mantengo el nombre original.
    #
    # Pero conceptualmente es un indicador de estado,
    # NO una señal BUY.

    df["ind_compress_range"] = compression.astype(int)

    # =========================================================
    # 19. EMA DISTANCE / SOBREEXTENSIÓN
    # =========================================================

    ema_dist = (
        (close - ema50) /
        ema50.replace(0, np.nan)
    )

    df["sign_sobreextema"] = np.select(
        [
            ema_dist > 0.03,
            ema_dist < -0.03
        ],
        [
            -1,
            1
        ],
        default=0
    )

    # =========================================================
    # 20. STOCHASTIC CROSS
    # =========================================================

    k = df["STOCH_K"]
    d = df["STOCH_D"]

    stoch_buy = (
        (k < 25) &
        (d < 25) &
        (k.shift(1) <= d.shift(1)) &
        (k > d)
    )

    stoch_sell = (
        (k > 75) &
        (d > 75) &
        (k.shift(1) >= d.shift(1)) &
        (k < d)
    )

    df["sign_stochastic"] = np.select(
        [
            stoch_buy,
            stoch_sell
        ],
        [
            1,
            -1
        ],
        default=0
    )

    # =========================================================
    # 21. CMF
    # =========================================================

    cmf_period = 20
    cmf_threshold = 0.05

    hl_range = (
        high - low
    ).replace(0, np.nan)

    clv = (
        ((close - low) -
         (high - close))
        / hl_range
    )

    cmf = (
        (clv * df["Volume"])
        .rolling(cmf_period)
        .sum()
        /
        df["Volume"]
        .rolling(cmf_period)
        .sum()
    )

    cmf_rising = cmf > cmf.shift(1)

    df["sign_cmf"] = np.select(
        [
            (cmf > cmf_threshold) &
            cmf_rising,

            (cmf < -cmf_threshold) &
            ~cmf_rising
        ],
        [
            1,
            -1
        ],
        default=0
    )

    # =========================================================
    # 22. PARABOLIC SAR
    # =========================================================

    def calc_psar(high, low, iaf=0.02, maxaf=0.20):

        high = high.reset_index(drop=True)
        low = low.reset_index(drop=True)

        n = len(high)

        psar = np.full(n, np.nan)

        if n < 3:
            return pd.Series(psar)

        bull = True
        af = iaf

        ep_high = high.iloc[0]
        ep_low = low.iloc[0]

        psar.iloc[0] = low.iloc[0]
        psar.iloc[1] = low.iloc[0]

        for i in range(2, n):

            if bull:

                psar.iloc[i] = (
                    psar.iloc[i - 1] +
                    af *
                    (ep_high - psar.iloc[i - 1])
                )

                psar.iloc[i] = min(
                    psar.iloc[i],
                    low.iloc[i - 1],
                    low.iloc[i - 2]
                )

                if low.iloc[i] < psar.iloc[i]:

                    bull = False

                    psar.iloc[i] = ep_high

                    ep_low = low.iloc[i]

                    af = iaf

                elif high.iloc[i] > ep_high:

                    ep_high = high.iloc[i]

                    af = min(
                        af + iaf,
                        maxaf
                    )

            else:

                psar.iloc[i] = (
                    psar.iloc[i - 1] +
                    af *
                    (ep_low - psar.iloc[i - 1])
                )

                psar.iloc[i] = max(
                    psar.iloc[i],
                    high.iloc[i - 1],
                    high.iloc[i - 2]
                )

                if high.iloc[i] > psar.iloc[i]:

                    bull = True

                    psar.iloc[i] = ep_low

                    ep_high = high.iloc[i]

                    af = iaf

                elif low.iloc[i] < ep_low:

                    ep_low = low.iloc[i]

                    af = min(
                        af + iaf,
                        maxaf
                    )

        return psar

    # df["PSAR"] = calc_psar(
    #     df["High"],
    #     df["Low"]
    # )

    # df["ind_psar"] = np.select(
    #     [
    #         close > df["PSAR"],
    #         close < df["PSAR"]
    #     ],
    #     [
    #         1,
    #         -1
    #     ],
    #     default=0
    # )

    # =========================================================
    # 23. ELDER RAY
    # =========================================================

    elder_period = 13

    ema13 = close.ewm(
        span=elder_period,
        adjust=False
    ).mean()

    bull_power = high - ema13
    bear_power = low - ema13

    bull_power_cross = (
        (bull_power > 0) &
        (bull_power.shift(1) <= 0)
    )

    bear_power_cross = (
        (bear_power < 0) &
        (bear_power.shift(1) >= 0)
    )

    df["sign_elderray"] = np.select(
        [
            bull_power_cross,
            bear_power_cross
        ],
        [
            1,
            -1
        ],
        default=0
    )

    # =========================================================
    # 24. WILLIAMS %R
    # =========================================================

    willr_period = 14

    highest_high = high.rolling(
        willr_period,
        min_periods=willr_period
    ).max()

    lowest_low = low.rolling(
        willr_period,
        min_periods=willr_period
    ).min()

    willr_range = (
        highest_high -
        lowest_low
    ).replace(0, np.nan)

    df["WillR"] = (
        -100 *
        (highest_high - close) /
        willr_range
    )

    willr = df["WillR"]

    # CORRECCIÓN CRÍTICA:
    #
    # Antes:
    #
    # shift(-2)
    # shift(-1)
    #
    # Eso utiliza información futura.
    #
    # Ahora esperamos a que el cruce haya ocurrido.

    willr_buy = (
        (willr.shift(1) < -80) &
        (willr >= -80)
    )

    willr_sell = (
        (willr.shift(1) > -20) &
        (willr <= -20)
    )

    df["sign_willr"] = np.select(
        [
            willr_buy,
            willr_sell
        ],
        [
            1,
            -1
        ],
        default=0
    )

    # =========================================================
    # 25. DONCHIAN BREAKOUT
    # =========================================================

    donchian_period = 20

    donchian_high = high.shift(1).rolling(
        donchian_period,
        min_periods=donchian_period
    ).max()

    donchian_low = low.shift(1).rolling(
        donchian_period,
        min_periods=donchian_period
    ).min()

    df["sign_donchian"] = np.select(
        [
            close > donchian_high,
            close < donchian_low
        ],
        [
            1,
            -1
        ],
        default=0
    )

    # =========================================================
    # 26. LIQUIDITY SWEEP
    # =========================================================

    liquidity_period = 20

    previous_high = high.shift(1).rolling(
        liquidity_period,
        min_periods=liquidity_period
    ).max()

    previous_low = low.shift(1).rolling(
        liquidity_period,
        min_periods=liquidity_period
    ).min()

    bearish_sweep = (
        (high > previous_high) &
        (close < previous_high)
    )

    bullish_sweep = (
        (low < previous_low) &
        (close > previous_low)
    )

    df["sign_liquidsweep"] = np.select(
        [
            bullish_sweep,
            bearish_sweep
        ],
        [
            1,
            -1
        ],
        default=0
    )

    # =========================================================
    # 27. MACD DIVERGENCE
    # =========================================================

    macd_pivot_high, macd_pivot_low = find_pivots(
        close,
        left=3,
        right=3
    )

    high_idx = np.where(
        macd_pivot_high.values
    )[0]

    low_idx = np.where(
        macd_pivot_low.values
    )[0]

    bullish_macd_div = pd.Series(
        False,
        index=df.index
    )

    bearish_macd_div = pd.Series(
        False,
        index=df.index
    )

    for i in range(1, len(low_idx)):

        prev_idx = low_idx[i - 1]
        curr_idx = low_idx[i]

        price_ll = (
            close.iloc[curr_idx] <
            close.iloc[prev_idx]
        )

        macd_hl = (
            macd.iloc[curr_idx] >
            macd.iloc[prev_idx]
        )

        if price_ll and macd_hl:

            confirmation_idx = curr_idx + 3

            if confirmation_idx < len(df):
                bullish_macd_div.iloc[
                    confirmation_idx
                ] = True

    for i in range(1, len(high_idx)):

        prev_idx = high_idx[i - 1]
        curr_idx = high_idx[i]

        price_hh = (
            close.iloc[curr_idx] >
            close.iloc[prev_idx]
        )

        macd_lh = (
            macd.iloc[curr_idx] <
            macd.iloc[prev_idx]
        )

        if price_hh and macd_lh:

            confirmation_idx = curr_idx + 3

            if confirmation_idx < len(df):
                bearish_macd_div.iloc[
                    confirmation_idx
                ] = True

    macd_div_signal = np.select(
        [
            bullish_macd_div,
            bearish_macd_div
        ],
        [
            1,
            -1
        ],
        default=0
    )

    # V1 = señal únicamente en confirmación
    df["sign_macd_div1"] = macd_div_signal

    # V2 = mantener señal 3 velas
    df["sign_macd_div3"] = (
        pd.Series(
            macd_div_signal,
            index=df.index
        )
        .replace(0, np.nan)
        .ffill(limit=2)
        .fillna(0)
        .astype(int)
    )

    # V3 = mantener señal 5 velas
    df["sign_macd_div5"] = (
        pd.Series(
            macd_div_signal,
            index=df.index
        )
        .replace(0, np.nan)
        .ffill(limit=4)
        .fillna(0)
        .astype(int)
    )


    # =========================================================
    # 28. RANDOM BASELINES
    # =========================================================

    rng = np.random.default_rng(42)

    df["sign_random"] = rng.choice(
        [-1, 0, 1],
        size=len(df),
        p=[0.05, 0.90, 0.05]
    )

    df["sign_random2"] = rng.choice(
        [-1, 0, 1],
        size=len(df),
        p=[0.025, 0.95, 0.025]
    )

    df["sign_random3"] = rng.choice(
        [-1, 0, 1],
        size=len(df),
        p=[0.10, 0.80, 0.10]
    )

    # =========================================================
    # LIMPIEZA
    # =========================================================

    temporary_columns = ["ROC"]

    for col in temporary_columns:
        if col in df.columns:
            del df[col]

    # ---------------------------------------------------------
    # LIMPIAR FILAS INICIALES
    # ---------------------------------------------------------

    warmup = max(
        EMA_LONG,
        BB_PERIOD,
        ATR_PERIOD,
        52,
        50
    )

    df = df.iloc[warmup:].copy()

    # ---------------------------------------------------------
    # VALIDAR QUE LAS SEÑALES SEAN NUMÉRICAS
    # ---------------------------------------------------------

    signal_columns = [
        col for col in df.columns
        if (
            col.startswith("sign_") or
            col.startswith("str_") or
            col.startswith("ind_")
        )
    ]

    for col in signal_columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        ).fillna(0)

    return df