import yfinance as yf
import pandas as pd
import numpy as np

# EMAS
EMA_SHORT       = 20
EMA_MID         = 50
EMA_LONG        = 200
# RSI
RSI_PERIOD      = 14
# MACD
MACD_FAST       = 12
MACD_SLOW       = 26
MACD_SIGNAL     = 9
# Bollinger
BB_PERIOD       = 20
BB_STD          = 2.0
# Estocastico
STOCH_K         = 9
STOCH_D         = 3
STOCH_SLOWING   = 3  
# ATR
ATR_PERIOD      = 14
# ATR - Gestión de riesgo
ATR_MULT_SL     = 1.5           # Multiplicador ATR para Stop Loss
ATR_MULT_TP     = 3.0           # Multiplicador ATR para Take Profit (ratio 2:1)
# Parámetro principal: mínimo de reglas que deben cumplirse 
# Rango válido: 1–8. Recomendado: 5 (conservador-moderado), 4 (agresivo), 6 (muy conservador)
MIN_RULES_TO_SIGNAL = 5
# Umbrales RSI 
RSI_OVERSOLD    = 35 # 35
RSI_OVERBOUGHT  = 70 # 65
# Umbrales Estocástico
STOCH_OVERSOLD  = 25
STOCH_OVERBOUGHT = 75
# Umbrales de posición dentro del canal BB
BB_ENTRY_SELL_MIN  = 0.75   # Para VENDER: precio debe estar en el > 75% del canal
BB_ENTRY_BUY_MAX   = 0.25   # Para COMPRAR: precio debe estar en el < 25% del canal
BB_WAIT_SELL_MIN   = 0.45   # Por debajo de esto en venta  → esperar subida
BB_WAIT_BUY_MAX    = 0.55   # Por encima de esto en compra → esperar bajada
# Volatility
VOLATILITY_PERIOD = 20


OBV_EMA_PERIOD = 20

ADX_PERIOD = 14

VOLATILITY_PERIOD = 20

CMF_PERIOD = 20

WILLR_PERIOD = 14

DONCHIAN_PERIOD = 20

ROC_PERIOD = 10

VPVMA_FAST = 12
VPVMA_SLOW = 26
VPVMA_SIGNAL = 9

PSAR_IAF = 0.02
PSAR_MAX_AF = 0.20

VWAP_MODE = "auto"


def load_yahoo(symbol: str, INTERVAL:int, LOOKBACK_DAYS:int) -> dict:
    # symbol = "CVS" # usar antes del 15/04 ya que 3 estrategias se alinean para sell
    # LOOKBACK_DAYS = 40
    # INTERVAL = "1h"

    for _ in range(3):
        try:
            df = yf.download(
                symbol,
                period=f"{LOOKBACK_DAYS}d",
                interval=INTERVAL,
                progress=False,
                auto_adjust=True,
            )
        except Exception as e:
            print(f"ERROR: {symbol} - {e}")
            return pd.DataFrame()

        if len(df) > 0:
            break   

    if df.empty or len(df) < EMA_LONG + 10:
        print(f"ERROR: {symbol} - len(df) < {EMA_LONG + 10}")
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if len(df) < 2:
        print(f"ERROR: {symbol} - len(df) < 2")
        return pd.DataFrame()
    
    df["date"] = df.index

    return df

def calc_ema(
    series: pd.Series,
    period: int
) -> pd.Series:

    if period <= 0:
        raise ValueError(
            "period debe ser mayor que 0."
        )

    return series.ewm(
        span=period,
        adjust=False,
        min_periods=period
    ).mean()


def calc_rsi(
    close: pd.Series,
    period: int = 14
) -> pd.Series:

    if period <= 0:
        raise ValueError(
            "period debe ser mayor que 0."
        )

    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder smoothing
    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    rsi = 100 - (
        100 / (1 + rs)
    )

    # Casos extremos
    rsi = rsi.where(
        avg_loss != 0,
        100
    )

    rsi = rsi.where(
        avg_gain != 0,
        0
    )

    # Cuando ambos son cero, no existe movimiento
    both_zero = (
        (avg_gain == 0) &
        (avg_loss == 0)
    )

    rsi = rsi.where(
        ~both_zero,
        50
    )

    return rsi



def calc_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
):

    if fast >= slow:
        raise ValueError(
            "fast debe ser menor que slow."
        )

    ema_fast = calc_ema(
        close,
        fast
    )

    ema_slow = calc_ema(
        close,
        slow
    )

    macd_line = (
        ema_fast -
        ema_slow
    )

    signal_line = calc_ema(
        macd_line,
        signal
    )

    histogram = (
        macd_line -
        signal_line
    )

    return (
        macd_line,
        signal_line,
        histogram
    )


# ============================================================
# BOLLINGER BANDS
# ============================================================

def calc_bollinger(
    close: pd.Series,
    period: int = 20,
    std_mult: float = 2.0
):

    mid = close.rolling(
        window=period,
        min_periods=period
    ).mean()

    std = close.rolling(
        window=period,
        min_periods=period
    ).std()

    upper = (
        mid +
        std_mult * std
    )

    lower = (
        mid -
        std_mult * std
    )

    return (
        upper,
        mid,
        lower
    )


# ============================================================
# STOCHASTIC
# ============================================================

def calc_stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 9,
    slowing: int = 3,
    d_period: int = 3
):

    lowest_low = low.rolling(
        window=k_period,
        min_periods=k_period
    ).min()

    highest_high = high.rolling(
        window=k_period,
        min_periods=k_period
    ).max()

    denominator = (
        highest_high -
        lowest_low
    )

    denominator = denominator.replace(
        0,
        np.nan
    )

    raw_k = (
        100 *
        (close - lowest_low) /
        denominator
    )

    k = raw_k.rolling(
        window=slowing,
        min_periods=slowing
    ).mean()

    d = k.rolling(
        window=d_period,
        min_periods=d_period
    ).mean()

    return (
        k,
        d
    )


# ============================================================
# ATR
# ============================================================

def calc_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> pd.Series:

    previous_close = close.shift(1)

    tr_components = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs()
        ],
        axis=1
    )

    true_range = tr_components.max(
        axis=1
    )

    # Wilder ATR
    atr = true_range.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    return atr


# ============================================================
# OBV
# ============================================================

def calc_obv(
    close: pd.Series,
    volume: pd.Series
) -> pd.Series:

    direction = np.sign(
        close.diff()
    ).fillna(0)

    obv = (
        direction *
        volume
    ).cumsum()

    return obv


# ============================================================
# ADX
# ============================================================

def calc_adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
):

    up_move = high.diff()

    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where(
            (up_move > down_move) &
            (up_move > 0),
            up_move,
            0.0
        ),
        index=close.index
    )

    minus_dm = pd.Series(
        np.where(
            (down_move > up_move) &
            (down_move > 0),
            down_move,
            0.0
        ),
        index=close.index
    )

    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs()
        ],
        axis=1
    ).max(axis=1)

    # Wilder smoothing
    atr_wilder = true_range.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    plus_dm_wilder = plus_dm.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    minus_dm_wilder = minus_dm.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    denominator = atr_wilder.replace(
        0,
        np.nan
    )

    plus_di = (
        100 *
        plus_dm_wilder /
        denominator
    )

    minus_di = (
        100 *
        minus_dm_wilder /
        denominator
    )

    di_sum = (
        plus_di +
        minus_di
    ).replace(
        0,
        np.nan
    )

    dx = (
        100 *
        (plus_di - minus_di).abs() /
        di_sum
    )

    adx_line = dx.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    return (
        adx_line,
        plus_di,
        minus_di
    )


# ============================================================
# BB POSITION
# ============================================================

def calc_bb_position(
    df: pd.DataFrame
) -> pd.Series:
    """
    Retorna la posición relativa del precio dentro del canal BB.
      0.0 → precio en la banda inferior
      0.5 → precio en la media (mitad del canal)
      1.0 → precio en la banda superior
    """

    width = (
        df["BB_upper"] -
        df["BB_lower"]
    )

    width = width.replace(
        0,
        np.nan
    )

    return (
        df["Close"] -
        df["BB_lower"]
    ) / width



def calc_consistencia(
    close: pd.Series
):

    prev26 = close.shift(26)
    prev52 = close.shift(52)

    return (
        prev26,
        prev52
    )


# ============================================================
# CMF
# ============================================================

def calc_cmf(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 20
) -> pd.Series:

    denominator = (
        high -
        low
    ).replace(
        0,
        np.nan
    )

    clv = (
        (close - low) -
        (high - close)
    ) / denominator

    volume_sum = volume.rolling(
        period,
        min_periods=period
    ).sum()

    money_flow = (
        clv * volume
    ).rolling(
        period,
        min_periods=period
    ).sum()

    return money_flow / volume_sum.replace(
        0,
        np.nan
    )


# ============================================================
# PARABOLIC SAR
# ============================================================

def calc_psar(
    high: pd.Series,
    low: pd.Series,
    iaf: float = 0.02,
    maxaf: float = 0.20
) -> pd.Series:

    high_values = high.to_numpy(
        dtype=float
    )

    low_values = low.to_numpy(
        dtype=float
    )

    length = len(high_values)

    psar = np.full(
        length,
        np.nan
    )

    if length == 0:
        return pd.Series(
            psar,
            index=high.index
        )

    if length == 1:
        psar[0] = low_values[0]

        return pd.Series(
            psar,
            index=high.index
        )

    bull = True

    af = iaf

    hp = high_values[0]
    lp = low_values[0]

    psar[0] = low_values[0]
    psar[1] = low_values[0]

    for i in range(2, length):

        if bull:

            psar[i] = (
                psar[i - 1] +
                af *
                (hp - psar[i - 1])
            )

            psar[i] = min(
                psar[i],
                low_values[i - 1],
                low_values[i - 2]
            )

            if low_values[i] < psar[i]:

                bull = False

                psar[i] = hp

                lp = low_values[i]

                af = iaf

            else:

                if high_values[i] > hp:

                    hp = high_values[i]

                    af = min(
                        af + iaf,
                        maxaf
                    )

        else:

            psar[i] = (
                psar[i - 1] +
                af *
                (lp - psar[i - 1])
            )

            psar[i] = max(
                psar[i],
                high_values[i - 1],
                high_values[i - 2]
            )

            if high_values[i] > psar[i]:

                bull = True

                psar[i] = lp

                hp = high_values[i]

                af = iaf

            else:

                if low_values[i] < lp:

                    lp = low_values[i]

                    af = min(
                        af + iaf,
                        maxaf
                    )

    return pd.Series(
        psar,
        index=high.index
    )


# ============================================================
# VWAP
# ============================================================

def calc_vwap(
    close: pd.Series,
    volume: pd.Series,
    mode: str = "auto"
) -> pd.Series:

    if mode not in [
        "auto",
        "daily",
        "global"
    ]:
        raise ValueError(
            "mode debe ser 'auto', 'daily' o 'global'."
        )

    if mode == "global":

        cumulative_volume = volume.cumsum()

        return (
            close * volume
        ).cumsum() / cumulative_volume.replace(
            0,
            np.nan
        )

    # --------------------------------------------------------
    # Detectar si los datos son intradía
    # --------------------------------------------------------

    if mode == "auto":

        if len(close.index) < 2:

            mode = "global"

        else:

            differences = (
                close.index.to_series()
                .diff()
                .dropna()
            )

            median_delta = differences.median()

            # Si la frecuencia típica es menor a un día,
            # asumimos datos intradía.
            if median_delta < pd.Timedelta(days=1):
                mode = "daily"
            else:
                mode = "global"

    if mode == "global":

        return (
            close * volume
        ).cumsum() / volume.cumsum().replace(
            0,
            np.nan
        )

    # --------------------------------------------------------
    # VWAP reiniciado diariamente
    # --------------------------------------------------------

    dates = pd.Series(
        close.index.normalize(),
        index=close.index
    )

    price_volume = (
        close *
        volume
    )

    cumulative_pv = (
        price_volume
        .groupby(dates)
        .cumsum()
    )

    cumulative_volume = (
        volume
        .groupby(dates)
        .cumsum()
    )

    return (
        cumulative_pv /
        cumulative_volume.replace(
            0,
            np.nan
        )
    )


# ============================================================
# VPVMA
# ============================================================

def calculate_vpvma(
    df: pd.DataFrame,
    window_fast: int = 12,
    window_slow: int = 26,
    window_sign: int = 9
) -> tuple:

    tp = (
        df["High"] +
        df["Low"] +
        df["Close"]
    ) / 3

    tp_x_volume = (
        tp *
        df["Volume"]
    )

    volume_fast = (
        df["Volume"]
        .rolling(
            window_fast,
            min_periods=window_fast
        )
        .sum()
    )

    volume_slow = (
        df["Volume"]
        .rolling(
            window_slow,
            min_periods=window_slow
        )
        .sum()
    )

    svwma = (
        tp_x_volume
        .rolling(
            window_fast,
            min_periods=window_fast
        )
        .sum()
        /
        volume_fast.replace(
            0,
            np.nan
        )
    )

    lvwma = (
        tp_x_volume
        .rolling(
            window_slow,
            min_periods=window_slow
        )
        .sum()
        /
        volume_slow.replace(
            0,
            np.nan
        )
    )

    dv = df[
        [
            "Open",
            "High",
            "Low",
            "Close"
        ]
    ].std(axis=1)

    svwma_x_dv = (
        svwma *
        dv
    )

    lvwma_x_dv = (
        lvwma *
        dv
    )

    esvmap = svwma_x_dv.ewm(
        span=window_fast,
        adjust=False
    ).mean()

    elvmap = lvwma_x_dv.ewm(
        span=window_slow,
        adjust=False
    ).mean()

    vpvma = (
        esvmap -
        elvmap
    )

    vpvmas = vpvma.rolling(
        window_sign,
        min_periods=window_sign
    ).mean()

    return (
        vpvma,
        vpvmas
    )


# ============================================================
# MACD DIVERGENCE
# ============================================================

def calc_macd_divergence(
    close: pd.Series,
    macd_line: pd.Series,
    left: int = 3,
    right: int = 3
):

    """
    Detecta divergencias usando pivotes confirmados.

    Importante:
    El pivote de precio se conoce únicamente después
    de que transcurren `right` velas.

    Por eso la señal se coloca en la vela de confirmación
    y no retrospectivamente en la vela del pivote.
    """

    values = close.to_numpy(
        dtype=float
    )

    length = len(values)

    bearish = pd.Series(
        False,
        index=close.index
    )

    bullish = pd.Series(
        False,
        index=close.index
    )

    if length <= left + right:
        return (
            bearish,
            bullish
        )

    highs = []
    lows = []

    for i in range(
        left,
        length - right
    ):

        window = values[
            i - left :
            i + right + 1
        ]

        current = values[i]

        if (
            not np.isnan(current) and
            current == np.nanmax(window)
        ):
            highs.append(i)

        if (
            not np.isnan(current) and
            current == np.nanmin(window)
        ):
            lows.append(i)

    # --------------------------------------------------------
    # Divergencia alcista
    # Precio: lower low
    # MACD: higher low
    # --------------------------------------------------------

    for previous, current in zip(
        lows[:-1],
        lows[1:]
    ):

        confirmation = current + right

        if confirmation >= length:
            continue

        price_condition = (
            values[current] <
            values[previous]
        )

        macd_condition = (
            macd_line.iloc[current] >
            macd_line.iloc[previous]
        )

        if (
            price_condition and
            macd_condition
        ):
            bullish.iloc[
                confirmation
            ] = True

    # --------------------------------------------------------
    # Divergencia bajista
    # Precio: higher high
    # MACD: lower high
    # --------------------------------------------------------

    for previous, current in zip(
        highs[:-1],
        highs[1:]
    ):

        confirmation = current + right

        if confirmation >= length:
            continue

        price_condition = (
            values[current] >
            values[previous]
        )

        macd_condition = (
            macd_line.iloc[current] <
            macd_line.iloc[previous]
        )

        if (
            price_condition and
            macd_condition
        ):
            bearish.iloc[
                confirmation
            ] = True

    return (
        bearish,
        bullish
    )



def evaluate_entry_quality(signal: str, bb_pos: float) -> dict:
    """
    Evalúa si el precio está en una zona ÓPTIMA para entrar o si es necesario esperar el REBOTE. 
    Se valida que no esteen la mitad del canal, sino en zonas XTREMs (superior para venta, inferior para compra).
    
    Zonas para VENDER (bb_pos):
    🟢 ZONA ÓPTIMA VENTA    (>= 0.75)      
    🟡 ZONA ACEPTABLE VENTA (0.55 – 0.75)  
    🔴 ESPERAR SUBIDA       (< 0.55)       

    Zonas para COMPRAR (bb_pos):
    🔴 ESPERAR BAJADA       (> 0.55)      
    🟡 ZONA ACEPTABLE COMPRA (0.25 – 0.45) 
    🟢 ZONA ÓPTIMA COMPRA   (<= 0.25) 
    """
    pct = bb_pos * 100 

    if -1 in signal:
        if bb_pos >= BB_ENTRY_SELL_MIN:
            return {
                "entry_ok":   True,
                "advice": -1,
                "entry_zone": -1,
                "bb_pct":     pct,
            }
        elif bb_pos >= BB_WAIT_SELL_MIN:
            return {
                "entry_ok":   False,
                "advice": "🟡", # Nivel aceptable, mejor esperar 
                "entry_zone":     (f"🟡 > {BB_ENTRY_SELL_MIN*100:.0f}%"),
                "bb_pct":     pct,
            }
        else:
            return {
                "entry_ok":   False,
                "advice": "⚪",
                "entry_zone":     (f"⚪ > {BB_ENTRY_SELL_MIN*100:.0f}%"), 
                "bb_pct":     pct,
            }

    elif 1 in signal:
        if bb_pos <= BB_ENTRY_BUY_MAX:
            return {
                "entry_ok":   True,
                "advice": 1,
                "entry_zone": 1,
                "bb_pct":     pct,
            }
        elif bb_pos <= BB_WAIT_BUY_MAX:
            return {
                "entry_ok":   False,
                "advice": "🟡",  
                "entry_zone":     (f"🟡 < {BB_ENTRY_BUY_MAX*100:.0f}%"),
                "bb_pct":     pct,
            }
        else:
            return {
                "entry_ok":   False,
                "advice": "⚪",
                "entry_zone":     (f"⚪ < {BB_ENTRY_BUY_MAX*100:.0f}%"),
                "bb_pct":     pct,
            }

    # Para ESPERAR no aplica
    return {"entry_ok": False, "entry_zone": "", "advice": "", "bb_pct": pct}


def validate_ohlcv(df: pd.DataFrame) -> None:

    required = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            f"Faltan columnas OHLCV requeridas: {missing}"
        )

    if df.empty:
        raise ValueError(
            "El DataFrame está vacío."
        )

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(
            "El índice debe ser un pandas.DatetimeIndex."
        )

    if not df.index.is_monotonic_increasing:
        raise ValueError(
            "El índice debe estar ordenado cronológicamente."
        )


def build_indicators(
    df: pd.DataFrame
) -> pd.DataFrame:

    validate_ohlcv(df)

    # --------------------------------------------------------
    # No modificar el DataFrame original
    # --------------------------------------------------------

    df = df.copy()

    df = df.sort_index()

    # --------------------------------------------------------
    # Series principales
    # --------------------------------------------------------

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # ========================================================
    # 1. EMA
    # ========================================================

    df["EMA20"] = calc_ema(
        close,
        EMA_SHORT
    )

    df["EMA50"] = calc_ema(
        close,
        EMA_MID
    )

    df["EMA200"] = calc_ema(
        close,
        EMA_LONG
    )

    # ========================================================
    # 2. RSI
    # ========================================================

    df["RSI"] = calc_rsi(
        close,
        RSI_PERIOD
    )

    # ========================================================
    # 3. MACD
    # ========================================================

    (
        df["MACD"],
        df["MACD_signal"],
        df["MACD_hist"]
    ) = calc_macd(
        close,
        MACD_FAST,
        MACD_SLOW,
        MACD_SIGNAL
    )

    # ========================================================
    # 4. Bollinger
    # ========================================================

    (
        df["BB_upper"],
        df["BB_mid"],
        df["BB_lower"]
    ) = calc_bollinger(
        close,
        BB_PERIOD,
        BB_STD
    )

    # ========================================================
    # 5. Bollinger position
    # ========================================================

    df["BB_position"] = calc_bb_position(
        df
    )

    # ========================================================
    # 6. Stochastic
    # ========================================================

    (
        df["STOCH_K"],
        df["STOCH_D"]
    ) = calc_stochastic(
        high,
        low,
        close,
        STOCH_K,
        STOCH_SLOWING,
        STOCH_D
    )

    # ========================================================
    # 7. ATR
    # ========================================================

    df["ATR"] = calc_atr(
        high,
        low,
        close,
        ATR_PERIOD
    )

    # ========================================================
    # 8. OBV
    # ========================================================

    df["OBV"] = calc_obv(
        close,
        volume
    )

    df["OBV_EMA"] = calc_ema(
        df["OBV"],
        OBV_EMA_PERIOD
    )

    # ========================================================
    # 9. ADX + DI
    # ========================================================

    (
        df["ADX"],
        df["ADX+DI"],
        df["ADX-DI"]
    ) = calc_adx(
        high,
        low,
        close,
        ADX_PERIOD
    )

    # ========================================================
    # 10. Consistencia
    # ========================================================

    (
        df["prev26"],
        df["prev52"]
    ) = calc_consistencia(
        close
    )

    # ========================================================
    # 11. VPVMA
    # ========================================================

    (
        df["VPVMA"],
        df["VPVMAS"]
    ) = calculate_vpvma(
        df,
        VPVMA_FAST,
        VPVMA_SLOW,
        VPVMA_SIGNAL
    )

    # ========================================================
    # 12. VWAP
    # ========================================================

    df["VWAP"] = calc_vwap(
        close,
        volume,
        mode=VWAP_MODE
    )

    # ========================================================
    # 13. Spread / rango relativo
    # ========================================================

    df["spread"] = (
        high -
        low
    ) / close.replace(
        0,
        np.nan
    )

    # ========================================================
    # 14. Media de volumen
    # ========================================================

    df["vol_ma"] = volume.rolling(
        20,
        min_periods=20
    ).mean()

    df["volume_ma"] = df["vol_ma"]

    df["volume_increasing"] = (
        volume >
        df["vol_ma"]
    )

    # ========================================================
    # 15. Order flow simple
    # ========================================================

    df["order_flow"] = np.where(
        close > df["Open"],
        "buy",
        np.where(
            close < df["Open"],
            "sell",
            "neutral"
        )
    )

    # ========================================================
    # 16. Volatilidad
    # ========================================================

    df["volatility"] = (
        close
        .pct_change()
        .rolling(
            VOLATILITY_PERIOD,
            min_periods=VOLATILITY_PERIOD
        )
        .std()
    )

    # ========================================================
    # 17. ROC / Momentum
    # ========================================================

    df["ROC"] = close.pct_change(
        ROC_PERIOD
    )

    # ========================================================
    # 18. CMF
    # ========================================================

    df["CMF"] = calc_cmf(
        high,
        low,
        close,
        volume,
        CMF_PERIOD
    )

    # ========================================================
    # 19. Parabolic SAR
    # ========================================================

    df["PSAR"] = calc_psar(
        high,
        low,
        PSAR_IAF,
        PSAR_MAX_AF
    )

    # ========================================================
    # 20. Williams %R
    # ========================================================

    highest_high = high.rolling(
        WILLR_PERIOD,
        min_periods=WILLR_PERIOD
    ).max()

    lowest_low = low.rolling(
        WILLR_PERIOD,
        min_periods=WILLR_PERIOD
    ).min()

    willr_denominator = (
        highest_high -
        lowest_low
    ).replace(
        0,
        np.nan
    )

    df["WillR"] = (
        -100 *
        (
            highest_high -
            close
        ) /
        willr_denominator
    )

    # ========================================================
    # 21. Donchian
    # ========================================================

    df["Donchian_high"] = high.rolling(
        DONCHIAN_PERIOD,
        min_periods=DONCHIAN_PERIOD
    ).max()

    df["Donchian_low"] = low.rolling(
        DONCHIAN_PERIOD,
        min_periods=DONCHIAN_PERIOD
    ).min()

    # ========================================================
    # 22. Distancia EMA50
    # ========================================================

    df["ema50_distance"] = (
        close -
        df["EMA50"]
    ) / df["EMA50"].replace(
        0,
        np.nan
    )

    # ========================================================
    # 23. Divergencia MACD
    # ========================================================

    (
        df["MACD_bearish_div"],
        df["MACD_bullish_div"]
    ) = calc_macd_divergence(
        close,
        df["MACD"]
    )

    df = df.iloc[48:]

    return df
