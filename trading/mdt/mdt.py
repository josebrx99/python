import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))

from indicators.indicators import load_yahoo, build_indicators
from strategies.signals import build_signals

def load_symbol(SYMBOL, INTERVAL, LOOKBACK_DAYS):
    df = load_yahoo(SYMBOL, INTERVAL, LOOKBACK_DAYS) 
    df = build_indicators(df)
    df = build_signals(df)

    return df


def load_symbol_paral(SYMBOLS, INTERVAL, LOOKBACK_DAYS):

    print(f"Se analizan {len(SYMBOLS)} símbolos")
    print(f"Intervalo: {INTERVAL}  |  Lookback: {LOOKBACK_DAYS} días")

    output = []
    chunk_size = 5

    for i in range(0, len(SYMBOLS), chunk_size):

        chunk = SYMBOLS[i:i + chunk_size]

        chunk_results = Parallel(n_jobs=-1)(
            delayed(load_symbol)(sym, INTERVAL, LOOKBACK_DAYS)
            for sym in chunk
        )

        print(f"Chunk {i // chunk_size + 1}")

        output.extend(chunk_results)

    valid_rows = [
        row
        for row in output
        if row is not None
        and len(row) == 2
        and row[1] is not None
    ]

    df = pd.concat([row[1] for row in valid_rows], ignore_index=True)

    return df