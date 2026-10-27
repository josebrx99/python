import pandas as pd
import numpy as np



def vc(variable):
    vc = variable.value_counts(normalize = False).reset_index()
    vc["porc"] = (vc["count"] / vc["count"].sum())*100
    return vc