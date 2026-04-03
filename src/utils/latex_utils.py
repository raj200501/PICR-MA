def to_pm(mean,se,prec=3):
    return f"{mean:.{prec}f} $\pm$ {se:.{prec}f}"
