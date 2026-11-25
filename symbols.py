"""CNN과 LSTM이 공유하는 기호 확률 열 순서."""

CLASSES = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz") + [
    "+", "-", "=", r"\times", "/", "(", ")", ".", "[", "]",
    r"\sqrt", r"\pm", r"\cdot", r"\approx", r"\circ",
    r"\Delta", r"\theta", r"\lambda", r"\mu", r"\rho", r"\pi", r"\omega",
    r"\alpha", r"\beta", r"\gamma", r"\epsilon", r"\sigma", r"\tau", r"\phi", r"\nu",
]
