ACCEPTANCE_THRESHOLD = 0.15
N_HEADS = 1

ALWAYS_REMOVE = ["#"]
ARTICLES = ["o", "a", "os", "as", "um", "uma", "uns", "umas"]
PREPOSITIONS = [
    "a", "à", "ao", "às", "aos", "ante", "após", "até", "com", "contra", "de", "do", "da", "dos", "das",
    "desde", "em", "na", "no", "entre", "para", "perante", "por", "pra", "pro", "pras", "pros",
    "pela", "pelo", "pelas", "pelos", "sem", "sob", "sobre", "trás"
]
STRIP_FROM_START = ARTICLES + PREPOSITIONS
STRIP_FROM_END = ARTICLES + PREPOSITIONS
