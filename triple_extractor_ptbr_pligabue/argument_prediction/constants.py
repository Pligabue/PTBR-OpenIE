ACCEPTANCE_THRESHOLD = 0.15
N_HEADS = 1

ALWAYS_REMOVE = ["#"]
ARTICLES = ["o", "a", "os", "as", "um", "uma", "uns", "umas"]
PREPOSITIONS = [
    "a", "à", "ao", "às", "aos", "ante", "após", "até", "com", "contra", "de", "do", "da", "dos", "das",
    "desde", "em", "na", "no", "entre", "para", "perante", "por", "pra", "pro", "pras", "pros",
    "pela", "pelo", "pelas", "pelos", "sem", "sob", "sobre", "trás"
]
PERSONAL_PRONOUNS = ["eu", "tu", "ele", "ela", "nós", "vós", "eles", "elas"]
RELATIVE_PRONOUNS = [
    "que", "qual", "quais", "quem", "cujo", "cujos", "cuja", "cujas", "onde", "quanto", "quanta", "quantos", "quantas",
]
DEMONSTRATIVE_PRONOUNS = [
    "este", "esta", "estes", "estas", "isto", "esse", "essa", "esses", "essas", "isso", "aquele", "aquela", "aqueles",
    "aquelas", "aquilo",
]
PRONOUNS = PERSONAL_PRONOUNS + RELATIVE_PRONOUNS + DEMONSTRATIVE_PRONOUNS
CONJUNCTIONS = [
    "e", "ainda", "mas", "também", "como", "quanto", "ou", "ora", "quer", "talvez", "nem",
    "mas", "porém", "senão", "entretanto", "contudo", "então", "portanto", "logo", "pois", "assim",
]

STRIP_FROM_START = ARTICLES + PREPOSITIONS + PRONOUNS + CONJUNCTIONS
STRIP_FROM_END = ARTICLES + PREPOSITIONS + PRONOUNS + CONJUNCTIONS
