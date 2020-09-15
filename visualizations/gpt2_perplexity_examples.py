from search_utils.Sentence import Sentence

for s in [
    "Python is a language programming.",
    "Python is a porgramming langauge.",
    "Python is a programming language.",

    "The plane is in the air.",
    "The plane is in the water.",
    "The boat is in the air.",
    "The boat is in the water.",

    "David Bowie is a famous musician.",
    "David Cameron is a famous musician.",
    "David Bowie is a musician.",
    "David Cameron is a musician.",

    "David Bowie was the prime minister.",
    "David Cameron was the prime minister.",
]:
    print(f"{Sentence(s).calc_perplexity():.2f} {s}")
