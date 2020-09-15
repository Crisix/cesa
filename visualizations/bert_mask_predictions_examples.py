from search_utils.Sentence import Sentence

Sentence("David [MASK] is a spanish tennis player.").calc_mask_predictions()  # ferrer
Sentence("David [MASK] is a famous musician.").calc_mask_predictions()  # bowie
Sentence("David [MASK] is a musician.").calc_mask_predictions()  # smith

Sentence("[MASK] is a musician.").calc_mask_predictions()  # he

Sentence("David [MASK] was the prime minister.").calc_mask_predictions()  # cameron
Sentence("David [MASK] is the prime minister.").calc_mask_predictions()  # cameron

Sentence("It's over Anakin! I have the [MASK] ground!").calc_mask_predictions()  # high
Sentence("It's over Anakin! I have the high [MASK]!").calc_mask_predictions()  # ground
Sentence("It's over Anakin! I [MASK] the high ground!").calc_mask_predictions()  # need

Sentence("Boy, that [MASK] quickly!").calc_mask_predictions()  # was, happened, moved
Sentence("The man who passes the sentence should swing the [MASK].").calc_mask_predictions()  # chair, bail, stick, wheel
Sentence("A Lannister always pays his [MASK].").calc_mask_predictions()  # debts, taxes, way
Sentence("This is the [MASK].").calc_mask_predictions()  # end, truth, way
Sentence("That's what I do: I [MASK] and I know things.").calc_mask_predictions()  # see, think, look

Sentence("The [MASK] will be with you").calc_mask_predictions()  # girls
Sentence("The [MASK] will be with you.").calc_mask_predictions()  # wolf, others, power, rest, lord

Sentence("[MASK] is a programming language").calc_mask_predictions()  # ruby, c, python, java, pascal, basic, r, php,
Sentence("[MASK] is a programming language.").calc_mask_predictions()  # ruby, python, c, ml, java, it, basic, php, pascal
