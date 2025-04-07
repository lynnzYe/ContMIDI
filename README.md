# Brief

All currently available MIDI encoding methods encodes duration as discrete tokens.
However, this usually requires a large vocabulary. Besides, they does not reflect the ordinal nature
of duration values.

This repository uses a vanilla BERT model to experiment with continuous embeddings for duration/velocity's performance
on MIDI reconstruction task. It is based on the repo MaskedExpressiveness. 