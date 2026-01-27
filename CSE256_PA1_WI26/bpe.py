# bpe.py
from collections import Counter, defaultdict


class BytePairEncoder:
    """
    Minimal BPE implementation:
      - Train merges on word types from train corpus
      - Encode words by applying merges in learned order
    """

    def __init__(self, merges=None):
        self.merges = merges or []          # list of tuple(pairA, pairB) in order
        self.merge_rank = {m: i for i, m in enumerate(self.merges)}

    @staticmethod
    def _word_to_symbols(word: str):
        # standard BPE trick: end-of-word marker to keep word boundaries
        return tuple(list(word) + ["</w>"])

    @staticmethod
    def _get_pair_counts(vocab_counter):
        """
        vocab_counter: Counter{ tuple(symbols): freq }
        Return counts of adjacent symbol pairs across all word types.
        """
        pair_counts = Counter()
        for symbols, freq in vocab_counter.items():
            for i in range(len(symbols) - 1):
                pair_counts[(symbols[i], symbols[i + 1])] += freq
        return pair_counts

    @staticmethod
    def _merge_pair_in_word(symbols, pair):
        """
        Replace occurrences of pair (a,b) with merged token "ab"
        """
        a, b = pair
        merged = a + b
        new_syms = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                new_syms.append(merged)
                i += 2
            else:
                new_syms.append(symbols[i])
                i += 1
        return tuple(new_syms)

    def train(self, words, vocab_size=8000):
        """
        words: iterable of word strings from train set
        vocab_size: target number of distinct symbols (subwords) INCLUDING '</w>' but excluding PAD/UNK
        """
        # build word type counts
        word_counts = Counter(words)

        # initialize as character vocab with </w>
        vocab_counter = Counter()
        for w, c in word_counts.items():
            vocab_counter[self._word_to_symbols(w)] += c

        # initial symbol set
        symbols_set = set()
        for syms in vocab_counter:
            symbols_set.update(syms)

        merges = []

        # Keep merging until symbol vocab reaches target or no pairs
        while len(symbols_set) < vocab_size:
            pair_counts = self._get_pair_counts(vocab_counter)
            if not pair_counts:
                break

            best_pair, best_count = pair_counts.most_common(1)[0]
            if best_count < 2:
                break

            merges.append(best_pair)

            # apply merge to all word types
            new_vocab_counter = Counter()
            for syms, freq in vocab_counter.items():
                new_syms = self._merge_pair_in_word(syms, best_pair)
                new_vocab_counter[new_syms] += freq
            vocab_counter = new_vocab_counter

            # update symbol set
            symbols_set = set()
            for syms in vocab_counter:
                symbols_set.update(syms)

        self.merges = merges
        self.merge_rank = {m: i for i, m in enumerate(self.merges)}

    def encode_word(self, word: str):
        """
        Apply merges in order to a single word -> list of subword tokens (without </w>)
        """
        symbols = list(self._word_to_symbols(word))

        # apply merges in training order (simple + clear, not super optimized)
        for pair in self.merges:
            symbols = list(self._merge_pair_in_word(tuple(symbols), pair))

        # remove end marker (either as '</w>' or merged into token ending with '</w>')
        out = []
        for s in symbols:
            if s == "</w>":
                continue
            if s.endswith("</w>"):
                out.append(s[:-4])  # strip '</w>'
            else:
                out.append(s)
        return out

    def encode_sentence(self, words):
        """
        words: list[str]
        return: list[str] subword tokens
        """
        toks = []
        for w in words:
            toks.extend(self.encode_word(w))
        return toks