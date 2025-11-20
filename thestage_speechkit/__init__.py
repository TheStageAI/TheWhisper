import transformers
import numpy as np

def _find_longest_common_sequence(sequences, token_timestamp_sequences=None):
    # It would be much harder to do O(n) because of fault tolerance.
    # We actually have a really good property which is that the total sequence
    # MUST be those subsequences in order.
    # If token_timestamp_sequences is provided, will split those sequences in
    # exactly the same way.

    left_sequence = sequences[0]
    left_length = len(left_sequence)
    total_sequence = []

    if token_timestamp_sequences:
        left_token_timestamp_sequence = token_timestamp_sequences[0]
        total_token_timestamp_sequence = []

    for seq_idx, right_sequence in enumerate(sequences[1:]):
        # index = 0
        max_ = 0.0
        max_indices = (left_length, left_length, 0, 0)
        # Here we're sliding matches
        # [a, b, c, d]
        #          [c, d, f]
        # =        [c] == [d]
        #
        # [a, b, c, d]
        #       [c, d, f]
        # =     [c, d] == [c, d]
        #
        #
        # [a, b, c, d]
        #    [c, d, f]
        #
        # =  [b, c, d] == [c, d, f]
        #
        # [a, b, c, d]
        # [c, d, f]
        #
        # [a, b, c] == [c, d, f]
        #
        # [a, b, c, d]
        # [d, f]
        #
        # [a, b] == [d, f]
        #
        # [a, b, c, d]
        # [f]
        #
        # [a] == [f]
        right_length = len(right_sequence)
        for i in range(1, left_length + right_length):
            # epsilon to favor long perfect matches
            eps = i / 10000.0

            # Slightly convoluted because we don't want out of bound indices
            # This will be necessary for a small conflict resolution optimization
            # later
            left_start = max(0, left_length - i)
            left_stop = min(left_length, left_length + right_length - i)
            left = np.array(left_sequence[left_start:left_stop])

            right_start = max(0, i - left_length)
            right_stop = min(right_length, i)
            right = np.array(right_sequence[right_start:right_stop])

            # We can only match subsequences of the same size.
            if len(left) != len(right):
                raise RuntimeError(
                    "There is a bug within whisper `decode_asr` function, please report it. Dropping to prevent bad inference."
                )

            def compare(left, right):
                if left[1] is None:
                    return True
                return left <= right

            if token_timestamp_sequences:
                # Get length of longest subsequence of tokens that match
                # and have timestamps that are in order
                matches = sum(
                    1
                    for idx, elem in enumerate(left)
                    if (
                        elem == right[idx]
                        and compare(left_token_timestamp_sequence[left_start + idx], token_timestamp_sequences[seq_idx + 1][right_start + idx])
                        # and left_token_timestamp_sequence[left_start + idx] <= token_timestamp_sequences[seq_idx + 1][right_start + idx]
                    )
                )

            else:
                matches = np.sum(left == right)

            matching = matches / i + eps
            if matches > 1 and matching > max_:
                max_ = matching
                max_indices = (left_start, left_stop, right_start, right_stop)

        (left_start, left_stop, right_start, right_stop) = max_indices

        # This is a small conflict optimization since those sequences overlap
        # in audio.
        # We're going to give more confidence to the left sequence
        # for the left of the overlap,
        # and to the right of the sequence, for the right of the overlap
        left_mid = (left_stop + left_start) // 2
        right_mid = (right_stop + right_start) // 2
        total_sequence.extend(left_sequence[:left_mid])
        left_sequence = right_sequence[right_mid:]
        left_length = len(left_sequence)

        if token_timestamp_sequences:
            total_token_timestamp_sequence.extend(left_token_timestamp_sequence[:left_mid])
            left_token_timestamp_sequence = token_timestamp_sequences[seq_idx + 1][right_mid:]

    total_sequence.extend(left_sequence)

    if token_timestamp_sequences is None:
        return total_sequence

    if len(token_timestamp_sequences) > 0:
        total_token_timestamp_sequence.extend(left_token_timestamp_sequence)
        return total_sequence, total_token_timestamp_sequence
    else:
        return total_sequence, []


transformers.models.whisper.tokenization_whisper._find_longest_common_sequence = _find_longest_common_sequence
