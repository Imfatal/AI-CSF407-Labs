import random
from collections import defaultdict, Counter


# ============================================================
# PART III: BUILD THE DATASET
# ============================================================

sentences = [
    "the cat sat on the mat",
    "the cat sat on the rug",
    "the dog sat on the mat",
    "the dog ran to the park",
    "the cat ran to the park",
    "the dog sat on the rug"
]

# Convert everything to lowercase and add special tokens
tokenized_sentences = []

for sentence in sentences:
    tokens = sentence.lower().split()
    tokens = ["<START>"] + tokens + ["<END>"]
    tokenized_sentences.append(tokens)

print("Tokenized dataset:")
for sentence in tokenized_sentences:
    print(sentence)


# ============================================================
# PART IV / V:
# FIRST-ORDER AUTOREGRESSIVE LANGUAGE MODEL
#
# Model:
#       P(X_t | X_{t-1})
# ============================================================

class FirstOrderLanguageModel:

    def __init__(self):
        # counts[current_word][next_word] = number of transitions
        self.counts = defaultdict(Counter)

        # probabilities[current_word][next_word] = probability
        self.probabilities = defaultdict(dict)

    # --------------------------------------------------------
    # Train the model
    # --------------------------------------------------------
    def train(self, sentences):

        for sentence in sentences:

            for i in range(len(sentence) - 1):

                current_word = sentence[i]
                next_word = sentence[i + 1]

                self.counts[current_word][next_word] += 1

        self._calculate_probabilities()

    # --------------------------------------------------------
    # Calculate:
    #
    # P(next_word | current_word)
    #
    # = count(current_word, next_word)
    #   --------------------------------
    #       total transitions from current_word
    # --------------------------------------------------------
    def _calculate_probabilities(self):

        for current_word in self.counts:

            total = sum(self.counts[current_word].values())

            for next_word in self.counts[current_word]:

                probability = (
                    self.counts[current_word][next_word] / total
                )

                self.probabilities[current_word][next_word] = probability

    # --------------------------------------------------------
    # Display transition counts
    # --------------------------------------------------------
    def display_counts(self):

        print("\nTransition Counts:")

        for current_word in self.counts:

            print(f"\n{current_word}:")

            for next_word, count in self.counts[current_word].items():

                print(f"    {next_word}: {count}")

    # --------------------------------------------------------
    # Display conditional probability table
    # --------------------------------------------------------
    def display_probabilities(self):

        print("\nConditional Probability Table:")

        for current_word in self.probabilities:

            print(f"\nP(next word | {current_word})")

            for next_word, probability in self.probabilities[
                current_word
            ].items():

                print(
                    f"    P({next_word} | {current_word}) "
                    f"= {probability:.4f}"
                )

    # --------------------------------------------------------
    # Get P(next_word | current_word)
    # --------------------------------------------------------
    def get_distribution(self, current_word):

        return self.probabilities.get(current_word, {})

    # --------------------------------------------------------
    # Greedy prediction
    #
    # argmax_w P(w | current_word)
    # --------------------------------------------------------
    def predict_greedy(self, current_word):

        distribution = self.get_distribution(current_word)

        if not distribution:
            return None

        return max(
            distribution,
            key=distribution.get
        )

    # --------------------------------------------------------
    # Sampling prediction
    #
    # Sample from P(w | current_word)
    # --------------------------------------------------------
    def predict_sample(self, current_word):

        distribution = self.get_distribution(current_word)

        if not distribution:
            return None

        words = list(distribution.keys())
        probabilities = list(distribution.values())

        return random.choices(
            words,
            weights=probabilities,
            k=1
        )[0]

    # --------------------------------------------------------
    # Generate sentence using greedy method
    # --------------------------------------------------------
    def generate_greedy(self, max_length=20):

        current_word = "<START>"
        generated = []

        for _ in range(max_length):

            next_word = self.predict_greedy(current_word)

            if next_word is None:
                break

            if next_word == "<END>":
                break

            generated.append(next_word)

            current_word = next_word

        return " ".join(generated)

    # --------------------------------------------------------
    # Generate sentence using sampling
    # --------------------------------------------------------
    def generate_sample(self, max_length=20):

        current_word = "<START>"
        generated = []

        for _ in range(max_length):

            next_word = self.predict_sample(current_word)

            if next_word is None:
                break

            if next_word == "<END>":
                break

            generated.append(next_word)

            current_word = next_word

        return " ".join(generated)

    # --------------------------------------------------------
    # Check probability normalization
    #
    # sum_v P(v | w) = 1
    # --------------------------------------------------------
    def check_normalization(self):

        print("\nProbability Normalization:")

        all_correct = True

        for current_word in self.probabilities:

            total = sum(
                self.probabilities[current_word].values()
            )

            print(
                f"{current_word}: "
                f"{total:.6f}"
            )

            if abs(total - 1.0) > 1e-9:
                all_correct = False

        return all_correct


# ============================================================
# TRAIN FIRST-ORDER MODEL
# ============================================================

first_order_model = FirstOrderLanguageModel()

first_order_model.train(tokenized_sentences)


# ============================================================
# DISPLAY COUNTS
# ============================================================

first_order_model.display_counts()


# ============================================================
# DISPLAY CONDITIONAL PROBABILITY TABLE
# ============================================================

first_order_model.display_probabilities()


# ============================================================
# PART VIII:
# PREDICTING THE NEXT WORD
# ============================================================

print("\n\nNext-word distributions:")

words_to_test = [
    "the",
    "cat",
    "dog",
    "sat",
    "ran"
]

for word in words_to_test:

    distribution = first_order_model.get_distribution(word)

    print(f"\nP(next word | {word})")

    for next_word, probability in distribution.items():

        print(
            f"    {next_word}: "
            f"{probability:.4f}"
        )

    prediction = first_order_model.predict_greedy(word)

    print(
        f"Most probable next word: {prediction}"
    )


# ============================================================
# PART VII:
# TEST PROBABILITY NORMALIZATION
# ============================================================

print("\n\nNormalization test:")

if first_order_model.check_normalization():

    print("\nAll probability distributions are normalized.")

else:

    print("\nERROR: Some distributions are not normalized.")


# ============================================================
# PART IX:
# GENERATE AT LEAST 20 SENTENCES
# ============================================================

print("\n\n20 Generated Sentences:")
print("----------------------------------------")

for i in range(20):

    sentence = first_order_model.generate_sample()

    print(f"{i + 1}. {sentence}")


# ============================================================
# PART X:
# GREEDY VS SAMPLING
# ============================================================

print("\n\nGREEDY GENERATION")
print("----------------------------------------")

for i in range(5):

    sentence = first_order_model.generate_greedy()

    print(f"{i + 1}. {sentence}")


print("\n\nSAMPLING GENERATION")
print("----------------------------------------")

for i in range(5):

    sentence = first_order_model.generate_sample()

    print(f"{i + 1}. {sentence}")


# ============================================================
# PART XI:
# SECOND-ORDER AUTOREGRESSIVE MODEL
#
# Model:
#
# P(X_t | X_{t-2}, X_{t-1})
# ============================================================

class SecondOrderLanguageModel:

    def __init__(self):

        # counts[(word1, word2)][next_word]
        self.counts = defaultdict(Counter)

        # probabilities[(word1, word2)][next_word]
        self.probabilities = defaultdict(dict)

    # --------------------------------------------------------
    # Train second-order model
    # --------------------------------------------------------
    def train(self, sentences):

        for sentence in sentences:

            # Need at least two previous tokens
            for i in range(2, len(sentence)):

                previous_two = (
                    sentence[i - 2],
                    sentence[i - 1]
                )

                next_word = sentence[i]

                self.counts[previous_two][next_word] += 1

        self._calculate_probabilities()

    # --------------------------------------------------------
    # Calculate:
    #
    # P(next | previous_word_1, previous_word_2)
    # --------------------------------------------------------
    def _calculate_probabilities(self):

        for context in self.counts:

            total = sum(
                self.counts[context].values()
            )

            for next_word in self.counts[context]:

                probability = (
                    self.counts[context][next_word] / total
                )

                self.probabilities[context][next_word] = (
                    probability
                )

    # --------------------------------------------------------
    # Display transition counts
    # --------------------------------------------------------
    def display_counts(self):

        print("\nSecond-Order Transition Counts:")

        for context in self.counts:

            print(f"\n{context}:")

            for next_word, count in self.counts[context].items():

                print(
                    f"    {next_word}: {count}"
                )

    # --------------------------------------------------------
    # Display conditional probability table
    # --------------------------------------------------------
    def display_probabilities(self):

        print("\nSecond-Order Conditional Probability Table:")

        for context in self.probabilities:

            print(
                f"\nP(next word | "
                f"{context[0]}, {context[1]})"
            )

            for next_word, probability in self.probabilities[
                context
            ].items():

                print(
                    f"    P({next_word} | "
                    f"{context[0]}, {context[1]}) "
                    f"= {probability:.4f}"
                )

    # --------------------------------------------------------
    # Get distribution
    # --------------------------------------------------------
    def get_distribution(self, context):

        return self.probabilities.get(context, {})

    # --------------------------------------------------------
    # Greedy prediction
    # --------------------------------------------------------
    def predict_greedy(self, context):

        distribution = self.get_distribution(context)

        if not distribution:
            return None

        return max(
            distribution,
            key=distribution.get
        )

    # --------------------------------------------------------
    # Sampling prediction
    # --------------------------------------------------------
    def predict_sample(self, context):

        distribution = self.get_distribution(context)

        if not distribution:
            return None

        words = list(distribution.keys())
        probabilities = list(distribution.values())

        return random.choices(
            words,
            weights=probabilities,
            k=1
        )[0]

        # --------------------------------------------------------
    # Generate using greedy method
    # --------------------------------------------------------
    def generate_greedy(self, max_length=20):

        previous_word_1 = "<START>"
        previous_word_2 = "the"

        generated = ["the"]

        for _ in range(max_length - 1):

            context = (
                previous_word_1,
                previous_word_2
            )

            next_word = self.predict_greedy(context)

            if next_word is None:
                break

            if next_word == "<END>":
                break

            generated.append(next_word)

            previous_word_1 = previous_word_2
            previous_word_2 = next_word

        return " ".join(generated)

    # --------------------------------------------------------
    # Generate using sampling
    # --------------------------------------------------------
    def generate_sample(self, max_length=20):

        previous_word_1 = "<START>"
        previous_word_2 = "the"

        generated = ["the"]

        for _ in range(max_length - 1):

            context = (
                previous_word_1,
                previous_word_2
            )

            next_word = self.predict_sample(context)

            if next_word is None:
                break

            if next_word == "<END>":
                break

            generated.append(next_word)

            previous_word_1 = previous_word_2
            previous_word_2 = next_word

        return " ".join(generated)

    # --------------------------------------------------------
    # Probability normalization test
    # --------------------------------------------------------
    def check_normalization(self):

        print("\nSecond-Order Probability Normalization:")

        all_correct = True

        for context in self.probabilities:

            total = sum(
                self.probabilities[context].values()
            )

            print(
                f"{context}: "
                f"{total:.6f}"
            )

            if abs(total - 1.0) > 1e-9:

                all_correct = False

        return all_correct


# ============================================================
# TRAIN SECOND-ORDER MODEL
# ============================================================

second_order_model = SecondOrderLanguageModel()

second_order_model.train(tokenized_sentences)


# ============================================================
# DISPLAY SECOND-ORDER COUNTS
# ============================================================

second_order_model.display_counts()


# ============================================================
# DISPLAY SECOND-ORDER PROBABILITIES
# ============================================================

second_order_model.display_probabilities()


# ============================================================
# SECOND-ORDER NORMALIZATION TEST
# ============================================================

print("\n\nSecond-order normalization test:")

if second_order_model.check_normalization():

    print(
        "\nAll second-order probability "
        "distributions are normalized."
    )

else:

    print(
        "\nERROR: Some second-order "
        "distributions are not normalized."
    )


# ============================================================
# SECOND-ORDER TEXT GENERATION
# ============================================================

print("\n\nSECOND-ORDER GREEDY GENERATION")
print("----------------------------------------")

for i in range(5):

    sentence = second_order_model.generate_greedy()

    print(f"{i + 1}. {sentence}")


print("\n\nSECOND-ORDER SAMPLING GENERATION")
print("----------------------------------------")

for i in range(5):

    sentence = second_order_model.generate_sample()

    print(f"{i + 1}. {sentence}")


# ============================================================
# PART XIII:
# COMPARE FIRST-ORDER AND SECOND-ORDER MODELS
# ============================================================

print("\n\nMODEL COMPARISON")
print("========================================")


# ------------------------------------------------------------
# Number of distinct parameters
# ------------------------------------------------------------

first_order_parameters = sum(
    len(first_order_model.probabilities[context])
    for context in first_order_model.probabilities
)

second_order_parameters = sum(
    len(second_order_model.probabilities[context])
    for context in second_order_model.probabilities
)

print(
    f"\nFirst-order distinct probabilities: "
    f"{first_order_parameters}"
)

print(
    f"Second-order distinct probabilities: "
    f"{second_order_parameters}"
)


# ------------------------------------------------------------
# Number of contexts
# ------------------------------------------------------------

first_order_contexts = len(
    first_order_model.probabilities
)

second_order_contexts = len(
    second_order_model.probabilities
)

print(
    f"\nFirst-order contexts: "
    f"{first_order_contexts}"
)

print(
    f"Second-order contexts: "
    f"{second_order_contexts}"
)


# ------------------------------------------------------------
# Zero-probability contexts
#
# We consider possible contexts from the vocabulary.
# ------------------------------------------------------------

vocabulary = set()

for sentence in tokenized_sentences:

    for word in sentence:

        vocabulary.add(word)


# First-order possible contexts
all_first_order_contexts = vocabulary

zero_first_order_contexts = 0

for word in all_first_order_contexts:

    if word not in first_order_model.probabilities:

        zero_first_order_contexts += 1


# Second-order possible contexts
all_second_order_contexts = []

for word1 in vocabulary:

    for word2 in vocabulary:

        all_second_order_contexts.append(
            (word1, word2)
        )


zero_second_order_contexts = 0

for context in all_second_order_contexts:

    if context not in second_order_model.probabilities:

        zero_second_order_contexts += 1


print(
    f"\nFirst-order zero-probability contexts: "
    f"{zero_first_order_contexts}"
)

print(
    f"Second-order zero-probability contexts: "
    f"{zero_second_order_contexts}"
)


# ============================================================
# GENERATE EXAMPLES FOR QUALITATIVE COMPARISON
# ============================================================

print("\n\nQUALITATIVE COMPARISON")
print("========================================")

print("\nFirst-order examples:")

for i in range(5):

    print(
        f"{i + 1}. "
        f"{first_order_model.generate_sample()}"
    )


print("\nSecond-order examples:")

for i in range(5):

    print(
        f"{i + 1}. "
        f"{second_order_model.generate_sample()}"
    )


# ============================================================
# END OF LAB
# ============================================================