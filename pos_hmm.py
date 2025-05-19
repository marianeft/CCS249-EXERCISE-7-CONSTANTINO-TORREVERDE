import numpy as np
from collections import defaultdict

class HMM:
    def __init__(self):
        self.states = set()
        self.vocab = set()
        self.start_probs = defaultdict(float)
        self.trans_probs = defaultdict(lambda: defaultdict(float))
        self.emit_probs = defaultdict(lambda: defaultdict(float))

    def train(self, tagged_sentences):
        tag_counts = defaultdict(int)
        transition_counts = defaultdict(lambda: defaultdict(int))
        emission_counts = defaultdict(lambda: defaultdict(int))
        start_counts = defaultdict(int)

        for sentence in tagged_sentences:
            prev_tag = None
            for i, (word, tag) in enumerate(sentence):
                self.states.add(tag)
                # Convert word to lowercase to handle case variations
                word = word.lower()
                self.vocab.add(word)
                tag_counts[tag] += 1
                emission_counts[tag][word] += 1

                if i == 0:
                    start_counts[tag] += 1
                if prev_tag is not None:
                    transition_counts[prev_tag][tag] += 1
                prev_tag = tag

        # Calculate probabilities
        total_starts = sum(start_counts.values())
        for tag in self.states:
            # Start probabilities
            self.start_probs[tag] = start_counts[tag] / total_starts if total_starts > 0 else 0.0
            
            # Transition probabilities
            total_transitions_from_tag = tag_counts[tag] # Sum of all transitions *from* this tag
            for next_tag in self.states:
                 # Add 1 for Laplace smoothing (add-one smoothing)
                numerator = transition_counts[tag][next_tag] + 1
                # Denominator is total transitions from tag + number of possible next states (len(self.states))
                denominator = total_transitions_from_tag + len(self.states) 
                self.trans_probs[tag][next_tag] = numerator / denominator

            # Emission probabilities
            total_emissions_from_tag = tag_counts[tag] # Sum of all emissions *from* this tag (count of the tag)
            for word in self.vocab:
                 # Add 1 for Laplace smoothing (add-one smoothing)
                numerator = emission_counts[tag][word] + 1
                # Denominator is total emissions from tag + number of possible words (len(self.vocab))
                denominator = total_emissions_from_tag + len(self.vocab)
                self.emit_probs[tag][word] = numerator / denominator
                
            # Handle words seen during training but not during testing (smoothing already helps)
            # and unseen words during testing (handled in viterbi)


    def viterbi(self, sentence):
        # Convert sentence words to lowercase for consistency
        sentence = [word.lower() for word in sentence]
        
        V = [{}]
        path = {}

        # Handle potential empty states or vocab after training
        if not self.states or not self.vocab:
             return ["UNK"] * len(sentence) # Cannot tag if no states/vocab learned

        # Initialize for the first word
        for tag in self.states:
            # Get start probability (use small value for unseen start tags if needed, though smoothing in train helps)
            start_prob = self.start_probs.get(tag, 1e-6) 
            # Get emission probability (use small value for unseen words)
            emit_prob = self.emit_probs[tag].get(sentence[0], 1e-6) 
            
            V[0][tag] = start_prob * emit_prob
            path[tag] = [tag]

        # Run Viterbi for t > 0
        for t in range(1, len(sentence)):
            V.append({})
            new_path = {}

            # Handle potential unseen word in the sentence
            current_word = sentence[t]

            for curr_tag in self.states:
                max_prob = -float('inf')
                best_prev_tag = None

                # Emission probability for the current word and current tag
                # Use small value for unseen words
                emit_prob = self.emit_probs[curr_tag].get(current_word, 1e-6) 

                for prev_tag in self.states:
                    # Get transition probability (use small value for unseen transitions)
                    trans_prob = self.trans_probs[prev_tag].get(curr_tag, 1e-6)
                    
                    # Calculate probability of path ending in prev_tag then transitioning to curr_tag
                    current_prob = V[t - 1].get(prev_tag, 0) * trans_prob * emit_prob
                    
                    if current_prob > max_prob:
                        max_prob = current_prob
                        best_prev_tag = prev_tag

                V[t][curr_tag] = max_prob
                
                # Reconstruct path
                if best_prev_tag is not None:
                    new_path[curr_tag] = path[best_prev_tag] + [curr_tag]
                else:
                    # If no path found (extremely low probability), maybe fallback or use a default
                    # This case is unlikely with smoothing and small prob defaults
                    new_path[curr_tag] = [curr_tag] # Fallback: just assign the current tag

            path = new_path

        # Find the best path
        max_final_prob = -float('inf')
        best_final_tag = None

        if not V[-1]: # Handle case where V[-1] empty 
             return ["UNK"] * len(sentence)

        for tag, prob in V[-1].items():
            if prob > max_final_prob:
                max_final_prob = prob
                best_final_tag = tag

        # Return best tag sequence
        if best_final_tag is not None and best_final_tag in path:
            return path[best_final_tag]
        else:
             return ["UNK"] * len(sentence) # Fallback if best tag not found in path 