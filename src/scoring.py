from data_handler import Relevance, Sentence

def score_function(model_answer: list, sentences: Sentence) -> float:
    score: float = 0.0
    try:
        answer_numbers = set(map(int, model_answer.split()))
        for sentence in sentences:
            if int(sentence.sentence_id) in answer_numbers:
                match sentence.relevance:
                    case Relevance.ESSENTIAL:
                        score += 1
                    case Relevance.NOT_RELEVANT:
                        score -= 1
                    case _:
                        pass
            elif sentence.relevance == Relevance.ESSENTIAL:
                score -= 1
        return score
    except Exception:
        pass
    return score

from sklearn.metrics import precision_score, recall_score, f1_score as sk_f1_score

def f1_score(y_true: list, y_pred: list, 
                      essential_label: str = 'essential', 
                      not_relevant_label: str = 'not-relevant'):
    # If y_pred contains indices (integers), convert them to labels
    if isinstance(y_pred[0], int):
        converted = [not_relevant_label] * len(y_true)
        for idx in y_pred:
            converted[idx] = essential_label
        y_pred = converted

    # Map any label not equal to essential_label or not_relevant_label to not_relevant_label
    y_true = [label if label in [essential_label, not_relevant_label] else not_relevant_label for label in y_true]
    y_pred = [label if label in [essential_label, not_relevant_label] else not_relevant_label for label in y_pred]

    # Calculate binary metrics for the essential_label
    f1 = sk_f1_score(y_true, y_pred, pos_label=essential_label, average='binary')
    precision = precision_score(y_true, y_pred, pos_label=essential_label, average='binary')
    recall = recall_score(y_true, y_pred, pos_label=essential_label, average='binary')

    return f1, precision, recall

# Example usage
if __name__=='__main__':
    # Example with explicit label predictions
    y_true = ['essential', 'not-relevant', 'supplementary']
    y_pred = ['essential', 'not-relevant', 'essential']
    f1, precision, recall = f1_score(y_true, y_pred)
    print("F1:", f1, "Precision:", precision, "Recall:", recall)
    
    # Example with sentence index predictions (indices of sentences to mark as essential)
    y_pred = [0, 2]  # Sentences at index 0 and 2 are 'essential'; rest are 'not-relevant'
    f1, precision, recall = f1_score(y_true, y_pred)
    print("F1:", f1, "Precision:", precision, "Recall:", recall)