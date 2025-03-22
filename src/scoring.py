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