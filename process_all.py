from utils.preprocessing import preprocess_all_subject

if __name__ == "__main__":
    preprocess_all_subject(subject_ids=list(range(1, 10)), concatenate=False)   
    