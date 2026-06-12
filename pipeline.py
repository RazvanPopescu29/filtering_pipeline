import os
import threading
import psutil
import time
import csv
import time
import hashlib
import gc
import difflib
from functools import lru_cache
import multiprocessing
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import fasttext
from pybloom_live import BloomFilter
from datasketch import MinHash, MinHashLSH
from wordfreq import zipf_frequency
import numpy as np

def resource_monitor(stop_event, log_file="hardware_usage.csv"):
    print("Started Hardware Monitor")
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Time_Seconds", "CPU_Percent", "RAM_GB"])
        
        start_time = time.time()
        while not stop_event.is_set():
            current_time = round(time.time() - start_time, 1)
            cpu = psutil.cpu_percent(interval=1)
            ram = psutil.virtual_memory().used / (1024 ** 3) 
            
            writer.writerow([current_time, cpu, round(ram, 2)])

CONFIG = {
    "batch_size": 100_000,
    "fasttext_min_conf": 0.6,
    "min_english_ratio": 0.6,
    "min_words": 5,
    "lsh_threshold": 0.85,
    "num_perm": 128,
    "bloom_capacity": 50_000_000,
    "bloom_error": 0.001,
    "min_unique_word_ratio": 0.25,  
    "min_alpha_ratio": 0.4,         
}

global_ft_model = None

def init_worker(fasttext_model_path):
    global global_ft_model
    global_ft_model = fasttext.load_model(fasttext_model_path)

@lru_cache(maxsize=500_000)
def is_english_word(word: str) -> bool:
    return zipf_frequency(word, "en") > 3.0

def filter_partial_english(text: str):
    words = str(text).split()
    if len(words) < CONFIG["min_words"]: return None
    english_words = [w for w in words if is_english_word(w)]
    if not words: return None
    ratio = len(english_words) / len(words)
    if ratio < CONFIG["min_english_ratio"]: return None
    if ratio > 0.9: return text
    return " ".join(english_words)

def generate_minhash_array(text):
    words = str(text).split()
    if len(words) < 3: return None
    m = MinHash(num_perm=CONFIG["num_perm"])
    for i in range(len(words) - 2):
        m.update(f"{words[i]} {words[i+1]} {words[i+2]}".encode("utf-8"))
    return m.hashvalues

def is_char_spam(text: str) -> bool:
    """True if fewer than 50% of the characters (excluding spaces) are alphanumeric."""
    if not text: 
        return True
        
    text_no_spaces = str(text).replace(" ", "")
    
    if len(text_no_spaces) == 0: 
        return True
        
    valid_chars = sum(c.isalnum() for c in text_no_spaces)
    
    return (valid_chars / len(text_no_spaces)) < CONFIG["min_alpha_ratio"]

def has_excessive_repetition(text: str) -> bool:
    """True if fewer than 25% of words are unique."""
    words = text.split()
    if not words:
        return True
    return (len(set(words)) / len(words)) < CONFIG["min_unique_word_ratio"]

def process_chunk_stateless(chunk):
    deleted_logs = {}

    if "sender_type" in chunk.columns:
        bot_mask = chunk["sender_type"] == "bot"
        deleted_logs["deleted_bots.csv"] = chunk[bot_mask].copy()
        chunk = chunk[~bot_mask]

    chunk = chunk.dropna(subset=["text"])

    placeholder_mask = chunk["text"].str.contains(r"^content could not be displayed$", case=False, na=False)
    deleted_logs["deleted_placeholders.csv"] = chunk[placeholder_mask].copy()
    chunk = chunk[~placeholder_mask]

    chunk["raw_md5"] = chunk["text"].apply(lambda t: hashlib.md5(str(t).encode("utf-8")).hexdigest())

    chunk["text"] = chunk["text"].astype(str)
    chunk["text"] = chunk["text"].str.replace(r"http\S+|www\S+", " ", regex=True)
    chunk["text"] = chunk["text"].str.replace(r"[\U00010000-\U0010ffff]", "", regex=True) 

    chunk["text"] = chunk["text"].str.replace(r"(?<=[a-zA-Z])\$|\$(?=[a-zA-Z])", "s", regex=True)
    
    chunk["text"] = chunk["text"].str.replace(r"(?<=[a-zA-Z])@|@(?=[a-zA-Z])", "a", regex=True)

    chunk["text"] = chunk["text"].str.replace(r"(?<=[a-zA-Z])0|0(?=[a-zA-Z])", "o", regex=True)

    chunk["text"] = chunk["text"].str.replace(r"(?<=[a-zA-Z])3|3(?=[a-zA-Z])", "e", regex=True)

    chunk["text"] = chunk["text"].str.replace(r"[^a-zA-Z0-9\s$€£.,]", " ", regex=True)
    
    chunk["text"] = chunk["text"].str.replace(r"\s+", " ", regex=True).str.strip() 

    empty_mask = chunk["text"] == ""
    deleted_logs["deleted_empty_after_norm.csv"] = chunk[empty_mask].copy()
    chunk = chunk[~empty_mask]

    too_short = chunk["text"].str.len() < 30
    deleted_logs["deleted_micro_spam.csv"] = chunk[too_short].copy()
    chunk = chunk[~too_short]

    too_long = chunk["text"].str.len() > 2500
    deleted_logs["deleted_mega_dumps.csv"] = chunk[too_long].copy()
    chunk = chunk[~too_long]

    char_spam_mask = chunk["text"].apply(is_char_spam)
    deleted_logs["deleted_char_spam.csv"] = chunk[char_spam_mask].copy()
    chunk = chunk[~char_spam_mask]

    chunk["clean_md5"] = chunk["text"].apply(lambda t: hashlib.md5(str(t).encode("utf-8")).hexdigest())

    texts = chunk["text"].tolist()
    labels, probs = global_ft_model.predict(texts)
    chunk["lang"] = [l[0] for l in labels]
    chunk["conf"] = [p[0] for p in probs]

    ft_mask = (chunk["lang"] == "__label__en") & (chunk["conf"] >= CONFIG["fasttext_min_conf"])
    deleted_logs["deleted_non_english.csv"] = chunk[~ft_mask].drop(columns=["lang", "conf"]).copy()
    chunk = chunk[ft_mask].drop(columns=["lang", "conf"])

    chunk["text"] = chunk["text"].apply(filter_partial_english)
    chunk = chunk.dropna(subset=["text"])

    repetition_mask = chunk["text"].apply(has_excessive_repetition)
    deleted_logs["deleted_repetition_spam.csv"] = chunk[repetition_mask].copy()
    chunk = chunk[~repetition_mask]

    chunk["minhash_values"] = chunk["text"].apply(generate_minhash_array)

    return chunk, deleted_logs

def log_deleted_rows(df, path):
    if df.empty: return
    mode = "a" if os.path.exists(path) else "w"
    df.to_csv(path, index=False, mode=mode, header=(mode == "w"))

class GlobalDeduplicator:
    def __init__(self):
        self.seen_raw = BloomFilter(CONFIG["bloom_capacity"], CONFIG["bloom_error"])
        self.seen_clean = BloomFilter(CONFIG["bloom_capacity"], CONFIG["bloom_error"])
        self.lsh = MinHashLSH(threshold=CONFIG["lsh_threshold"], num_perm=CONFIG["num_perm"])
        self.counter = 0
        self.md5_to_text = {} 

    def apply_bloom(self, chunk, md5_col):
        bloom_mask = chunk[md5_col].apply(lambda h: h in self.seen_raw if md5_col == 'raw_md5' else h in self.seen_clean)
        local_dupe_mask = chunk.duplicated(subset=[md5_col], keep='first')
        final_mask = bloom_mask | local_dupe_mask
        deleted = chunk[final_mask].copy()
        chunk = chunk[~final_mask]
        for h in chunk[md5_col]:
            if md5_col == 'raw_md5': self.seen_raw.add(h)
            else: self.seen_clean.add(h)
        return chunk, deleted

    def determine_change_location(self, orig_text, curr_text):
        if not orig_text or not curr_text: 
            return "Unknown"
            
        orig_words = str(orig_text).split()
        curr_words = str(curr_text).split()
        
        if len(orig_words) == 0:
            return "Unknown"

        matcher = difflib.SequenceMatcher(None, orig_words, curr_words)
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag != 'equal':
                position_ratio = i1 / len(orig_words)
                if position_ratio < 0.33:
                    return "Beginning"
                elif position_ratio > 0.66:
                    return "End"
                else:
                    return "Middle"
                    
        return "Scattered Minor Edits"

    def apply_lsh(self, chunk):
        keep_mask = []
        to_insert = []
        matched_original_ids = []
        change_locations = []
        
        for hashvals, current_msg_id, current_text in zip(chunk["minhash_values"], chunk["raw_md5"], chunk["text"]):
            if hashvals is None or not isinstance(hashvals, np.ndarray):
                keep_mask.append(False)
                matched_original_ids.append("Invalid/Short")
                change_locations.append("N/A")
                continue
            
            m = MinHash(num_perm=CONFIG["num_perm"], hashvalues=hashvals)
            matches = self.lsh.query(m)
            
            if matches:
                matched_id = matches[0]
                keep_mask.append(False)
                matched_original_ids.append(matched_id)
                
                orig_text = self.md5_to_text.get(matched_id, "")
                location = self.determine_change_location(orig_text, current_text)
                change_locations.append(location)
            else:
                keep_mask.append(True)
                matched_original_ids.append(None)
                change_locations.append(None)
                
                to_insert.append((current_msg_id, m))
                self.md5_to_text[current_msg_id] = current_text
                self.counter += 1
                
        if to_insert:
            with self.lsh.insertion_session() as session:
                for key, m in to_insert:
                    session.insert(key, m)
                    
        mask_series = pd.Series(keep_mask, index=chunk.index)
        
        deleted_chunk = chunk[~mask_series].copy()
        deleted_chunk["matched_with_original_md5"] = [m_id for msk, m_id in zip(keep_mask, matched_original_ids) if not msk]
        deleted_chunk["change_location"] = [loc for msk, loc in zip(keep_mask, change_locations) if not msk]
        
        return chunk[mask_series], deleted_chunk

def process_dataset_parallel_unordered(input_file, output_file, fasttext_model_path):
    start = time.time()
    print("Initializing Unordered Parallel Pipelie")

    log_files = ["deleted_bots.csv", "deleted_placeholders.csv", "deleted_raw_dupes.csv",
                 "deleted_clean_dupes.csv", "deleted_empty_after_norm.csv",
                 "deleted_non_english.csv", "deleted_near_dupes.csv",
                 "deleted_char_spam.csv", "deleted_repetition_spam.csv",
                 "deleted_micro_spam.csv", "deleted_mega_dumps.csv",
                 output_file]
    for f in log_files:
        if os.path.exists(f): os.remove(f)

    state_manager = GlobalDeduplicator()
    reader = pd.read_csv(input_file, chunksize=CONFIG["batch_size"], dtype=str, usecols=["text", "sender_type"])

    reader_iter = iter(reader)
    num_workers = min(6, max(1, multiprocessing.cpu_count() - 1))

    print(f"Firing up {num_workers} CPU cores for UNORDERED processing")

    with ProcessPoolExecutor(max_workers=num_workers, initializer=init_worker, initargs=(fasttext_model_path,)) as executor:

        active_futures = {}
        batch_counter = 0

        for _ in range(num_workers * 2):
            try:
                chunk = next(reader_iter)
                batch_counter += 1
                future = executor.submit(process_chunk_stateless, chunk)
                active_futures[future] = batch_counter
            except StopIteration:
                break

        first_save = True
        while active_futures:
            done, _ = concurrent.futures.wait(active_futures.keys(), return_when=concurrent.futures.FIRST_COMPLETED)

            for future in done:
                batch_num = active_futures.pop(future)
                processed_chunk, deleted_logs = future.result()

                print(f"\n Merging Batch {batch_num} into Main Process ")

                for log_name, df_del in deleted_logs.items():
                    log_deleted_rows(df_del, log_name)

                processed_chunk, raw_dupes = state_manager.apply_bloom(processed_chunk, "raw_md5")
                log_deleted_rows(raw_dupes.drop(columns=["raw_md5", "clean_md5", "minhash_values"], errors='ignore'), "deleted_raw_dupes.csv")

                processed_chunk, clean_dupes = state_manager.apply_bloom(processed_chunk, "clean_md5")
                log_deleted_rows(clean_dupes.drop(columns=["raw_md5", "clean_md5", "minhash_values"], errors='ignore'), "deleted_clean_dupes.csv")

                processed_chunk, near_dupes = state_manager.apply_lsh(processed_chunk)
                log_deleted_rows(near_dupes.drop(columns=["raw_md5", "clean_md5", "minhash_values"], errors='ignore'), "deleted_near_dupes.csv")

                processed_chunk = processed_chunk.drop(columns=["raw_md5", "clean_md5", "minhash_values"], errors='ignore')

                mode, header = ("w", True) if first_save else ("a", False)
                processed_chunk.to_csv(output_file, index=False, mode=mode, header=header)
                first_save = False

                print(f"       Batch {batch_num} Saved | Kept {len(processed_chunk):,} unique rows")
                gc.collect()

                try:
                    next_chunk = next(reader_iter)
                    batch_counter += 1
                    new_future = executor.submit(process_chunk_stateless, next_chunk)
                    active_futures[new_future] = batch_counter
                except StopIteration:
                    pass

    print(f"\nUnordered Pipeline Complete in {int((time.time() - start)//60)}m {int((time.time() - start)%60)}s")
    print(f"Final LSH Index size: {state_manager.counter:,} messages")

if __name__ == "__main__":
    stop_monitor = threading.Event()
    
    monitor_thread = threading.Thread(target=resource_monitor, args=(stop_monitor,))
    monitor_thread.start()
    
    process_dataset_parallel_unordered("unfiltered-messages.csv", "filtered_dataset.csv", "lid.176.bin")
    
    stop_monitor.set()
    monitor_thread.join()
    print("Hardware log saved to hardware_usage.csv!")