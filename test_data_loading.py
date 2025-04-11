import logging
import sys

# Ensure the src directory is in the Python path
sys.path.insert(0, './src')

from data_utils import load_reddit_self_disclosure_dataset, DatasetDict

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting dataset loading test...")
    
    try:
        # --- Configuration --- 
        # Set cache_dir to your preferred cache location if needed
        cache_dir = None 
        # Set data_dir if you want to load from/save to a specific local path
        # If data_dir exists and contains the *raw* dataset (with 'text' column),
        # it will load from there. Otherwise, it loads from HF Hub and saves to data_dir.
        data_dir = "./data/reddit_self_disclosure_raw" # Example path
        # Set your Hugging Face token if the dataset is private (not needed for this one)
        hf_token = None 
        # ---------------------

        logger.info(f"Loading dataset... Cache: {cache_dir}, Local Raw Dir: {data_dir}")
        dataset = load_reddit_self_disclosure_dataset(
            token=hf_token,
            cache_dir=cache_dir,
            data_dir=data_dir
        )
        logger.info("Dataset loaded successfully.")

        # --- Basic Checks --- 
        assert isinstance(dataset, DatasetDict), f"Expected DatasetDict, but got {type(dataset)}"
        logger.info(f"Dataset is an instance of DatasetDict: OK")

        expected_splits = ["train", "validation", "test"]
        assert all(split in dataset for split in expected_splits), \
               f"Dataset missing splits. Found: {list(dataset.keys())}, Expected: {expected_splits}"
        logger.info(f"Found expected splits ({', '.join(expected_splits)}): OK")

        for split_name, split_data in dataset.items():
            logger.info(f"Split '{split_name}': {len(split_data)} examples found.")
            assert len(split_data) > 0, f"Split '{split_name}' should not be empty."
        
        # --- Inspect First Example --- 
        logger.info("\nInspecting the first example from the 'train' split:")
        first_train_example = dataset["train"][0]
        
        assert "tokens" in first_train_example, "'tokens' field missing in the first train example."
        assert "tags" in first_train_example, "'tags' field missing in the first train example."
        logger.info(f"First example contains 'tokens' and 'tags' fields: OK")

        tokens = first_train_example["tokens"]
        tags = first_train_example["tags"]

        assert isinstance(tokens, list), f"'tokens' should be a list, but got {type(tokens)}"
        assert isinstance(tags, list), f"'tags' should be a list, but got {type(tags)}"
        logger.info(f"'tokens' and 'tags' are lists: OK")

        assert len(tokens) > 1, f"Expected more than one token in the first example, found {len(tokens)}. This might indicate the old line-by-line processing."
        logger.info(f"First example has multiple tokens ({len(tokens)}): OK (Indicates correct sentence/document parsing)")
        
        assert len(tokens) == len(tags), f"Number of tokens ({len(tokens)}) does not match number of tags ({len(tags)}) in the first example."
        logger.info(f"Number of tokens matches number of tags ({len(tokens)}): OK")

        logger.info("\n--- First Training Example --- ")
        # Limit printing for very long examples
        max_print_len = 100 
        print(f"Tokens ({len(tokens)}): {tokens[:max_print_len]}{'...' if len(tokens) > max_print_len else ''}")
        print(f"Tags   ({len(tags)}): {tags[:max_print_len]}{'...' if len(tags) > max_print_len else ''}")
        logger.info("-----------------------------")

        logger.info("\nDataset loading test completed successfully!")

    except Exception as e:
        logger.error(f"Dataset loading test failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 