# Upload Instructions for Rory's Processed Documents

## Files Created:
- `rory_documents_processed.json` (299.4 KB)

## Upload to Cloud Server:
1. Upload `rory_documents_processed.json` to your cloud server at:
   **`/root/processed_data/rory_documents_processed.json`**

2. Use SCP command:
   ```bash
   scp rory_documents_processed.json root@1.32.228.33:/root/processed_data/
   ```

3. Or use any file transfer method to place the file at:
   `/root/processed_data/rory_documents_processed.json`

## What's in the file:
- All 3 documents processed and chunked
- Hugging Face embeddings for each chunk
- Metadata for each document and chunk
- Ready for ChromaDB import on cloud

## Next Steps:
After uploading, run the cloud import script to load into ChromaDB.
