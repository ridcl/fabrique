import os
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from google.cloud import storage


def download_gcs_folder(url: str, local_dir="./downloads", max_workers=8):
    parsed_url = urlparse(url)
    assert parsed_url.scheme == "gs", "Cannot download from URL with schema {parsed_url.scheme}"
    bucket_name, prefix = parsed_url.netloc, parsed_url.path.lstrip("/")
    client = storage.Client.create_anonymous_client()
    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))

    # Filter out directory entries
    files_to_download = [blob for blob in blobs if not blob.name.endswith('/')]

    os.makedirs(local_dir, exist_ok=True)
    # Track progress with thread-safe counter
    download_lock = Lock()
    downloaded_files = []

    def download_blob(blob):
        """Download a single blob"""
        try:
            relative_path = blob.name[len(prefix):].lstrip('/')
            local_file_path = os.path.join(local_dir, relative_path)
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            blob.download_to_filename(local_file_path)
            # Thread-safe progress update
            with download_lock:
                downloaded_files.append(local_file_path)
                print(f"[{len(downloaded_files)}/{len(files_to_download)}] Downloaded: {blob.name}")
            return local_file_path, None
        except Exception as e:
            return blob.name, str(e)

    # Download files in parallel
    print(f"\nStarting parallel download with {max_workers} workers...\n")

    failed_downloads = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all download tasks
        future_to_blob = {executor.submit(download_blob, blob): blob for blob in files_to_download}
        # Process completed downloads
        for future in as_completed(future_to_blob):
            result, error = future.result()
            if error:
                failed_downloads.append((result, error))

    # Print summary
    print(f"\n{'='*60}")
    print(f"Download complete!")
    print(f"Successfully downloaded: {len(downloaded_files)}/{len(files_to_download)} files")
    print(f"Destination: {local_dir}")

    if failed_downloads:
        print(f"\nFailed downloads ({len(failed_downloads)}):")
        for file_name, error in failed_downloads:
            print(f"  - {file_name}: {error}")

    return downloaded_files