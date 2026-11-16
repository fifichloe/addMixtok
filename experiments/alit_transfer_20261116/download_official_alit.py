"""Resume the official ALIT ImageNet checkpoint download by verified byte ranges."""

import concurrent.futures
import json
import os
from pathlib import Path
import threading
import time

import requests

URL = ('https://www.dropbox.com/scl/fi/u8jyro5wysttp5mvs6gir/'
       'alit_vqgan_small_quantized_latent.pth?'
       'rlkey=1py1s9755gjhdyw5ifd9h2wff&st=swdbevjk&dl=1')
ROOT = Path('/tmp/mixtok_alit20_20261116/official')
FINAL = ROOT / 'alit_vqgan_small_quantized_latent.pth'
PART = ROOT / 'alit_vqgan_small_quantized_latent.pth.ranges.part'
RECEIPT = ROOT / 'alit_vqgan_small_quantized_latent.ranges.json'
SIZE = 1607965488
CHUNK = 8 << 20
COUNT = (SIZE + CHUNK - 1) // CHUNK
lock = threading.Lock()


def save(done):
    temp = RECEIPT.with_suffix('.json.tmp')
    temp.write_text(json.dumps({'size': SIZE, 'chunk': CHUNK, 'done': sorted(done)}))
    temp.replace(RECEIPT)


def fetch(index):
    start = index * CHUNK
    end = min(start + CHUNK, SIZE) - 1
    expected = end - start + 1
    for attempt in range(12):
        try:
            response = requests.get(URL, headers={'Range': f'bytes={start}-{end}'},
                                    stream=True, timeout=(30, 75))
            response.raise_for_status()
            assert response.status_code == 206, response.status_code
            assert response.headers.get('Content-Range') == f'bytes {start}-{end}/{SIZE}'
            data = bytearray()
            for block in response.iter_content(1 << 20):
                if block:
                    data.extend(block)
            response.close()
            assert len(data) == expected, (index, len(data), expected)
            return index, data
        except (requests.RequestException, AssertionError) as error:
            print(json.dumps({'chunk': index, 'attempt': attempt+1,
                              'error': repr(error)}), flush=True)
            time.sleep(min(30, 2*(attempt+1)))
    raise RuntimeError(f'chunk {index} exhausted retries')


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    assert not FINAL.exists() or FINAL.stat().st_size < SIZE
    if RECEIPT.exists():
        receipt = json.loads(RECEIPT.read_text())
        assert receipt['size'] == SIZE and receipt['chunk'] == CHUNK
        done = set(receipt['done'])
    else:
        assert not PART.exists()
        done = set()
    fd = os.open(PART, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        os.ftruncate(fd, SIZE)
        pending = [i for i in range(COUNT) if i not in done]
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            source = iter(pending)
            active = {pool.submit(fetch, next(source)) for _ in range(min(4, len(pending)))}
            while active:
                completed, active = concurrent.futures.wait(
                    active, return_when=concurrent.futures.FIRST_COMPLETED)
                for future in completed:
                    index, data = future.result()
                    try:
                        active.add(pool.submit(fetch, next(source)))
                    except StopIteration:
                        pass
                    assert os.pwrite(fd, data, index*CHUNK) == len(data)
                    with lock:
                        done.add(index)
                        save(done)
                        if len(done) % 8 == 0 or len(done) == COUNT:
                            print(json.dumps({'completed_chunks': len(done), 'total_chunks': COUNT,
                                              'bytes': min(len(done)*CHUNK, SIZE)}), flush=True)
        assert len(done) == COUNT
        os.fsync(fd)
    finally:
        os.close(fd)
    PART.replace(FINAL)
    assert FINAL.stat().st_size == SIZE
    print(json.dumps({'status': 'complete', 'path': str(FINAL), 'bytes': SIZE}), flush=True)


if __name__ == '__main__':
    main()
