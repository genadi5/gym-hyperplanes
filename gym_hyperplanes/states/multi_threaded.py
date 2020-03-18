import concurrent.futures

pool_executor = concurrent.futures.ThreadPoolExecutor(max_workers=100)


def create_key(row, index):
    return index, ''.join(row.astype(str))


start_keys = round(time.time())
future_to_index = [pool_executor.submit(create_key, side, i) for i, side in enumerate(sides)]
self.total_keys += (round(time.time()) - start_keys)

for future in concurrent.futures.as_completed(future_to_index):
    i = future.result()[0]
    key = future.result()[1]
