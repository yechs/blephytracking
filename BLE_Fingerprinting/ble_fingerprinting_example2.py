import time

from ble_fingerprinting import run_example


def main():
    start = time.time()
    fingerprints = run_example("Example_Data")
    elapsed = time.time() - start
    print("fingerprint matrix shape:", fingerprints.shape)
    print("elapsed seconds:", elapsed)

    # print fingerprints matrix
    print("fingerprints:")
    print(fingerprints)


if __name__ == "__main__":
    main()
