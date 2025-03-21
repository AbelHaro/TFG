import tensorrt as trt


def check_dla_availability():
    # Create a logger
    logger = trt.Logger(trt.Logger.INFO)

    # Create a builder object
    builder = trt.Builder(logger)

    # Check DLA attributes
    num_dla_cores = builder.num_DLA_cores
    max_dla_batch_size = builder.max_DLA_batch_size

    if num_dla_cores > 0:
        print(f"DLA is available with {num_dla_cores} cores.")
        print(f"Maximum DLA batch size: {max_dla_batch_size}")
    else:
        print("DLA is not available on this device.")


if __name__ == "__main__":
    check_dla_availability()
