import cupy as cp

def main():
    # Check CuPy + CUDA
    print("CuPy version:", cp.__version__)
    print("CUDA available:", cp.cuda.runtime.getDeviceCount() > 0)

    # Select device 0
    cp.cuda.Device(0).use()
    print("Using device:", cp.cuda.runtime.getDeviceProperties(0)["name"])

    # Simple GPU array test
    x = cp.arange(10, dtype=cp.float32)
    y = cp.ones_like(x)

    z = x + y

    print("x:", x)
    print("y:", y)
    print("z = x + y:", z)

    # GPU → Host transfer
    print("z (host):", z.get())

    # Simple kernel: square elements
    @cp.fuse()
    def square(a):
        return a * a

    s = square(z)
    print("square(z):", s)

if __name__ == "__main__":
    main()

