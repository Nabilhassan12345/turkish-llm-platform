import torch

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device count:", torch.cuda.device_count())
    print("Device name:", torch.cuda.get_device_name(0))
    device = torch.device("cuda")
    print("Memory allocated:", torch.cuda.memory_allocated(device) / 1024**3, "GB")
    try:
        test_tensor = torch.randn(100, 100, device=device)
        print("GPU tensor test: SUCCESS")
        del test_tensor
        torch.cuda.empty_cache()
        print("Memory cleanup: SUCCESS")
    except Exception as e:
        print("GPU test failed:", e)
else:
    print("No GPU - will use CPU offloading")
