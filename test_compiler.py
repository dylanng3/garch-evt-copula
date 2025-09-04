"""
Test PyTensor compiler detection with MinGW g++
"""
import os

# Set MinGW g++ compiler path
mingw_gpp = r"C:\msys64\mingw64\bin\g++.exe"
if os.path.exists(mingw_gpp):
    os.environ["PYTENSOR_FLAGS"] = f"cxx={mingw_gpp}"
    os.environ["PATH"] = "C:\msys64\mingw64\\bin;" + os.environ.get("PATH", "")
    print(f"Set PYTENSOR_FLAGS to use: {mingw_gpp}")
else:
    print("MinGW g++ compiler not found!")

# Try importing PyTensor
try:
    import pytensor
    print("PyTensor imported successfully")
    
    # Check config
    import pytensor.configdefaults
    print("PyTensor config loaded")
    
except Exception as e:
    print(f"Error importing PyTensor: {e}")
