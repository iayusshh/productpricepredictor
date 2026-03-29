"""
Quick test to verify the 8B model is ready for training
Tests imports, model creation, and basic functionality
"""

import sys
from pathlib import Path

def test_imports():
    """Test if all required packages are available"""
    print("=" * 70)
    print("🔍 TESTING IMPORTS")
    print("=" * 70)
    
    required_packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'PIL': 'Pillow',
        'pandas': 'Pandas',
        'numpy': 'NumPy',
        'tqdm': 'tqdm'
    }
    
    missing = []
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"✅ {name} - OK")
        except ImportError:
            print(f"❌ {name} - MISSING")
            missing.append(name)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("Install with: pip install torch torchvision pillow pandas numpy tqdm")
        return False
    
    print("\n✅ All required packages installed!")
    return True


def test_model_import():
    """Test if the 8B model can be imported"""
    print("\n" + "=" * 70)
    print("🔍 TESTING MODEL IMPORT")
    print("=" * 70)
    
    try:
        from src.models import LargeMultimodalModelWrapper
        print("✅ LargeMultimodalModelWrapper imported successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to import model: {str(e)}")
        return False


def test_model_creation():
    """Test if the model can be created"""
    print("\n" + "=" * 70)
    print("🔍 TESTING MODEL CREATION")
    print("=" * 70)
    
    try:
        from src.models import LargeMultimodalModelWrapper
        
        # Create model with small vocab for testing
        model = LargeMultimodalModelWrapper(
            vocab_size=1000,
            max_seq_len=128,
            device='cpu'  # Use CPU for testing
        )
        
        param_count = model.model.count_parameters()
        print(f"✅ Model created successfully")
        print(f"   Parameters: {param_count:,} ({param_count/1e9:.2f}B)")
        print(f"   Device: {model.device}")
        
        return True
    except Exception as e:
        print(f"❌ Failed to create model: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_data_files():
    """Test if required data files exist"""
    print("\n" + "=" * 70)
    print("🔍 TESTING DATA FILES")
    print("=" * 70)
    
    required_files = {
        'dataset/train.csv': 'Training data',
        'dataset/test.csv': 'Test data',
    }
    
    missing = []
    for filepath, description in required_files.items():
        if Path(filepath).exists():
            print(f"✅ {description} - {filepath}")
        else:
            print(f"❌ {description} - {filepath} NOT FOUND")
            missing.append(filepath)
    
    # Check if images directory exists
    if Path('images/train').exists():
        print(f"✅ Training images directory - images/train")
    else:
        print(f"⚠️  Training images directory - images/train NOT FOUND")
        print("   Run image download first!")
    
    if missing:
        print(f"\n❌ Missing files: {', '.join(missing)}")
        return False
    
    print("\n✅ All required data files found!")
    return True


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("🚀 PRE-TRAINING VERIFICATION")
    print("=" * 70)
    
    tests = [
        ("Imports", test_imports),
        ("Model Import", test_model_import),
        ("Model Creation", test_model_creation),
        ("Data Files", test_data_files),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ {test_name} test crashed: {str(e)}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 TEST SUMMARY")
    print("=" * 70)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n" + "=" * 70)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 70)
        print("\n✅ Your model is ready for training!")
        print("\nNext steps:")
        print("1. Download images: python src/utils.py")
        print("2. Train model: python3 train_8b_model_simple.py")
        return True
    else:
        print("\n" + "=" * 70)
        print("❌ SOME TESTS FAILED")
        print("=" * 70)
        print("\nPlease fix the issues above before training.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
