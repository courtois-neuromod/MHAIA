#!/bin/bash
# Verification script for MHAIA CL setup

echo "=== MHAIA CL Setup Verification ==="
echo ""

# Check if env exists
if [ ! -d "env" ]; then
    echo "❌ Virtual environment not found. Run:"
    echo "   python -m venv env"
    echo "   source env/bin/activate"
    echo "   pip install -e '.[cl]'"
    exit 1
fi

# Activate environment
source env/bin/activate 2>/dev/null || {
    echo "❌ Failed to activate environment"
    exit 1
}

echo "✓ Virtual environment activated"

# Check Python version
PYTHON_VERSION=$(python --version 2>&1)
echo "✓ $PYTHON_VERSION"

# Check TensorFlow
if python -c "import tensorflow" 2>/dev/null; then
    TF_VERSION=$(python -c "import tensorflow as tf; print(tf.__version__)" 2>/dev/null | grep -v "tensorflow")
    echo "✓ TensorFlow $TF_VERSION"
else
    echo "❌ TensorFlow not installed. Run:"
    echo "   pip install -e '.[cl]'"
    exit 1
fi

# Check stable-retro
if python -c "import retro" 2>/dev/null; then
    RETRO_VERSION=$(python -c "import retro; print(retro.__version__)" 2>/dev/null)
    echo "✓ stable-retro $RETRO_VERSION"
else
    echo "❌ stable-retro not installed. Run:"
    echo "   pip install -e '.[cl]'"
    exit 1
fi

# Check MHAIA
if python -c "import MHAIA" 2>/dev/null; then
    echo "✓ MHAIA installed"
else
    echo "❌ MHAIA not installed. Run:"
    echo "   pip install -e '.[cl]'"
    exit 1
fi

# Check ROM
echo ""
echo "Checking for Super Mario Bros ROM..."
if python -c "import retro; retro.data.get_romfile_path('SuperMarioBros-Nes', 'Level1-1')" 2>/dev/null; then
    echo "✓ Super Mario Bros ROM found"
else
    echo "⚠️  Super Mario Bros ROM not found"
    echo "   Import with: python -m retro.import /path/to/roms/"
fi

# Check config files
echo ""
echo "Checking configuration files..."
[ -f "config.py" ] && echo "✓ config.py" || echo "❌ config.py missing"
[ -f "QUICKSTART_CL.md" ] && echo "✓ QUICKSTART_CL.md" || echo "⚠️  QUICKSTART_CL.md missing"
[ -f "CL_MIGRATION_SUMMARY.md" ] && echo "✓ CL_MIGRATION_SUMMARY.md" || echo "⚠️  CL_MIGRATION_SUMMARY.md missing"

echo ""
echo "=== Verification Complete ==="
echo ""
echo "To run a quick test:"
echo "  source env/bin/activate"
echo "  python -m CL.run_cl --sequence WORLD_PROGRESSION_4 --steps_per_env 100 --no_test --multihead_archs False"
echo ""
echo "For more info, see QUICKSTART_CL.md"
