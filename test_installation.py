#!/usr/bin/env python3
"""
Test script to verify Quantum Video Trainer installation
"""

import sys
from pathlib import Path

print("=" * 60)
print("Quantum Video Trainer - Installation Test")
print("=" * 60)

# Test 1: Import core modules
print("\n[1/6] Testing core module imports...")
try:
    from core.video_loader import VideoLoader
    from core.decomposer import VideoDecomposer
    from core.knowledge_base import KnowledgeBase
    from core.incremental_trainer import IncrementalTrainer
    print("✓ Core modules imported successfully")
except Exception as e:
    print(f"✗ Failed to import core modules: {e}")
    sys.exit(1)

# Test 2: Import theory modules
print("\n[2/6] Testing theory module imports...")
try:
    from theories.musical_theories import CircleOfFifthsMapper, JazzHarmonyMapper
    from theories.physics_theories import PhysicsMapper
    print("✓ Theory modules imported successfully")
except Exception as e:
    print(f"✗ Failed to import theory modules: {e}")
    sys.exit(1)

# Test 3: Import VFE modules
print("\n[3/6] Testing VFE module imports...")
try:
    from pmc_vfe.vfe_gaussian import vfe_gaussian
    from pmc_vfe.vfe_categorical import vfe_categorical
    from pmc_vfe.efe import expected_free_energy
    from pmc_vfe.strategy_scorer import score_batch, StrategyInput
    print("✓ VFE modules imported successfully")
except Exception as e:
    print(f"✗ Failed to import VFE modules: {e}")
    sys.exit(1)

# Test 4: Import ComfyUI nodes
print("\n[4/6] Testing ComfyUI node imports...")
try:
    from nodes.input_nodes import QVT_LoadVideo, QVT_LoadVideoBatch, QVT_LoadKnowledgeBase
    from nodes.decomposition_nodes import QVT_DecomposeVideo
    from nodes.theory_nodes import QVT_MapToCircleOfFifths, QVT_MapToPhysics
    from nodes.vfe_nodes import QVT_ComputeVFE_Gaussian, QVT_ComputeVFE_Categorical
    from nodes.training_nodes import QVT_IncrementalTrain
    from nodes.state_nodes import QVT_SaveKnowledgeBase
    print("✓ ComfyUI nodes imported successfully")
except Exception as e:
    print(f"✗ Failed to import ComfyUI nodes: {e}")
    sys.exit(1)

# Test 5: Test knowledge base creation
print("\n[5/6] Testing knowledge base creation...")
try:
    kb = KnowledgeBase(save_path='./test_kb.pkl')
    kb.initialize()
    assert kb.state['total_videos'] == 0
    assert 'theory_weights' in kb.state
    assert 'vfe_priors_gaussian' in kb.state
    assert 'vfe_priors_categorical' in kb.state
    print("✓ Knowledge base creation successful")

    # Clean up
    if Path('./test_kb.pkl').exists():
        Path('./test_kb.pkl').unlink()
except Exception as e:
    print(f"✗ Failed to create knowledge base: {e}")
    sys.exit(1)

# Test 6: Test VFE computation
print("\n[6/6] Testing VFE computation...")
try:
    import numpy as np

    # Gaussian VFE test
    mu_q = np.array([0.5])
    Sigma_q = np.array([[0.1]])
    mu_p = np.array([0.7])
    Sigma_p = np.array([[0.2]])

    vfe_val = vfe_gaussian(mu_q, Sigma_q, mu_p, Sigma_p)
    assert isinstance(vfe_val, float)
    assert vfe_val >= 0  # VFE should be non-negative

    # Categorical VFE test
    q_h = np.array([0.6, 0.3, 0.1])
    p_h = np.array([0.5, 0.3, 0.2])

    vfe_cat = vfe_categorical(q_h, p_h)
    assert isinstance(vfe_cat, float)

    print("✓ VFE computation successful")
except Exception as e:
    print(f"✗ Failed VFE computation: {e}")
    sys.exit(1)

# All tests passed
print("\n" + "=" * 60)
print("✓ ALL TESTS PASSED")
print("=" * 60)
print("\nQuantum Video Trainer is properly installed!")
print("\nNext steps:")
print("  1. Prepare test videos in ./test_videos/")
print("  2. Load ComfyUI and look for QuantumVideoTrainer nodes")
print("  3. Create a training workflow")
print("  4. Run your first training session!")
print("\n" + "=" * 60)
