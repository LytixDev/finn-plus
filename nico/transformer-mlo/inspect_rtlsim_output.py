#!/usr/bin/env python3
# poetry run finn run inspect_rtlsim_output.py

# The rtl sim of the stitched ip that has MLO nodes does not sim the parent (the stuff not on the FPGA)
# TODO: Set up the pre processing necessary and send the correct inputs to the RTL sim 
# TODO: Perform the correct post-processing

import os
import numpy as np

from finn.util.settings import initialize_dummy_settings
initialize_dummy_settings()

from qonnx.core.modelwrapper import ModelWrapper

# NOTE: tmp path
output_dir = os.environ.get("FINN_BUILD_DIR", "/tmp") + "/transformer-mlo-test"

out_path = os.path.join(output_dir, "verification_output", "verify_stitched_ip_rtlsim_0_FAIL.npy")
exp_path = "out.npy"

out = np.load(out_path)
exp = np.load(exp_path)

parent_path = os.path.join(output_dir, "intermediate_models", "dataflow_parent.onnx")
parent = ModelWrapper(parent_path)
scale = parent.get_initializer("Mul_8_param0")
if scale is not None:
    print("Dequant scale factor:", scale.flatten()[:5], "...")
    out_scaled = out * scale
else:
    print("WARNING: Could not find scale factor")
    out_scaled = out

print("RTLsim raw output (first 20):", out.flatten()[:20])
print("RTLsim scaled output (first 20):", out_scaled.flatten()[:20])
print("Expected output (first 20):", exp.flatten()[:20])

print("Comparison with ad hoc post processing")
if out_scaled.size == exp.size:
    diff = np.abs(out_scaled.flatten() - exp.flatten())
    print("Max abs diff:", np.max(diff))
    print("Mean abs diff:", np.mean(diff))
    print("Close (atol=1e-1)?", np.allclose(out_scaled, exp, atol=1e-1))
    print("Close (atol=1)?", np.allclose(out_scaled, exp, atol=1))
    print("Fraction matching (atol=1e-1):", np.mean(np.isclose(out_scaled, exp, atol=1e-1)))
    print("Fraction matching (atol=1):", np.mean(np.isclose(out_scaled, exp, atol=1)))
else:
    print("Size mismatch:", out_scaled.size, "vs", exp.size)
