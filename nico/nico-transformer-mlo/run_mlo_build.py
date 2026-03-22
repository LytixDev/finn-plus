# poetry run finn run run_mlo_build.py

# Model has been created using a modified version of the FINN-T repository
# Specifically, certain quantizers across transformer layers are shared so that 
# the scaled dot dot-product attention operator has constant inputs across loop iterations.
# Even more specifically, thresholds_a_softmax, thresholds_av_matmul, must be shared.
# This means the softmax output quantizer and the A*V matmul and V*activation matmul 
# nodes have shared quantizers across layers.


# TODO:
# fix the vitis hls bug?
#     - internal accumulator width is too large
#     - the folding step must have this as a contraint?
#     in SetFolding: 
#       - Add a constraint in the PE phase that ensures PE * SIMD * weight_bits <= 8191
#     - on idun, fails when target fps is 1000 but works when it is 500

import os
# it times out at 1m
os.environ["LIVENESS_THRESHOLD"] = "2000000" # 2m

from qonnx.core.modelwrapper import ModelWrapper

output_dir = os.environ.get("FINN_BUILD_DIR", "/tmp") + "/nico-transformer-mlo"
os.makedirs(output_dir, exist_ok=True)
print(f"Output dir: {output_dir}")

from finn.util.settings import initialize_dummy_settings
initialize_dummy_settings()

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg

input_npy = "inp.npy"
output_npy = "out.npy"

#print(f"Loaded model: {len(model.graph.node)} nodes")
#for i, node in enumerate(model.graph.node):
#    print(f"  [{i}] {node.op_type} ({node.name})")

steps_pre_rolling = [
    # NOTE: The commented out passes here are handled by the FINN-T frontend
    #"finn.builder.passes.export",
    #"step_qonnx_to_finn",
    #"step_tidy_up",
    #"step_streamline",
    ## Customized adhoc hardware conversion step: Includes inferring the fused
    ## operator for scaled dot-product attention
    #"finn.builder.custom_step_library.transformer_adhoc.step_convert_to_hw",

    # This particular transformer model has weights tagged as INT64 when they should be INT4 or INT8
    # This sets the appropriate dtype before folding.
    "finn.builder.custom_step_library.transformer_adhoc.step_fix_mvau_weight_dtype",

    # Default FINN partitioning and specialization steps
    "step_create_dataflow_partition",
    "step_specialize_layers",
]

steps_rolling_and_beyond = [
    "step_loop_rolling",

    # The rest here are stuff that should happen after the loop rolling

    "finn.builder.custom_step_library.transformer_adhoc.step_set_folding",

    "step_minimize_bit_width",
    "step_generate_estimate_reports",
    "step_hw_codegen",
    "step_hw_ipgen",
    "step_set_fifo_depths",
    "step_create_stitched_ip",
    #"step_measure_rtlsim_performance",
    "step_out_of_context_synthesis",
    #"step_synthesize_bitfile",
    #"step_make_driver",
    #"step_deployment_package",
]

target_fps=1_000
clk_period_ns=20.0 # 50 MHz
board="U250"
shell_flow_type="vitis_alveo"
rtl_sim_batch_size=10 # 100

cfg_pre_rolling = build_cfg.DataflowBuildConfig(
    output_dir=output_dir,
    steps=steps_pre_rolling,
    target_fps=target_fps,
    synth_clk_period_ns=clk_period_ns,
    board=board,
    shell_flow_type=shell_flow_type,
    rtlsim_batch_size=rtl_sim_batch_size,
    standalone_thresholds=True,
    specialize_layers_config_file="transformer_specialization.json",
    max_multithreshold_bit_width=16,
    mvau_wwidth_max=2048,
    split_large_fifos=True,
    auto_fifo_depths=True,
    generate_outputs=[
        build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
        build_cfg.DataflowOutputType.STITCHED_IP,
    ],
    # Uncomment to enable verification (needs cppsim reference from create_mlo_model.py):
    # verify_steps=["folded_hls_cppsim", "node_by_node_rtlsim", "stitched_ip_rtlsim"],
    verify_input_npy=input_npy,
    verify_expected_output_npy=output_npy,
)
print(f"Running steps up to loop rolling: {steps_pre_rolling}")
print(f"Intermediate models will be saved to: {output_dir}/intermediate_models/")

build.build_dataflow_cfg("step_convert_to_hw.onnx", cfg_pre_rolling)

model_path = f"{output_dir}/intermediate_models/step_specialize_layers.onnx"
model = ModelWrapper(model_path)
loop_body_range = (model.graph.node[2], model.graph.node[30])
cfg_rolling_and_beyond = build_cfg.DataflowBuildConfig(
    output_dir=output_dir,
    steps=steps_rolling_and_beyond,
    #start_step="step_out_of_context_synthesis", # TODO: TMP
    target_fps=target_fps,
    synth_clk_period_ns=clk_period_ns,
    board=board,
    shell_flow_type=shell_flow_type,
    rtlsim_batch_size=rtl_sim_batch_size,
    standalone_thresholds=True,
    specialize_layers_config_file="transformer_specialization.json",
    max_multithreshold_bit_width=16,
    mvau_wwidth_max=2048,
    split_large_fifos=True,
    auto_fifo_depths=True,
    mlo=True,
    loop_body_hierarchy=[["", "layers.0"]],
    loop_body_range=loop_body_range,
    generate_outputs=[
        build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
        build_cfg.DataflowOutputType.STITCHED_IP,
        build_cfg.DataflowOutputType.BITFILE,
        build_cfg.DataflowOutputType.RTLSIM_PERFORMANCE,
        build_cfg.DataflowOutputType.OOC_SYNTH,
    ],
    # Uncomment to enable verification (needs cppsim reference from create_mlo_model.py):
    #verify_steps=["folded_hls_cppsim", "node_by_node_rtlsim", "stitched_ip_rtlsim"],
    verify_steps=["stitched_ip_rtlsim"],
    verify_input_npy=input_npy,
    verify_expected_output_npy=output_npy,
)


print(f"Running loop rolling and everything else : {steps_rolling_and_beyond}")
print(f"Intermediate models will be saved to: {output_dir}/intermediate_models/")

build.build_dataflow_cfg(model_path, cfg_rolling_and_beyond)
