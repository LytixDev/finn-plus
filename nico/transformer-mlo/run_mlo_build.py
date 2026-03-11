# poetry run finn run run_mlo_build.py
# Two-step process
# First we run up until loop_rolling. Then we contunie from loop_rolling. 
# This is necessary because the inputs to the build function needs the loop_body_range,
# but these nodes are only available after running all transformations up until loop_rolling.

# The model used here is from finn-transformers/language/

import os

from qonnx.core.modelwrapper import ModelWrapper

output_dir = os.environ.get("FINN_BUILD_DIR", "/tmp") + "/transformer-mlo"
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
    "finn.builder.passes.export",
    "step_qonnx_to_finn",
    "step_tidy_up",
    "step_streamline",
    # Customized adhoc hardware conversion step: Includes inferring the fused
    # operator for scaled dot-product attention
    "finn.builder.custom_step_library.transformer_adhoc.step_convert_to_hw",

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
    "step_measure_rtlsim_performance",
    "step_out_of_context_synthesis",
    "step_synthesize_bitfile",
    #"step_make_driver",
    #"step_deployment_package",
]

skip_after_minimize = False
if skip_after_minimize:
    steps_rolling_and_beyond = steps_rolling_and_beyond[:steps_rolling_and_beyond.index("step_minimize_bit_width") + 1]


target_fps=1_000
clk_period_ns=10.0
board="U250"
rtl_sim_batch_size=10, # 100

cfg_pre_rolling = build_cfg.DataflowBuildConfig(
    output_dir=output_dir,
    steps=steps_pre_rolling,
    target_fps=target_fps,
    synth_clk_period_ns=clk_period_ns,
    board=board,
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

build.build_dataflow_cfg("streamlined.onnx", cfg_pre_rolling)

model_path = f"{output_dir}/intermediate_models/step_specialize_layers.onnx"
model = ModelWrapper(model_path)
loop_body_range = (model.graph.node[4], model.graph.node[33])
cfg_rolling_and_beyond = build_cfg.DataflowBuildConfig(
    output_dir=output_dir,
    steps=steps_rolling_and_beyond,
    target_fps=target_fps,
    synth_clk_period_ns=clk_period_ns,
    board=board,
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
