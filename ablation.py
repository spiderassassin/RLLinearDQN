import subprocess

resnet = "python agent.py real_resnet --network-type=resnet --train"
# linear = "python agent.py ablation_linear --network-type=linear --train --power"
# res_only = "python agent.py ablation_res --network-type=res_only --train --power"
# norm_only = "python agent.py ablation_norm --network-type=norm_only --train --power"
baseline = "python agent.py nonlinresnet --network-type=nonlinear --train"

subprocess.call(resnet, shell=True)
# subprocess.call(linear, shell=True)
# subprocess.call(res_only, shell=True)
# subprocess.call(norm_only, shell=True)
subprocess.call(baseline, shell=True)

