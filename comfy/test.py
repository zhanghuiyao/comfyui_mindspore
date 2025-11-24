import utils

ckpt_path = ""

sd, metadata = uitls.load_mindspore_file(ckpt_path, return_metadata=True)

print(sd)
