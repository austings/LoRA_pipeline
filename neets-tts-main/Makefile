train_gpt:
	accelerate launch src/neets_tts/trainers/tortoise/gpt.py

train6:
	make train_gpt > train6.log 2>&1

train_decoder_1:
	accelerate launch src/neets_tts/trainers/tortoise/dumb_decoder.py

train_decoder_2:
	accelerate launch src/neets_tts/trainers/tortoise/decoder.py

train_hifigan:
	accelerate launch src/neets_tts/trainers/tortoise/hifigan.py

# > output.log 2>&1
# WANDB_MODE=disabled
