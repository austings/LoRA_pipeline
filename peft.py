import argparse
import os
import statistics
from time import time
import torch
import torchaudio
from inference import TextToSpeech, MODELS_DIR

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--i',  type=str, help='Selects the onnx file name to use for generation.(.onnx)', default='myonnx')
    parser.add_argument('--o',  type=str, help='Selects the output name to use for generation. (.wav)', default='myonnx')
    parser.add_argument('--l',  type=str, help='Selects the path to latent .pth to use for generation. ~/cur_Dir/myonnx/(your path).pth ', default='dt_latent')
    parser.add_argument('--cp', type=str, help='Selects the path to ar checkpoint w/ deepspeed  .pth to use for generation. ~/cur_Dir/myonnx/(your path).pth ', default='dt_ar_checkpoint_ds')
    parser.add_argument('--ar', type=str, help='Selects the path to autoregressive .pth to use for generation. ~/cur_Dir/myonnx/(your path).pth ', default='dt_ar')
    parser.add_argument('--t' , type=str, help='text to input', default='Moby Dick is a great book by a wonderful author and everyone should read it, or else.')
  
    args = parser.parse_args()
    torch.set_float32_matmul_precision('high')
    torch.set_num_threads(4)
    os.makedirs('results', exist_ok=True)
    tts = TextToSpeech()


    sentences = ["Call me Ishmael.",
                    "Some years ago—never mind how long precisely—having little or no money in my purse, and nothing particular to interest me on shore, I thought I would sail about a little and see the watery part of the world.",
                    "It is a way I have of driving off the spleen and regulating the circulation.",
                    "Whenever I find myself growing grim about the mouth; whenever it is a damp, drizzly November in my soul",
                    "whenever I find myself involuntarily pausing before coffin warehouses, and bringing up the rear of every funeral I meet",
                    "and especially whenever my hypos get such an upper hand of me, that it requires a strong moral principle to prevent me from deliberately stepping into the street, and methodically knocking people’s hats off",
                    "then, I account it high time to get to sea as soon as I can.",
                    "This is my substitute for pistol and ball.",
                    "With a philosophical flourish Cato throws himself upon his sword; I quietly take to the ship.",
                    "There is nothing surprising in this.",
                    "If they but knew it, almost all men in their degree, some time or other, cherish very nearly the same feelings towards the ocean with me."]

    # Initialize an empty list to store average values
    average_values = []
    gen = tts.tts ("This is a warm up.")
    gen = tts.tts ("This is a second warm up.")
    startt_time = time()
    for i in sentences:
        start_time = time()
        print(i)
        gen = tts.tts(i)
        average = time() - start_time
        average_values.append(average)

    final_time = time() - startt_time
    print(final_time)
    
    torchaudio.save(os.path.join('results', args.o +'.wav'), gen.squeeze(0).cpu(), 24000)
    mean_average = statistics.mean(average_values)
    print('average time: ', mean_average)