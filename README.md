# qwen3.c

`qwen3.c` is a minimal, single-file C implementation of Qwen3 model inference, designed to run without any dependencies and inherited from llama2.c. It directly loads GGUF-format tensors without any conversion, making it self-contained.

Just for clarity, vocab and merges are read from .txt files, even though the GGUF  already includes binary tokenizer. The overhead from tokenization and detokenization is negligible compared to the forward pass, so it has little to no effect on TTS. This implementation supports multi-turn conversation, but it uses a naive full-token pass, which causes TTFT to increase as the number of tokens grows at the moment. With OpenMP enabled in a single-turn conversation, TPS stays decent. 

The C code runs the `0.6B Qwen3` model in full precision for simplicity. Since GGUF models are quantized to 8-bit or lower, you should use the FP32 version by cloning from HF, or you can convert from BF16 yourself via the conversion script in this repo. I tweaked it to ensure the layers are sorted in consecutive numerical order, since memory mapping in C jumps block by block.



## Quick Start

```sh
# Clone this repo
git clone https://github.com/gigit0000/qwen3.c.git
cd qwen3.c

# Download FP32 model from Hugging Face
git clone https://huggingface.co/huggit0000/Qwen3-0.6B-GGUF-FP32
mv Qwen3-0.6B-GGUF-FP32/Qwen3-0.6B-FP32.gguf ./

# Compile and run
make run
./run Qwen3-0.6B-FP32.gguf
```

## Faster Inference
Use OpenMP and set threads:
```sh
# Compile and run
make runomp
OMP_NUM_THREADS=16 ./run Qwen3-0.6B-FP32.gguf  # the number of your cores
```

You can enable reasoning (-k 0) or multi-turn (-m 0):
```
./run Qwen3-0.6B-FP32.gguf -k 0 -m 0 
```
## Description

If you want to extract text files (vocab.txt, merges.txt and header.txt) on your own, you can use the scripts:
```sh
# tokenizer - vocab.txt and merges.txt
python extract_v_m.py Qwen3-0.6B-FP32.gguf

```

### Inference Examples

Multi-turn Conversation
```
# OMP_NUM_THREADS=16 ./runb Qwen3-0.6B-FP32.gguf -m 0 -k 1
Multi-turn = on, thinKing = off, Temperature = 0.60, top-P = 0.95
Strike Enter when you wanna exit this chat
Enter system prompt (or Enter to skip): Tell me in one sentence
Q: Where is the best spot in Paris?
A: The best spot in Paris is the Eiffel Tower.
Q: What about the second-best spot?
A: The second-best spot in Paris is the Louvre Museum.
```

Reasoning
```
# OMP_NUM_THREADS=16 ./run Qwen3-0.6B-FP32.gguf -k 0
Multi-turn = off, thinKing = on, Temperature = 0.60, top-P = 0.95
Strike Enter when you wanna exit this chat
Enter system prompt (or Enter to skip): 
Q: Why do stars shine? Give me a quick answer!
A: <think>
Okay, the user is asking why stars shine. Let me start by recalling what I know about stars. Stars are luminous objects that emit light. So, the main reason they shine is because they produce light through nuclear fusion.

Wait, but I should make sure. Stars form from clouds of gas and dust in space. When these clouds cool, they start fusing hydrogen into helium, which releases energy. This energy is what we see as light. So the process is nuclear fusion of hydrogen into helium, which gives off energy.

I should also mention that the energy from stars is what we perceive as light. Maybe add that this light travels through space and we see it on Earth. But the question is why they shine, so the answer should focus on the energy production.

I need to keep it simple and concise. The user probably wants a quick answer, so no need for too much detail. Let me check if there's any other reason, but I think that's the main one. Alright, I think that's it.
</think>

Stars shine because they produce light through nuclear fusion of hydrogen into helium in their cores. This energy is then released as visible light, giving them their luminous glow.
```

## (Probable) To Do
- [ ] Quantized versions
- [ ] CUDA version
- [ ] Accelerated versions

## Acknoledgement
- Inspired and baselined from Andrej Kapathy's [llama2.c](https://github.com/karpathy/llama2.c)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- FGPF

## License
MIT






